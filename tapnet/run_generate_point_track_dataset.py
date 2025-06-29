import os
import mediapy as media
import numpy as np
import shutil
import time
from fractions import Fraction
from datetime import timedelta

from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tapnet.torch import tapir_model

from datasets import load_dataset, load_from_disk, Video, Dataset, DatasetDict, concatenate_datasets, Array
from datasets.arrow_writer import ArrowWriter

from accelerate import Accelerator, InitProcessGroupKwargs
from video_utils import camera_motion_detection, crop_still_edges

import av
av.logging.set_level(av.logging.ERROR)
import logging
logging.disable(logging.CRITICAL)

from filtering_utils import topk_track_movement
import wandb

import psutil

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

def parse_args():
	parser = ArgumentParser()
	
	parser.add_argument("--input_dataset", type=str, default="")
	parser.add_argument("--model_checkpoint", type=str, default="colabs/tapnet/checkpoints/bootstapir_checkpoint_v2.pt")
	parser.add_argument("--num_points", type=int, default=1000)
	parser.add_argument("--resize_height", type=int, default=256)
	parser.add_argument("--resize_width", type=int, default=256)
	parser.add_argument("--target_frame_rate", type=int, default=None)
	parser.add_argument("--dataset_save_name", type=str, default=None)
	parser.add_argument("--user_upload_name", type=str, default="")
	parser.add_argument("--backward_consistency_check", action="store_true", default=False)
	parser.add_argument("--grayscale_consistency_check", action="store_true", default=False)
	parser.add_argument("--arrow_dir", type=str, default="arrow_data")
	parser.add_argument("--writer_batch_size", type=int, default=16)
	parser.add_argument("--save_to_disk_dir", type=str, default=None)
	parser.add_argument("--max_frames", type=int, default=None)
	parser.add_argument("--max_samples_per_split", type=int, default=None)
	parser.add_argument("--crop_still_edges", action="store_true", default=False)
	parser.add_argument("--camera_motion_detection", action="store_true", default=False)
	parser.add_argument("--min_width_height", type=int, default=64)
	parser.add_argument("--max_distortion", type=float, default=2.0)
	parser.add_argument("--local_dataset", action="store_true", default=False)
	parser.add_argument("--keep_arrow_files", action="store_true", default=False)
	parser.add_argument("--dataloader_num_workers", type=int, default=0)
	parser.add_argument("--depthcrafter_unet_path", type=str, default=None)
	parser.add_argument("--depthcrafter_pre_train_path", type=str, default=None)
	parser.add_argument("--num_shards", type=int, default=None)
	parser.add_argument("--shard_index", type=int, default=None)
	parser.add_argument("--height_num_points", type=int, default=None)
	parser.add_argument("--width_num_points", type=int, default=None)
	parser.add_argument("--fps_force", type=float, default=None)
	parser.add_argument("--raise_exceptions", action="store_true", default=False)
	parser.add_argument("--gather_shards_only", action="store_true", default=False)
	parser.add_argument("--continual_save_retry", action="store_true", default=False)
	parser.add_argument("--continual_load_retry", action="store_true", default=False)

	args = parser.parse_args()
	return args

def preprocess_frames(frames):
	"""Preprocess frames to model inputs.

	Args:
	frames: [num_frames, height, width, 3], [0, 255], np.uint8

	Returns:
		frames: [num_frames, height, width, 3], [-1, 1], np.float32
	"""
	frames = frames.float()
	frames = frames / 255 * 2 - 1
	return frames

def sample_random_points(frame_max_idx, height, width, num_points):
	"""Sample random points with (time, height, width) order."""
	y = np.random.randint(0, height, (num_points, 1))
	x = np.random.randint(0, width, (num_points, 1))
	t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
	points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
	return points

def sample_grid_points(height, width, height_num_points, width_num_points):
	"""Sample grid points with (time, height, width) order."""
	height_offset = int(height / height_num_points / 2)
	width_offset = int(width / width_num_points / 2)
	
	y = np.linspace(height_offset, height - height_offset, height_num_points).repeat(width_num_points).astype(int)
	x = np.tile(np.linspace(width_offset, width - width_offset, width_num_points), height_num_points).astype(int)
	t = np.zeros_like(y)
	points = np.stack((t, y, x), axis=-1)
	return points

def postprocess_occlusions(occlusions, expected_dist):
	visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
	return visibles

def bootstap_inference(frames, query_points, model):
	# Preprocess video to match model inputs format
	frames = preprocess_frames(frames)

	query_points = query_points.float()
	frames, query_points = frames, query_points

	# Model inference
	outputs = model(frames, query_points)
	tracks, occlusions, expected_dist = outputs["tracks"][0], outputs["occlusion"][0], outputs["expected_dist"][0]

	# Binarize occlusions
	visibles = postprocess_occlusions(occlusions, expected_dist)
	return tracks, visibles, occlusions, expected_dist

def get_depth_queries_tracks(frame_depths, query_points, tracks):
    max_height, max_width = frame_depths.shape[1], frame_depths.shape[2]
    height_queries, width_queries = query_points[..., 1].clip(0, max_height - 1), query_points[..., 2].clip(0, max_width - 1)
    height_tracks, width_tracks = tracks[..., 0].round().clip(0, max_height - 1).astype(int), tracks[..., 1].round().clip(0, max_width - 1).astype(int)
    depth_queries = frame_depths[0, width_queries.round().astype(int), height_queries.round().astype(int)].reshape(height_queries.shape + (1, ))
    
    frame_indices = torch.arange(height_tracks.shape[1]).unsqueeze(0).expand(height_tracks.shape[0], -1)
    depth_tracks = frame_depths[frame_indices.flatten(), width_tracks.flatten(), height_tracks.flatten()].reshape(height_tracks.shape + (1, ))
    
    return depth_queries, depth_tracks

@torch.no_grad()
def get_depth(
    pipeline,
    frames,
    num_denoising_steps=5,
    guidance_scale=0,
    window_size=110,
    overlap=25,
):
    with torch.inference_mode():
        depth = pipeline(
            frames.astype(np.float32) / 255.0,
            height=frames.shape[1],
            width=frames.shape[2],
            output_type="np",
            guidance_scale=guidance_scale,
            num_inference_steps=num_denoising_steps,
            window_size=window_size,
            overlap=overlap,
        ).frames[0]
    
    depth = depth.sum(-1) / depth.shape[-1]
    return depth

TIME_DEBUG = False
MAIN_PROCESS_ONLY = False
if TIME_DEBUG:
	import time
	process = psutil.Process()

def generate_shards(args, dataset, accelerator):
	model = tapir_model.TAPIR(pyramid_level=1)
	model.load_state_dict(torch.load(args.model_checkpoint))
	model.eval()
	torch.set_grad_enabled(False)
	accelerator.prepare(model)

	if args.depthcrafter_unet_path is not None and args.depthcrafter_pre_train_path is not None:
		unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
			args.depthcrafter_unet_path,
			low_cpu_mem_usage=True,
		)
		depth_pipeline = DepthCrafterPipeline.from_pretrained(
			args.depthcrafter_pre_train_path,
			unet=unet,
		)
		depth_pipeline = depth_pipeline.to(accelerator.device)
		depth_pipeline.set_progress_bar_config(disable=True)
	else:
		depth_pipeline = None
	
	if accelerator.is_main_process:
		Path(os.path.join(args.arrow_dir, args.dataset_save_name)).mkdir(parents=True, exist_ok=True)

	def preprocess(examples):
		for idx in range(len(examples["video"])):
			frames = examples["video"][idx]["frames"]
			if frames is None:
				examples["video"][idx]["frames"] = torch.empty((1,))
			else:
				if args.crop_still_edges:
					frames = crop_still_edges(frames)
				if (
					min(frames.shape[1], frames.shape[2]) < args.min_width_height
					or max(frames.shape[1], frames.shape[2]) / min(frames.shape[1], frames.shape[2]) > args.max_distortion
					or (args.camera_motion_detection and camera_motion_detection(frames))
				):
					examples["video"][idx]["frames"] = torch.empty((1,))
				else:
					examples["video"][idx]["frames"] = media.resize_video(frames, (args.resize_height, args.resize_width))
				examples["video"][idx]["fps"] = str(examples["video"][idx]["fps"])
		examples["query_points"] = [sample_random_points(0, args.resize_height, args.resize_width, args.num_points) for i in range(len(examples["video"]))]
		if args.height_num_points is not None and args.width_num_points is not None:
			grid_points = sample_grid_points(args.resize_height, args.resize_width, args.height_num_points, args.width_num_points)
			examples["query_points"] = [
				torch.cat((query_points[i], grid_points), axis=0) for i in range(len(examples["video"]))
			]
		return examples
	
	dataset.set_transform(preprocess)
	
	video_feature = Video()
	array_feature = Array()

	accelerator.wait_for_everyone()
	
	for split in dataset:
		if args.num_shards is not None and args.shard_index is not None:
			arrow_file_name = f"{split}-{accelerator.process_index}-{args.shard_index}.arrow"
			split_dataset = dataset[split].shard(num_shards=args.num_shards, index=args.shard_index)
		else:
			arrow_file_name = f"{split}-{accelerator.process_index}.arrow"
			split_dataset = dataset[split]
		writer = ArrowWriter(
			path=os.path.join(args.arrow_dir, args.dataset_save_name, arrow_file_name),
			writer_batch_size=args.writer_batch_size
		)
		loader = DataLoader(split_dataset, batch_size=1, shuffle=False, num_workers=args.dataloader_num_workers)
		loader = accelerator.prepare(loader)
		if TIME_DEBUG:
			start_time = time.time()
			current_memory = process.memory_info().rss
		pbar = tqdm(enumerate(loader), total=len(loader))
		for idx, sample in pbar:
			try:
				if TIME_DEBUG and (not MAIN_PROCESS_ONLY or accelerator.is_main_process):
					torch.cuda.synchronize()
					print(f"data time sample {idx}, process {accelerator.process_index}: {time.time() - start_time}, memory allocation change: {process.memory_info().rss - current_memory}")
					start_time = time.time()
					current_memory = process.memory_info().rss

				frames = sample["video"]["frames"]
				# video error
				if frames.numel() == 1:
					print(f"discarded {idx}")
					continue
				if args.max_frames is not None:
					frames = frames[:, :args.max_frames]
				
				query_points = sample["query_points"]
				if TIME_DEBUG and (not MAIN_PROCESS_ONLY or accelerator.is_main_process):
					torch.cuda.synchronize()
					print(f"processing time sample {idx}, process {accelerator.process_index}: {time.time() - start_time}, memory allocation change: {process.memory_info().rss - current_memory}")
					start_time = time.time()
					current_memory = process.memory_info().rss
				tracks, visibles, _, _ = bootstap_inference(frames, query_points, model)
				if TIME_DEBUG and (not MAIN_PROCESS_ONLY or accelerator.is_main_process):
					torch.cuda.synchronize()
					print(f"inference time sample {idx}, process {accelerator.process_index}: {time.time() - start_time}, memory allocation change: {process.memory_info().rss - current_memory}")
					start_time = time.time()
					current_memory = process.memory_info().rss
				# Flipped for some reason
				height_queries = query_points[..., 2].clone()
				query_points[..., 2] = query_points[..., 1]
				query_points[..., 1] = height_queries
				query_points = query_points[0]
				if TIME_DEBUG and (not MAIN_PROCESS_ONLY or accelerator.is_main_process):
					torch.cuda.synchronize()
					print(f"post processing time sample {idx}, process {accelerator.process_index}: {time.time() - start_time}, memory allocation change: {process.memory_info().rss - current_memory}")
					start_time = time.time()
					current_memory = process.memory_info().rss
				# Add depth here
				tracks = tracks.cpu().numpy().astype(np.float16)
				query_points = query_points.cpu().numpy().astype(np.float16)
				visibles = visibles.cpu().numpy().astype(bool)
				movements = topk_track_movement(tracks, visibles)
				
				if depth_pipeline is not None:
					frame_depths = get_depth(depth_pipeline, frames[0].cpu().numpy())
					depth_queries, depth_tracks = get_depth_queries_tracks(frame_depths, query_points, tracks)
					query_points = np.concatenate((query_points, depth_queries), axis=-1).astype(np.float16)
					tracks = np.concatenate((tracks, depth_tracks), axis=-1).astype(np.float16)
				
				if TIME_DEBUG and (not MAIN_PROCESS_ONLY or accelerator.is_main_process):
					torch.cuda.synchronize()
					print(f"casting time sample {idx}, process {accelerator.process_index}: {time.time() - start_time}, memory allocation change: {process.memory_info().rss - current_memory}")
					start_time = time.time()
					current_memory = process.memory_info().rss

				example = {
					"tracks": tracks,
					"query_points": query_points,
					"visibles": visibles,
					"movements": movements
				}
				if sample.get("text") is not None:
					example["text"] = sample["text"][0]
				
				sample["video"]["frames"] = frames.cpu().numpy().astype(np.uint8)
				example["camera_motion"] = camera_motion_detection(sample["video"]["frames"][0])
				sample["video"]["fps"] = [Fraction(sample["video"]["fps"][0])]

				sample["video"] = {k: v[0] for k,v in sample["video"].items()}
				example["video"] = video_feature.encode_example(sample["video"])
				example["tracks"] = array_feature.encode_example(example["tracks"])
				example["visibles"] = array_feature.encode_example(example["visibles"])
				example["query_points"] = array_feature.encode_example(example["query_points"])
				
				if TIME_DEBUG and (not MAIN_PROCESS_ONLY or accelerator.is_main_process):
					torch.cuda.synchronize()
					print(f"result preparation time sample {idx}, process {accelerator.process_index}: {time.time() - start_time}, memory allocation change: {process.memory_info().rss - current_memory}")
					start_time = time.time()
					current_memory = process.memory_info().rss

				writer.write(example, key=idx * accelerator.state.num_processes + accelerator.process_index)
				if TIME_DEBUG and (not MAIN_PROCESS_ONLY or accelerator.is_main_process):
					torch.cuda.synchronize()
					print(f"write time sample {idx}, process {accelerator.process_index}: {time.time() - start_time}, memory allocation change: {process.memory_info().rss - current_memory}")
					start_time = time.time()
					current_memory = process.memory_info().rss
			except Exception as e:
				if args.raise_exceptions:
					raise e
				else:
					print(f"Failed to process example {idx}, {e}")
			if args.max_samples_per_split is not None and idx >= args.max_samples_per_split - 1:
				break
		writer.finalize()
		if args.num_shards is not None:
			with open(os.path.join(args.arrow_dir, args.dataset_save_name, f"completion-{args.shard_index}.txt"), "w") as f:
				f.write("done")
		accelerator.wait_for_everyone()

def save_dataset(args, dataset):
	num_tries = 0
	while num_tries < 1 or args.continual_save_retry:
		try:
			if args.save_to_disk_dir is not None:
				print(f"saving locally to {os.path.join(args.save_to_disk_dir, args.dataset_save_name)}")
				dataset.save_to_disk(os.path.join(args.save_to_disk_dir, args.dataset_save_name))
			else:
				print(f"saving to hub at {args.user_upload_name + '/' + args.dataset_save_name}")
				dataset.push_to_hub(args.user_upload_name + "/" + args.dataset_save_name)
			break
		except Exception as e:
			print(f"Failed to save dataset, retrying in 5 seconds: {e}")
			time.sleep(5)
		num_tries += 1
	
	if not args.keep_arrow_files:
		shutil.rmtree(os.path.join(args.arrow_dir, args.dataset_save_name), ignore_errors=True)

def gather_shards(args, dataset, accelerator):
	if ((args.num_shards is None and args.shard_index is None) or args.num_shards == 1):
		new_dataset = {}
		for split in dataset:
			split_datasets = []
			for process_index in range(accelerator.state.num_processes):
				split_datasets.append(Dataset.from_file(os.path.join(args.arrow_dir, args.dataset_save_name, f"{split}-{process_index}.arrow")))
			
			split_dataset = concatenate_datasets(split_datasets)
			new_dataset[split] = split_dataset

		new_dataset = DatasetDict(new_dataset)
		save_dataset(args, new_dataset)
	elif args.num_shards is not None and args.shard_index == args.num_shards - 1:
		while True:
			if all(os.path.exists(os.path.join(args.arrow_dir, args.dataset_save_name, f"completion-{i}.txt")) for i in range(args.num_shards)):
				break
			print("waiting for other shards to finish")
			time.sleep(5)
		new_dataset = {}
		for split in dataset:
			split_datasets = []
			for process_index in range(accelerator.state.num_processes):
				for shard_index in range(args.num_shards):
					split_datasets.append(Dataset.from_file(os.path.join(args.arrow_dir, args.dataset_save_name, f"{split}-{process_index}-{shard_index}.arrow")))
			
			split_dataset = concatenate_datasets(split_datasets)
			new_dataset[split] = split_dataset

		new_dataset = DatasetDict(new_dataset)
		save_dataset(args, new_dataset)

		
			
def main():
	args = parse_args()
	accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=360000))])
	if accelerator.is_main_process:
		wandb.init()
	if args.dataset_save_name is None:
		args.dataset_save_name = f"{args.input_dataset.split('/')[-1]}-{args.model_checkpoint.split('/')[-1].split('.')[0]}"
		if accelerator.is_main_process:
			print(f"No dataset name provided, uploading to {args.user_upload_name + '/' + args.dataset_save_name}")

	num_tries = 0
	while num_tries < 1 or args.continual_load_retry:
		try:
			if args.local_dataset:
				dataset = load_from_disk(args.input_dataset)
			else:
				dataset = load_dataset(args.input_dataset)	
			break
		except Exception as e:
			print(f"Failed to load dataset, retrying in 5 seconds: {e}")
			time.sleep(5)
		num_tries += 1
	
	if not args.gather_shards_only:
		generate_shards(args, dataset, accelerator)
	
	if accelerator.is_main_process:
		gather_shards(args, dataset, accelerator)

if __name__ == '__main__':
	main()
