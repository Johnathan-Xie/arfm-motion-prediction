import torch
import numpy as np
import os
import shutil
import mediapy as media

from tapnet.torch import tapir_model
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import torch.nn.functional as F

def parse_args():
	parser = ArgumentParser()

	parser.add_argument("--input_path", type=str, default="")
	parser.add_argument("--output_path", type=str, default="")
	parser.add_argument("--model_checkpoint", type=str, default="checkpoints/bootstapir_checkpoint_v2.pt")
	parser.add_argument("--num_points", type=int, default=1000)
	parser.add_argument("--resize_height", type=int, default=256)
	parser.add_argument("--resize_width", type=int, default=256)
	parser.add_argument("--device", type=str, default="cuda:0")
	parser.add_argument("--relative_annotation_path", type=str, default="lang_annotations/auto_lang_ann.npy")

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

def copy_non_episode_data(input_path, output_path):
    non_episode_contents = [i for i in os.listdir(input_path) if "episode_" not in i]
    for non_episode_content in non_episode_contents:
        content_path = os.path.join(input_path, non_episode_content)
        if os.path.isdir(content_path):
            shutil.copytree(os.path.join(input_path, non_episode_content), os.path.join(output_path, non_episode_content), dirs_exist_ok=True)
        else:
            shutil.copyfile(os.path.join(input_path, non_episode_content), os.path.join(output_path, non_episode_content))

def sample_random_points(frame_max_idx, height, width, num_points):
	"""Sample random points with (time, height, width) order."""
	y = np.random.randint(0, height, (num_points, 1))
	x = np.random.randint(0, width, (num_points, 1))
	t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
	points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
	return points

timestep_formatting = lambda x: f"episode_{str(x).zfill(7)}.npz"

def add_point_tracks(
	input_dir,
	output_dir,
	tracking_model,
	height=256, width=256,
	num_point_tracks=1000,
	track_padding=50,
	observation_key="rgb_static",
	device="cuda:0",
	relative_annotation_path="lang_annotations/auto_lang_ann.npy"
):
	Path(output_dir).mkdir(exist_ok=True, parents=True)
	copy_non_episode_data(input_dir, output_dir)
	section_start_end_indices = np.load(os.path.join(input_dir, relative_annotation_path), allow_pickle=True).item()["info"]["indx"]
	start_end_ids = np.load(os.path.join(input_dir, "ep_start_end_ids.npy"), allow_pickle=True)
	ep_end_ids = start_end_ids[:, 1]
	for start_index, end_index in tqdm(section_start_end_indices):
		frames = []
		episode_data = []
		for timestep in range(start_index, end_index + track_padding + 1):
			filepath = os.path.join(input_dir, timestep_formatting(timestep))
			data = dict(np.load(filepath))
			frames.append(data[observation_key])
			episode_data.append(data)
			if timestep in ep_end_ids:
				break
		frames = media.resize_video(np.stack(frames), (256, 256))
		query_points = sample_random_points(0, height, width, num_point_tracks)
		frames, query_points = torch.Tensor(frames).unsqueeze(0).to(device), torch.Tensor(query_points).unsqueeze(0).to(device)
		tracks, visibles, _, _ = bootstap_inference(frames, query_points, tracking_model)
		tracks = tracks.cpu().numpy()
		visibles = visibles.cpu().numpy()
		for timestep in range(start_index, end_index + 1):
			filename = timestep_formatting(timestep)
			filepath = os.path.join(output_dir, filename)
			relative_timestep = timestep - start_index
			data = episode_data[relative_timestep]
			data["future_tracks"] = tracks[:, relative_timestep:]
			data["current_visible"] = visibles[:, relative_timestep]
			np.savez(os.path.join(output_dir, filename), **data)
			if timestep in ep_end_ids:
				print("Ending timestep in language annotation, this shouldn't occur")
				break

def main():
	args = parse_args()

	tracking_model = tapir_model.TAPIR(pyramid_level=1)
	tracking_model.load_state_dict(torch.load(args.model_checkpoint))
	tracking_model.eval().to(args.device)
	torch.set_grad_enabled(False)

	train_dir = os.path.join(args.input_path, "training")
	train_out_dir = os.path.join(args.output_path, "training")
	validation_dir = os.path.join(args.input_path, "validation")
	validation_out_dir = os.path.join(args.output_path, "validation")
	
	add_point_tracks(train_dir, train_out_dir, tracking_model, relative_annotation_path=args.relative_annotation_path)
	add_point_tracks(validation_dir, validation_out_dir, tracking_model)

if __name__ == '__main__':
	main()