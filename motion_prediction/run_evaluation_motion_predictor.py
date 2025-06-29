#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import logging
import os
import json
import sys

from dataclasses import dataclass, field, asdict
from typing import Optional, List
from pathlib import Path
from tqdm import tqdm
import numpy as np
import av

from accelerate import Accelerator
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from motion_predictor import MotionPredictorConfig, MotionPredictorProcessor, MotionPredictorForRectifiedFlow

from trainer_utils import EMATrainer, DataCollator, get_dataset
from motion_predictor.evaluation import EvaluationProcessorTransformWrapper, get_preds_labels_visibles, calculate_metrics
from motion_predictor.viz_utils import paint_point_track, paint_future_point_track

""" Training a MotionPredictor model """

logger = logging.getLogger(__name__)

    
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    validation_split: Optional[str] = field(
        default=None, metadata={"help": "name of the dataset validation split"}
    )
    evaluation_num_frames: Optional[int] = field(
        default=8,
        metadata={"help": "number of video frames to encode as context"}
    )
    num_times_to_sample: Optional[int] = field(
        default=5,
        metadata={"help": "Number of times to sample for sampling based evaluation"}
    )
    evaluation_num_sampling_steps: Optional[int] = field(
        default=16,
        metadata={"help": "How many flow steps to use for evaluation"}
    )
    evaluation_solver: Optional[str] = field(
        default="euler",
    )
    mask_non_visible_tracks: Optional[bool] = field(
        default=False,
    )
    min_track_count: Optional[int] = field(
        default=None,
    )
    max_track_count: Optional[int] = field(
        default=None,
    )
    total_points_needed_before_selection: Optional[int] = field(
        default=None,
    )
    output_folder: Optional[str] = field(
        default="evaluation_results"
    )
    mask_camera_motion_indicator: Optional[bool] = field(
        default=False
    )
    write_visualizations: Optional[bool] = field(
        default=False
    )
    max_examples: Optional[int] = field(
        default=None
    )
    provide_tracks_for_visible_frames: Optional[bool] = field(
        default=False,
    )
    evaluation_output_prepend_id: Optional[str] = field(
        default=None
    )
    thresholds: Optional[List[int]] = field(
        default_factory=lambda: [1, 2, 4, 8, 16],
        metadata={"nargs": "+"}
    )
    visualization_horizon: Optional[int] = field(
        default=0,
    )
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    sampling_apply_noise_schedule: bool = field(
        default=False,
    )

def frames_to_video(frames, output_file_name, codec_name, fps, pix_fmt, video_format='mp4') -> bytes:
    if codec_name == "hevc":
        codec_name = "h264"
    output_video = av.open(output_file_name, 'w', format=video_format)
    out_stream = output_video.add_stream(codec_name, fps, options={"x265-params": "log-level=0", "loglevel": "quiet"})
    
    out_stream.height = frames.shape[1]
    out_stream.width = frames.shape[2]
    
    out_stream.pix_fmt = pix_fmt

    for frame in frames:
        out_frame = av.VideoFrame.from_ndarray(frame)  # Note: to_image and from_image is not required in this specific example.
        out_packet = out_stream.encode(out_frame)  # Encode video frame
        output_video.mux(out_packet)  # "Mux" the encoded frame (add the encoded frame to MP4 file).
    # Flush the encoder
    out_packet = out_stream.encode(None)

    output_video.mux(out_packet)
    output_video.close()

def generate_visualizations(dataset, all_preds, all_labels, all_visibles, output_folder, frame_start=8, visualization_horizon=0):
    for data_idx in tqdm(range(len(dataset))):
        frames = dataset[data_idx]["video"]["frames"][frame_start:]
        encoding_args = {k:v for k, v in dataset[data_idx]["video"].items() if k != "frames"}

        num_tracks_mask = all_visibles[data_idx].sum(-1) > 0
        filtered_preds = all_preds[data_idx, :, num_tracks_mask].swapaxes(0, 1)
        filtered_labels = all_labels[data_idx, num_tracks_mask]
        filtered_visibles = all_visibles[data_idx, num_tracks_mask]

        if len(frames) > filtered_labels.shape[1]:
            frames = frames[:filtered_labels.shape[1]]
        else:
            filtered_preds = filtered_preds[:, :, :frames.shape[0]]
            filtered_labels = filtered_labels[:, :frames.shape[0]]
            filtered_visibles = np.ones_like(filtered_visibles[:, :frames.shape[0]])
        
        video_out_dir = os.path.join(output_folder, f"example_{data_idx}")
        Path(video_out_dir).mkdir(parents=True, exist_ok=True)
        
        gt_viz = paint_point_track(frames, filtered_labels, filtered_visibles)
        
        frames_to_video(gt_viz, output_file_name=os.path.join(video_out_dir, "gt_vizualization.mp4"), **encoding_args)

        for sample_idx in range(all_preds.shape[1]):
            if visualization_horizon > 0:
                horizon_preds = filtered_preds[sample_idx]
                horizon_preds = np.stack([horizon_preds[:, i:i+visualization_horizon] for i in range(horizon_preds.shape[1] - visualization_horizon)])
                sample_viz = paint_future_point_track(frames[:horizon_preds.shape[0]], horizon_preds)
            else:
                sample_viz = paint_point_track(frames, filtered_preds[sample_idx], filtered_visibles)
            frames_to_video(sample_viz, output_file_name=os.path.join(video_out_dir, f"sample_{sample_idx}_visualization.mp4"), **encoding_args)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args.remove_unused_columns = False
    # Set seed before initializing model.
    set_seed(101)
    
    evaluation_dataset = get_dataset(data_args.dataset, split=data_args.validation_split)
    
    config = MotionPredictorConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if data_args.max_track_count is not None:
        config.track_subsample_count = data_args.max_track_count
    
    processor = MotionPredictorProcessor(config)
    model = MotionPredictorForRectifiedFlow.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    evaluation_transform = EvaluationProcessorTransformWrapper(
        processor,
        eval_max_frames=data_args.evaluation_num_frames,
        total_points_needed_before_selection=data_args.total_points_needed_before_selection,
        mask_camera_motion_indicator=data_args.mask_camera_motion_indicator,
    )

    if data_args.max_examples is not None:
        evaluation_dataset = evaluation_dataset.select(range(data_args.max_examples))
    collate_fn = DataCollator(data_args.evaluation_num_frames, data_args.min_track_count)

    # Initialize our trainer
    trainer = EMATrainer(
        model=model,
        args=training_args,
        eval_dataset=evaluation_dataset.with_transform(evaluation_transform),
        data_collator=collate_fn,
    )
    eval_dataloader = trainer.get_eval_dataloader()
    sampling_kwargs = {
        "num_steps": data_args.evaluation_num_sampling_steps,
        "use_kv_cache": False,
        "solver": data_args.evaluation_solver,
        "apply_noise_schedule": model_args.sampling_apply_noise_schedule,
    }

    accelerator = Accelerator()
    all_preds, all_labels, all_visibles, all_query_points = get_preds_labels_visibles(
        model,
        eval_dataloader,
        num_pred_per_sample=data_args.num_times_to_sample,
        sampling_kwargs=sampling_kwargs,
        is_world_process_zero=accelerator.is_main_process,
        accelerator=accelerator,
        provide_tracks_for_visible_frames=data_args.provide_tracks_for_visible_frames,
    )
    
    keys_to_log = [f"pts_within_{i}" for i in data_args.thresholds]
    keys_to_log = keys_to_log + ["average_pts_within_thresh"]
    if accelerator.is_main_process:
        metrics = calculate_metrics(all_preds, all_labels, all_visibles, all_query_points, thresholds=data_args.thresholds)
        metrics = {k: v.item() for k, v in metrics.items() if k in keys_to_log}
        print(metrics)
        print(" & ".join(str(round(metrics[k], 3)) for k in keys_to_log))
        if model_args.model_name_or_path.split("/")[-1].split("-")[0] != "checkpoint":
            model_name = model_args.model_name_or_path.split("/")[-1]
        else:
            model_name = model_args.model_name_or_path.split("/")[-2]
        evaluation_output_name = f"{data_args.dataset.split('/')[-1]}-{model_name}"
        if data_args.evaluation_output_prepend_id is not None:
            evaluation_output_name = data_args.evaluation_output_prepend_id + "-" + evaluation_output_name
            evaluation_output_name = evaluation_output_name[:os.pathconf("/", "PC_NAME_MAX")]
        evaluation_output_folder = os.path.join(data_args.output_folder, evaluation_output_name)
        Path(evaluation_output_folder).mkdir(parents=True, exist_ok=True)
        if data_args.write_visualizations:
            generate_visualizations(
                dataset=evaluation_dataset,
                all_preds=all_preds,
                all_labels=all_labels,
                all_visibles=all_visibles,
                output_folder=os.path.join(evaluation_output_folder, "visualizations"),
                frame_start=data_args.evaluation_num_frames,
                visualization_horizon=data_args.visualization_horizon
            )
        with open(os.path.join(evaluation_output_folder, "results.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        with open(os.path.join(evaluation_output_folder, "config.json"), "w") as f:
            json.dump(asdict(data_args), f, indent=4)

if __name__ == "__main__":
    main()