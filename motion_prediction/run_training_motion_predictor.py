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
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, Video, Array, Value

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry

from motion_predictor import MotionPredictorConfig
from motion_predictor import MotionPredictorProcessor
from motion_predictor import MotionPredictorForRectifiedFlow

from trainer_utils import EMATrainer, WandbVideoTrackVisualizer, TrainProcessorTransformWrapper, DataCollator, EvaluationCallback, NanLossCallback, data_config_to_dataset
from motion_predictor.evaluation import EvaluationProcessorTransformWrapper
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
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split: Optional[str] = field(
        default=None, metadata={"help": "name of the dataset train split"}
    )
    validation_split: Optional[str] = field(
        default=None, metadata={"help": "name of the dataset validation split"}
    )
    video_column_name: Optional[str] = field(
        default="video"
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    train_max_frames: Optional[int] = field(
        default=8,
        metadata={"help": "number of video frames to encode as context"}
    )
    train_min_frames: Optional[int] = field(
        default=None,
        metadata={"help": "minimum number of video frames to encode as context"}
    )
    visualization_interval: Optional[int] = field(
        default=None,
        metadata={"help": "How often to visualize sampled tracks. None will never visualize"}
    )
    visualization_train_indices: Optional[str] = field(
        default=None,
        metadata={"help": "comma separated training indices for visualization"}
    )
    visualization_validation_indices: Optional[str] = field(
        default=None,
        metadata={"help": "comma separated validation indices for visualization"}
    )
    visualization_num_sampling_steps: Optional[int] = field(
        default=16,
        metadata={"help": "number of sampling steps (per timestep) to use for visualization"}
    )
    sampling_evaluation_interval: Optional[int] = field(
        default=None,
        metadata={"help": "How often to perform sampling based evaluation. None will never evaluate"}
    )
    num_times_to_sample: Optional[int] = field(
        default=5,
        metadata={"help": "Number of times to sample for sampling based evaluation"}
    )
    evaluation_num_sampling_steps: Optional[int] = field(
        default=16,
        metadata={"help": "How many flow steps to use for evaluation"}
    )
    movement_weighting_temperature: Optional[float] = field(
        default=0,
    )
    evaluation_mask_camera_motion_indicator: Optional[bool] = field(
        default=False
    )
    evaluation_num_frames: Optional[int] = field(
        default=8
    )
    evaluation_provide_tracks_for_visible_frames: Optional[bool] = field(
        default=True
    )
    random_start: Optional[bool] = field(
        default=False
    )
    visible_min_ratio: Optional[float] = field(
        default=0,
    )
    mask_non_visible_tracks: Optional[bool] = field(
        default=False,
    )
    min_track_count: Optional[int] = field(
        default=None,
    )
    random_movement_weighting_temperature: Optional[bool] = field(
        default=False,
    )
    track_skip_min: Optional[int] = field(
        default=1,
    )
    track_skip_max: Optional[int] = field(
        default=1,
    )
    max_track_count: Optional[int] = field(
        default=None,
    )
    total_points_needed_before_selection: Optional[int] = field(
        default=None,
    )
    movement_weighting_loss_temperature: Optional[int] = field(
        default=None,
    )
    solver: Optional[str] = field(
        default="euler",
    )
    timestep_exponential_decay_loss_factor: Optional[float] = field(
        default=None,
    )
    train_frames_geometric_distribution_prob: Optional[float] = field(
        default=None,
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
    #config_name: Optional[str] = field(
    #    default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    #)
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
    stop_training_on_nan: bool = field(
        default=False,
        metadata={"help": "Will automatically stop training on nan loss"}
    )
    reset_ema_model: Optional[bool] = field(
        default=False,
    )

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
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_motion_prediction", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(101)
    
    # Is dataset config
    if os.path.exists(data_args.dataset) and data_args.dataset.split(".")[-1] == "json":
        with open(data_args.dataset) as f:
            data_config = json.load(f)
        dataset = data_config_to_dataset(data_config)
    else:
        dataset = load_dataset(
            data_args.dataset,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        track_array_columns = ["tracks", "query_points", "visibles"]
        indicator_split = list(dataset.keys())[0]
        dataset = dataset.cast_column(data_args.video_column_name, Video())
        for c in track_array_columns:
            if dataset[indicator_split].features[c] == Value(dtype="binary", id=None):
                dataset = dataset.cast_column(c, Array())
    
    config = MotionPredictorConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if data_args.max_track_count is not None:
        config.track_subsample_count = data_args.max_track_count
    if data_args.timestep_exponential_decay_loss_factor is not None:
        config.timestep_exponential_decay_loss_factor = data_args.timestep_exponential_decay_loss_factor
    
    processor = MotionPredictorProcessor(config)
    if os.path.exists(os.path.join(model_args.model_name_or_path, "model.safetensors")):
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
        if model_args.reset_ema_model:
            model.reset_ema_model()
    else:
        model = MotionPredictorForRectifiedFlow(config)

    train_transform = TrainProcessorTransformWrapper(
        processor,
        random_start=data_args.random_start,
        train_max_frames=data_args.train_max_frames,
        movement_weighting_temperature=data_args.movement_weighting_temperature,
        mask_non_visible_tracks=data_args.mask_non_visible_tracks,
        visible_min_ratio=data_args.visible_min_ratio,
        random_movement_weighting_temperature=data_args.random_movement_weighting_temperature,
        track_skip_max=data_args.track_skip_max,
        track_skip_min=data_args.track_skip_min,
    )
    evaluation_transform = EvaluationProcessorTransformWrapper(
        processor,
        eval_max_frames=data_args.evaluation_num_frames,
        track_skip=data_args.track_skip_min
    )
    if training_args.do_train:
        if data_args.train_split not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset[data_args.train_split] = (
                dataset[data_args.train_split].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        dataset[data_args.train_split].set_transform(train_transform)

    if training_args.do_eval:
        if data_args.validation_split not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset[data_args.validation_split] = (
                dataset[data_args.validation_split].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        dataset[data_args.validation_split].set_transform(evaluation_transform)
    
    collate_fn = DataCollator(
        data_args.train_max_frames,
        data_args.min_track_count,
        train_frames_geometric_distribution_prob=data_args.train_frames_geometric_distribution_prob
    )

    # Initialize our trainer
    trainer = EMATrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[data_args.train_split] if training_args.do_train else None,
        eval_dataset=dataset[data_args.validation_split] if training_args.do_eval else None,
        data_collator=collate_fn,
    )
    if data_args.visualization_interval is not None:
        trainer.add_callback(WandbVideoTrackVisualizer(
            log_step_interval=data_args.visualization_interval,
            train_sample_indices=[int(i) for i in data_args.visualization_train_indices.split(",")] if data_args.visualization_train_indices is not None else None,
            validation_sample_indices=[int(i) for i in data_args.visualization_validation_indices.split(",")] if data_args.visualization_validation_indices is not None else None,
            sampling_kwargs={"num_steps": data_args.visualization_num_sampling_steps, "solver": data_args.solver},
            max_track_length=config.max_track_length,
            train_dataloader=trainer.get_train_dataloader(),
            eval_dataloader=trainer.get_eval_dataloader(),
        ))
    if data_args.sampling_evaluation_interval is not None:
        trainer.add_callback(EvaluationCallback(
            eval_dataloader=trainer.get_eval_dataloader(),
            evaluation_interval=data_args.sampling_evaluation_interval,
            num_times_to_sample=data_args.num_times_to_sample,
            sampling_kwargs={"num_steps": data_args.evaluation_num_sampling_steps, "solver": data_args.solver},
            provide_tracks_for_visible_frames=data_args.evaluation_provide_tracks_for_visible_frames,
        ))
    if model_args.stop_training_on_nan:
        trainer.add_callback(NanLossCallback())
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    #if training_args.push_to_hub:
    #    trainer.push_to_hub(**kwargs)
    #else:
    #    trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()