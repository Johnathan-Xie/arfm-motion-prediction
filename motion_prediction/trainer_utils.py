import random
import numpy as np
from copy import deepcopy
from functools import partial
import wandb
import torch

from transformers.trainer_callback import TrainerCallback, TrainerState
from accelerate import Accelerator
from datasets import load_dataset, Video, Array, Value, load_from_disk, concatenate_datasets, interleave_datasets, DatasetDict
from transformers import Trainer, TrainerCallback, TrainerState, TrainerControl, TrainingArguments, BatchFeature
from motion_predictor.viz_utils import paint_point_track

class EMATrainer(Trainer):
    def training_step(
        self, model, inputs, num_items_in_batch=None
    ) -> torch.Tensor:
        training_step_output = super().training_step(model, inputs, num_items_in_batch)
        if self.model.use_consistency:
            self.model.ema_model.update()
        return training_step_output

class WandbVideoTrackVisualizer(TrainerCallback):
    def __init__(
        self,
        log_step_interval=1000,
        num_samples=5,
        random_sample=False,
        repeats=1,
        train_sample_indices=None,
        validation_sample_indices=None,
        sampling_kwargs={"num_steps": 16, "solver": "euler"},
        max_track_length=50,
        train_dataloader=None,
        eval_dataloader=None,
    ):
        super().__init__()
        self.log_step_interval = log_step_interval
        self.num_samples = num_samples
        self.random_sample = random_sample
        self.train_sample_indices = train_sample_indices
        self.validation_sample_indices = validation_sample_indices
        self.repeats = repeats
        self.sampling_kwargs = sampling_kwargs
        self.max_track_length = max_track_length
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            self.run = wandb.run if wandb.run is not None else wandb.init()
            self.table = wandb.Table(columns=["step", "predicted_tracks", "true_tracks"])
    
    def make_inference_sample(self, batch):
        keys = ["pixel_values", "global_pixel_values", "attention_mask", "input_ids", "camera_motion", "track_rate", "text_input_ids", "text_attention_mask"]
        batch = BatchFeature({k:batch.get(k) for k in batch if k in keys if batch.get(k) is not None})
        
        batch["input_ids"] = batch["input_ids"][:, :, :1]
        return batch
    
    def sample_dataloader(self, loader, sample_indices=None):
        dataset = deepcopy(loader.dataset)
        # Hacky way of getting dataset transform. Maybe better way to access
        new_transform_fn = partial(dataset.format["format_kwargs"]["transform"], return_visualization_info=True)
        dataset = dataset.with_transform(new_transform_fn)
        collate_fn = loader.collate_fn
        if sample_indices is None:
            if self.random_sample:
                sample_indices = random.sample(list(range(len(dataset))), k=self.num_samples)
            else:
                sample_indices = list(range(self.num_samples))

        samples = [collate_fn([dataset[idx]]) for idx in sample_indices]
        return samples

    @torch.no_grad()
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        model=None,
        train_dataloader=None,
        eval_dataloader=None,
        **kwargs
    ):
        if state.global_step % self.log_step_interval or not state.is_world_process_zero:
            return
        # Weird hack required due to current wandb table issues
        self.table = wandb.Table(
            columns=self.table.columns, data=self.table.data
        )
        # deepcopy to avoid possibly propogating any changes

        visualization_model = model
        visualization_model.eval()
        samples = []
        if self.train_sample_indices is not None:
            train_samples = self.sample_dataloader(self.train_dataloader, sample_indices=self.train_sample_indices)
            samples.extend(train_samples)

        if self.validation_sample_indices is not None:
            eval_samples = self.sample_dataloader(self.eval_dataloader, sample_indices=self.validation_sample_indices)
            samples.extend(eval_samples)
        
        for sample in samples:
            inference_batch = self.make_inference_sample(sample).to(visualization_model.device)
            for _ in range(self.repeats):
                output = visualization_model.sample(
                    **inference_batch,
                    **self.sampling_kwargs,
                )
                predictions = output.predictor_output
                frames = sample["visualization_frames"][0]
                
                fps = float(sample["video_fps"][0])
                max_track_length = min(self.max_track_length, len(frames))
                
                visibles = sample["label_mask"][0, :, :max_track_length].numpy()
                predicted_tracks = predictions[0, :, :max_track_length].cpu().detach().numpy()
                predicted_video_viz = paint_point_track(frames, predicted_tracks, visibles)
                predicted_video_viz = np.transpose(predicted_video_viz, (0, 3, 1, 2))
                predicted_video_viz = wandb.Video(predicted_video_viz, fps=fps)

                labels = sample["labels"][0, :, :max_track_length].numpy()

                ground_truth_video_viz = paint_point_track(frames, labels, visibles)
                ground_truth_video_viz = np.transpose(ground_truth_video_viz, (0, 3, 1, 2))
                ground_truth_video_viz = wandb.Video(ground_truth_video_viz, fps=fps)

                self.table.add_data(state.global_step, predicted_video_viz, ground_truth_video_viz)
        
        self.run.log({"Track Prediction Visualizations": self.table}, commit=False)

class TrainProcessorTransformWrapper:
    def __init__(
        self,
        processor,
        train_max_frames=8,
        random_start=False,
        movement_weighting_temperature=0.0,
        random_movement_weighting_temperature=False,
        mask_non_visible_tracks=True,
        visible_min_ratio=0.0,
        track_skip_max=1,
        track_skip_min=1,
        min_frame_rate_for_track_skip=3.0,
        also_sample_video_skip_prob=1.0,
    ):
        self.processor = processor
        self.train_max_frames = train_max_frames
        self.random_start = random_start
        self.movement_weighting_temperature = movement_weighting_temperature
        self.random_movement_weighting_temperature = random_movement_weighting_temperature
        self.mask_non_visible_tracks = mask_non_visible_tracks
        self.visible_min_ratio = visible_min_ratio
        self.max_track_length = processor.track_processor.max_track_length
        self.track_skip_min = track_skip_min
        self.track_skip_max = max(track_skip_max, track_skip_min)
        self.min_frame_rate_for_track_skip = min_frame_rate_for_track_skip
        self.also_sample_video_skip_prob = also_sample_video_skip_prob

    def __call__(self, examples, return_visualization_info=False):
        if isinstance(examples, dict):
            videos = examples["video"]
            tracks = examples["tracks"]
            query_points = examples["query_points"]
            visibles = examples["visibles"]
            camera_motion = examples.get("camera_motion")
            text_inputs = examples.get("text")
        elif isinstance(examples, list):
            videos = [i["video"] for i in examples]
            tracks = [i["tracks"] for i in examples]
            query_points = [i["query_points"] for i in examples]
            visibles = [i["visibles"] for i in examples]
            camera_motion = [i.get("camera_motion") for i in examples]
            text_inputs = [i.get("text") for i in examples]
        
        if self.random_movement_weighting_temperature:
            movement_weighting_temperature = random.random() * self.movement_weighting_temperature
        else:
            movement_weighting_temperature = self.movement_weighting_temperature
        if self.track_skip_max > 1:
            sampled_track_skip = [random.randint(self.track_skip_min, min(self.track_skip_max, int(i["fps"] / self.min_frame_rate_for_track_skip))) for i in videos]
            tracks = [np.array(i) for i in tracks]
            tracks = [tracks[i][:, ::sampled_track_skip[i]] for i in range(len(sampled_track_skip))]
            
            visibles = [np.array(i) for i in visibles]
            visibles = [visibles[i][:, ::sampled_track_skip[i]] for i in range(len(sampled_track_skip))]
            
            also_skip_video = [random.random() < self.also_sample_video_skip_prob for _ in range(len(sampled_track_skip))]
            for i in range(len(sampled_track_skip)):
                if also_skip_video[i]:
                    videos[i]["frames"] = videos[i]["frames"][::sampled_track_skip[i]]
        if self.random_start:
            start_indices = [random.randint(0, max(t.shape[1] - self.max_track_length, 0)) for t in tracks]
            video_start_indices = deepcopy(start_indices)
            if self.track_skip_max > 1:
                video_start_indices = [video_start_indices[i] * sampled_track_skip[i] if not also_skip_video[i] else video_start_indices[i] for i in range(len(sampled_track_skip))]
        else:
            start_indices = [0] * len(videos)
            video_start_indices = [0] * len(videos)
        processor_output = self.processor(
            videos=videos,
            tracks=tracks,
            query_points=query_points,
            visibles=visibles,
            camera_motion=camera_motion,
            text_inputs=text_inputs,
            max_frames=self.train_max_frames,
            start_indices=start_indices,
            video_start_indices=video_start_indices,
            movement_weighting_temperature=movement_weighting_temperature,
            mask_non_visible_tracks=self.mask_non_visible_tracks,
            visible_min_ratio=self.visible_min_ratio,
        )
        processor_output["training"] = [True] * len(videos)
        if return_visualization_info:
            processor_output["visualization_frames"] = [videos[idx]["frames"][start_indices[idx]:start_indices[idx] + processor_output["labels"][idx].shape[1]] for idx in range(len(videos))]
            processor_output["video_fps"] = [videos[idx]["fps"] for idx in range(len(videos))]
        return processor_output

class DataCollator:
    def __init__(self, train_max_frames, min_track_count, train_frames_geometric_distribution_prob=None):
        self.train_max_frames = train_max_frames
        self.min_track_count = min_track_count

        if train_frames_geometric_distribution_prob is not None:
            self.num_frames_to_condition_dist = torch.distributions.Geometric(train_frames_geometric_distribution_prob)
        else:
            self.num_frames_to_condition_dist = None
    def __call__(self, examples):
        if isinstance(examples, list):
            keys = ["pixel_values", "global_pixel_values", "input_ids", "labels", "attention_mask", "label_mask", "track_rate", "camera_motion", "text_input_ids", "text_attention_mask"]
            visualization_keys = ["visualization_frames", "video_fps"]
            training = examples[0]["training"]
            collated_examples = dict()
            for k in keys:
                collated_examples[k] = torch.stack([i[k] for i in examples])
            for k in visualization_keys:
                if examples[0].get(k) is not None:
                    collated_examples[k] = [i[k] for i in examples]
        else:
            collated_examples = examples
            training = examples["training"][0]
        
        # Augmentation for training
        if training:
            if self.num_frames_to_condition_dist is not None:
                num_frames_to_condition = (self.num_frames_to_condition_dist.sample().long() + 1).clip(1, self.train_max_frames)
            else:
                num_frames_to_condition = random.randint(1, self.train_max_frames)
            collated_examples["pixel_values"] = collated_examples["pixel_values"][:, :num_frames_to_condition]
            collated_examples["global_pixel_values"] = collated_examples["global_pixel_values"][:, :num_frames_to_condition]
            if self.min_track_count is not None:
                num_tracks_to_use = random.randint(self.min_track_count, collated_examples["input_ids"].shape[1])
                collated_examples["input_ids"] = collated_examples["input_ids"][:, :num_tracks_to_use]
                collated_examples["labels"] = collated_examples["labels"][:, :num_tracks_to_use]
                collated_examples["attention_mask"] = collated_examples["attention_mask"][:, :num_tracks_to_use]
                collated_examples["label_mask"] = collated_examples["label_mask"][:, :num_tracks_to_use]
        return BatchFeature(collated_examples)

from motion_predictor.evaluation import get_preds_labels_visibles, calculate_metrics

class EvaluationCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataloader,
        evaluation_interval=10000,
        num_times_to_sample=5,
        sampling_kwargs={
            "num_steps": 16,
            "use_kv_cache": False,
            "solver": "euler"
        },
        keys_to_log=[
            "pts_within_1",
            "pts_within_2",
            "pts_within_4",
            "pts_within_8",
            "pts_within_16",
            "average_pts_within_thresh"
        ],
        provide_tracks_for_visible_frames=True,
    ):
        super().__init__()
        self.evaluation_interval = evaluation_interval
        self.num_times_to_sample = num_times_to_sample
        self.sampling_kwargs = sampling_kwargs
        self.keys_to_log = keys_to_log
        self.loader = eval_dataloader
        self.provide_tracks_for_visible_frames = provide_tracks_for_visible_frames

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            self.run = wandb.run if wandb.run is not None else wandb.init()
        self.accelerator = Accelerator()
        
    @torch.no_grad()
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        model=None,
        train_dataloader=None,
        eval_dataloader=None,
        **kwargs
    ):
        if state.global_step % self.evaluation_interval:
            return
        
        all_preds, all_labels, all_visibles, all_query_points = get_preds_labels_visibles(
            deepcopy(model),
            self.loader,
            num_pred_per_sample=self.num_times_to_sample,
            sampling_kwargs=self.sampling_kwargs,
            is_world_process_zero=state.is_world_process_zero,
            accelerator=self.accelerator,
            provide_tracks_for_visible_frames=self.provide_tracks_for_visible_frames,
        )
        if state.is_world_process_zero:
            metrics = calculate_metrics(all_preds, all_labels, all_visibles, all_query_points)
            self.run.log({k:metrics[k] for k in self.keys_to_log}, commit=False)

def create_query_point_sampling_visualization(
    images,
    track_processor,
    query_points,
    movements,
    visible_ratios,
    movement_weighting_temperature=0.5,
    visible_min_ratio=0.5
):
    track_processor_batch = {
        "query_points": query_points,
        "movements": movements,
        "visible_ratios": visible_ratios
    }

    inputs, _, _, _ = track_processor(**track_processor_batch, movement_weighting_temperature=movement_weighting_temperature, visible_min_ratio=visible_min_ratio)
    selected_query_points = inputs[:, :, 0]
    selected_query_points = selected_query_points.permute(1, 0, 2).cpu().numpy()
    visualizations = paint_point_track(np.stack(images), point_tracks=selected_query_points, visibles=np.ones_like(selected_query_points[..., 0], dtype=np.bool))
    return visualizations

def create_colormap(values):
    rescaled_values = (values - values.min(dim=1).values) / (values.max(dim=1).values - values.min(dim=1).values)
    return torch.repeat_interleave(rescaled_values.unsqueeze(-1), 3, dim=-1) * 255

def create_query_point_value_visualization(
    images,
    query_points,
    values,
):
    colormap = create_colormap(values)
    visualizations = []
    for i in range(len(images)):
        sample_query_points = np.expand_dims(query_points[i], 1)
        visualizations.append(
            paint_point_track(
                frames=np.expand_dims(images[i], 0),
                point_tracks=sample_query_points,
                visibles=np.ones_like(sample_query_points[..., 0], dtype=np.bool),
                colormap=colormap[i]
            )[0]
        )
    return visualizations

class NanLossCallback(TrainerCallback):
    def __init__(
        self,
    ):
        super().__init__()
        
    @torch.no_grad()
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        model=None,
        train_dataloader=None,
        eval_dataloader=None,
        **kwargs
    ):
        if state.is_world_process_zero:
            loader = deepcopy(train_dataloader)
            with torch.no_grad():
                for batch in loader: break
                output = model(**batch)
                if torch.isnan(output.loss):
                    print("Stopping training due to nan Loss")
                    control.should_training_stop = True

def get_dataset(
    name,
    split,
    storage_location="remote",
    camera_motion_force=None,
    min_index=None,
    max_index=None,
    min_movement=None,
    max_movement=None,
    **kwargs
):
    if storage_location == "remote":
        dataset = load_dataset(name, split=split)
    else:
        dataset = load_from_disk(name)[split]
    if camera_motion_force is not None:
        if "camera_motion" in dataset.column_names:
            dataset = dataset.remove_columns("camera_motion")
        dataset.add_column("camera_motion", [camera_motion_force] * len(dataset))
    dataset = dataset.cast_column("video", Video())
    track_array_columns = ["tracks", "query_points", "visibles"]
    for c in track_array_columns:
        if dataset.features[c] == Value(dtype="binary", id=None):
            dataset = dataset.cast_column(c, Array())
    if min_index is not None or max_index is not None:
        min_index = min_index if min_index is not None else 0
        max_index = max_index if max_index is not None else len(dataset)
        dataset = dataset.select(range(min_index, max_index))
    
    if min_movement is not None or max_movement is not None:
        min_movement = min_movement or 0
        max_movement = max_movement or 10 ** 8
        movements = np.array(dataset["movements"])
        within_movement_samples = (movements >= min_movement) & (movements < max_movement)
        within_movement_indices = [i for i in within_movement_samples if i]
        dataset = dataset.select(within_movement_indices)
    
    return dataset

def data_config_to_dataset(
    data_config,
    interleave_stopping_strategy="all_exhausted",
    seed=101,
    is_main_process=True,
):
    dataset_dict = {}
    for split, split_info in data_config.items():
        if all([i.get("interleave_ratio") is not None for i in split_info]):
            interleave_ratios = [i.pop("interleave_ratio") for i in split_info]
        else:
            interleave_ratios = None

        split_datasets = [get_dataset(**dataset_info) for dataset_info in split_info]
        for i in range(len(split_datasets)):
            split_datasets[i] = split_datasets[i].add_column("source", [split_info[i]["name"] + "-" + split_info[i]["split"]] * len(split_datasets[i]))
        if interleave_ratios is not None:
            interleave_total = sum(interleave_ratios)
            interleave_ratios = [i / interleave_total for i in interleave_ratios]
            # fixing any potential rounding errors
            interleave_ratios[0] += (1 - sum(interleave_ratios))
            merged_dataset = interleave_datasets(split_datasets, interleave_ratios, stopping_strategy=interleave_stopping_strategy, seed=seed)
        else:
            merged_dataset = concatenate_datasets(split_datasets)
        if is_main_process:
            print(f"{split} data source counts after reweighting if applied")
            print({k.item():v.item() for k,v in zip(*np.unique(np.array(merged_dataset["source"]), return_counts=True))})
        
        dataset_dict[split] = merged_dataset
    
    return DatasetDict(dataset_dict)