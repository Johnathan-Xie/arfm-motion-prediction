import numpy as np

import torch
from torch import nn
from typing import Optional, Dict, Tuple, Union, Any
import torch.nn.functional as F

from tapnet.torch import tapir_model
from transformers import BatchFeature, PretrainedConfig

from .motion_predictor import MotionPredictorForRectifiedFlow, MotionPredictorProcessor
from .query_predictor import QueryPredictorForRegression, QueryPredictorForRectifiedFlow, QueryPredictorConfig, QueryPredictorProcessor
from transformers.cache_utils import StaticCache
from copy import deepcopy


class SlidingWindowCache(StaticCache):
    """
    Sliding Window Cache class to be used with `torch.compile` for models like Mistral that support sliding window attention.
    Every time when we try to update the cache, we compute the `indices` based on `cache_position >= self.config.sliding_window - 1`,
    if true(which means the cache can not hold all the old key value states and new states together because of the sliding window constraint),
    we need to do a cycle shift based on `indices` to replace the oldest states by the new key value states passed in.

    The `to_shift` is only true once we are above sliding_window. Thus with `sliding_window==64`:

    indices = (slicing + to_shift[-1].int()-1) % self.config.sliding_window
    tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 57, 58, 59, 60, 61, 62, 63,  0])

    We overwrite the cache using these, then we always write at cache_position (clamped to `sliding_window`)

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map(`Dict[int, Union[str, torch.device, int]]]`, `optional`):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, SlidingWindowCache

        >>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

        >>> inputs = tokenizer(text="My name is Mistral", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = SlidingWindowCache(config=model.config, batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        SlidingWindowCache()
        ```
    """

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config: PretrainedConfig,
        spatial_num_tokens: int,
        batch_size: int = None,
        max_cache_len: int = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: Optional[int] = None,
        non_sliding_length: Optional[int] = 0,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        if not hasattr(config, "sliding_window") or config.sliding_window is None:
            raise ValueError(
                "Setting `cache_implementation` to 'sliding_window' requires the model config supporting "
                "sliding window attention, please check if there is a `sliding_window` field in the model "
                "config and it's not set to None."
            )
        max_cache_len = min(config.sliding_window, max_cache_len)
        if batch_size is not None:
            batch_size = batch_size * spatial_num_tokens
        if max_batch_size is not None:
            max_batch_size = max_batch_size * spatial_num_tokens
        self.non_sliding_length = non_sliding_length
        super().__init__(
            config=config,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
            max_batch_size=max_batch_size,
            layer_device_map=layer_device_map,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        # assume this only happens in prefill phase when prompt length > sliding_window_size (= max_cache_len)
        if cache_position.shape[0] > self.max_cache_len:
            k_out = key_states[:, :, -self.max_cache_len :, :]
            v_out = value_states[:, :, -self.max_cache_len :, :]
            # Assumption: caches are all zeros at this point, `+=` is equivalent to `=` but compile-friendly
            self.key_cache[layer_idx] += k_out
            self.value_cache[layer_idx] += v_out
            # we should return the whole states instead of k_out, v_out to take the whole prompt
            # into consideration when building kv cache instead of just throwing away tokens outside of the window
            return key_states, value_states
        max_cache_position = cache_position[-1] #.max()
        if max_cache_position > self.max_cache_len - 1:
            shift_amount = max_cache_position - (self.max_cache_len - 1)
            self.key_cache[layer_idx][:, :, self.non_sliding_length:-shift_amount] = self.key_cache[layer_idx][:, :, self.non_sliding_length + shift_amount:]
            self.value_cache[layer_idx][:, :, self.non_sliding_length:-shift_amount] = self.value_cache[layer_idx][:, :, self.non_sliding_length + shift_amount:]
            cache_position = cache_position - shift_amount
    
        self.key_cache[layer_idx][:, :, cache_position] = key_states
        self.value_cache[layer_idx][:, :, cache_position] = value_states
        return self.key_cache[layer_idx][:, :, :cache_position[-1] + 1], self.value_cache[layer_idx][:, :, :cache_position[-1] + 1]

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def reset(self):
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()


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

class OnlineTAPIR(nn.Module):
    def __init__(
        self,
        model_path_or_name
    ):
        super().__init__()
        self.model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
        self.model.load_state_dict(torch.load(model_path_or_name))
        self.model.eval()
    
    def initialize(self, frames, query_points):
        """Initialize query features for the query points."""
        frames = preprocess_frames(frames)
        feature_grids = self.model.get_feature_grids(frames, is_training=False)
        query_features = self.model.get_query_features(
            frames,
            is_training=False,
            query_points=query_points,
            feature_grids=feature_grids,
        )
        causal_context = self.model.construct_initial_causal_state(
            query_points.shape[1], len(query_features.resolutions) - 1
        )
        return query_features, causal_context
    
    def predict(self, frames, query_features, causal_context):
        """Compute point tracks and occlusions given frames and query points."""
        frames = preprocess_frames(frames)
        feature_grids = self.model.get_feature_grids(frames, is_training=False)
        trajectories = self.model.estimate_trajectories(
            frames.shape[-3:-1],
            is_training=False,
            feature_grids=feature_grids,
            query_features=query_features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=causal_context,
            get_causal_context=True,
        )
        causal_context = trajectories["causal_context"]
        del trajectories["causal_context"]
        # Take only the predictions for the final resolution.
        # For running on higher resolution, it"s typically better to average across
        # resolutions.
        tracks = trajectories["tracks"][-1]
        occlusions = trajectories["occlusion"][-1]
        uncertainty = trajectories["expected_dist"][-1]
        visibles = postprocess_occlusions(occlusions, uncertainty)
        return tracks, visibles, causal_context

def euclidean_distance(
    predicted_track_positions,
    true_track_positions,
):
    return ((predicted_track_positions - true_track_positions) ** 2).sum(dim=-1).sqrt().mean(dim=-1)

def error_based_edit_magnitude(
    predicted_track_positions,
    true_track_positions,
    min_edit_sampling_steps=0,
    max_edit_sampling_steps=16,
    min_edit_sampling_start_step=0.0,
    max_edit_sampling_start_step=1.0,
    error_function="euclidean",
    error_top_k=None,
    error_cap=10.0,
):
    if error_function == "euclidean":
        errors = euclidean_distance(predicted_track_positions, true_track_positions)
    else:
        raise ValueError(f"Unknown error_function {error_function}")
    if error_top_k is not None:
        errors = errors.topk(k=error_top_k, dim=-1).values
    average_errors = errors.mean(-1)
    average_errors = average_errors.clip(max=error_cap)
    edit_sampling_steps = ((average_errors / error_cap) * (max_edit_sampling_steps - min_edit_sampling_steps)).round() + min_edit_sampling_steps
    edit_sampling_steps = edit_sampling_steps.clip(min_edit_sampling_steps, max_edit_sampling_steps)
    edit_sampling_steps = int(edit_sampling_steps.cpu().item())

    edit_sampling_start_step = max_edit_sampling_start_step - ((average_errors / error_cap) * (max_edit_sampling_start_step - min_edit_sampling_start_step)) + min_edit_sampling_start_step
    edit_sampling_start_step = edit_sampling_start_step.clip(min_edit_sampling_start_step, max_edit_sampling_start_step)
    edit_sampling_start_step = edit_sampling_start_step.cpu().item()
    return edit_sampling_steps, edit_sampling_start_step


class MotionPredictorPipeline(nn.Module):
    def __init__(
        self,
        motion_predictor_path,
        query_predictor_path,
        device="cuda",
        query_predictor_movement_inference_multiplier=1.0,
        point_tracker_path=None,
        max_context_length=None,
        accelerator=None,
    ):
        super().__init__()
        self.motion_predictor_processor = MotionPredictorProcessor.from_pretrained(motion_predictor_path)
        self.motion_predictor = MotionPredictorForRectifiedFlow.from_pretrained(motion_predictor_path).eval()
        
        if accelerator is not None:
            self.motion_predictor = accelerator.prepare(self.motion_predictor)
        else:
            self.motion_predictor = self.motion_predictor.to(device)
        if max_context_length is None:
            max_context_length = self.motion_predictor.config.max_track_length
        self.max_context_length = max_context_length

        self.height = self.motion_predictor.config.height
        self.width = self.motion_predictor.config.width
        
        self.query_predictor_processor = QueryPredictorProcessor.from_pretrained(query_predictor_path)
        query_predictor_config = QueryPredictorConfig.from_pretrained(query_predictor_path)
        if query_predictor_config.predictor_type == "rectified_flow":
            self.query_predictor = QueryPredictorForRectifiedFlow.from_pretrained(query_predictor_path).eval()
        else:
            self.query_predictor = QueryPredictorForRegression.from_pretrained(query_predictor_path).eval()
        
        if accelerator is not None:
            self.query_predictor = accelerator.prepare(self.query_predictor)
        else:
            self.query_predictor = self.query_predictor.to(device)
        self.query_predictor.config.movement_inference_multiplier = query_predictor_movement_inference_multiplier
        if point_tracker_path is not None:
            self.online_point_tracker = OnlineTAPIR(point_tracker_path)
        
        if accelerator is not None:
            self.online_point_tracker = accelerator.prepare(self.online_point_tracker)
        else:
            self.online_point_tracker = self.online_point_tracker.to(device)
        
        self.device = device
        self.reset_caches()


    def sample_random_points(self, frame_max_idx=0, num_points=None):
        """Sample random points with (time, height, width) order."""
        if num_points is None:
            num_points = self.query_predictor.config.track_subsample_count
        y = np.random.randint(0, self.height, (num_points, 1))
        x = np.random.randint(0, self.width, (num_points, 1))
        t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
        points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
        return points

    def sample_grid_points(self, height, width, height_num_points, width_num_points):
        """Sample grid points with (time, height, width) order."""
        height_offset = int(height / height_num_points / 2)
        width_offset = int(width / width_num_points / 2)
        
        y = np.linspace(height_offset, height - height_offset, height_num_points).repeat(width_num_points).astype(int)
        x = np.tile(np.linspace(width_offset, width - width_offset, width_num_points), height_num_points).astype(int)
        t = np.zeros_like(y)
        points = np.stack((t, y, x), axis=-1)
        return points

    @torch.no_grad()
    def get_query_information(
        self,
        images,
        full_query_point_set=None,
        text_inputs=None,
        camera_motion=None,
    ):
        if full_query_point_set is None:
            full_query_point_set = [self.sample_random_points() for _ in range(len(images))]

        query_predictor_batch = self.query_predictor_processor(
            images=images,
            query_points=full_query_point_set,
            camera_motion=camera_motion,
            text_inputs=text_inputs,
        )
        del query_predictor_batch["label_mask"]
        query_predictor_output = self.query_predictor.sample(**BatchFeature(query_predictor_batch).to(self.device))
        return full_query_point_set, query_predictor_output
    
    def reset_caches(self):
        self.tracker_query_features = None
        self.tracker_causal_context = None
        self.current_tracks = None
        self.current_visibles = None
        self.current_track_step = 0
        self.all_frames = None
    
        cache_config = deepcopy(self.motion_predictor.config.track_predictor_config)
        num_text_tokens = self.motion_predictor.config.text_encoder_max_seq_length if self.motion_predictor.config.text_encoder_name is not None else 0
        window_size = self.motion_predictor.config.max_track_length + num_text_tokens
        cache_config.sliding_window = window_size

        self.track_predictor_past_key_values = SlidingWindowCache(
            config=cache_config,
            spatial_num_tokens=(
                self.motion_predictor.config.track_subsample_count
                + (self.motion_predictor.config.num_global_image_tokens if self.motion_predictor.config.global_image_model_name is not None else 0)
            ),
            max_cache_len=window_size - 1,
            batch_size=1,
            device=self.device,
            non_sliding_length=num_text_tokens
        )
        self.motion_predictor.reset_caches()
    
    @torch.no_grad()
    def predict_tracks(
        self,
        frames,
        query_points=None,
        tracks=None,
        text_inputs=None,
        camera_motion=None,
        sampling_kwargs={},
        query_predictor_movement_inference_multiplier=1.0,
    ):
        if query_points is None:
            full_query_point_set, query_predictor_output = self.get_query_information(
                [i[0] for i in frames],
                text_inputs=text_inputs,
                camera_motion=camera_motion
            )
        else:
            full_query_point_set = None
        motion_predictor_input_batch = self.motion_predictor_processor(
            videos=[{"frames": i} for i in frames],
            query_points=query_points if query_points is not None else full_query_point_set,
            tracks=tracks,
            camera_motion=camera_motion,
            text_inputs=text_inputs,
            movements=[i.cpu() for i in query_predictor_output.movements * query_predictor_movement_inference_multiplier] if full_query_point_set is not None else None,
            visible_ratios=[i.cpu() for i in query_predictor_output.visible_ratios] if (full_query_point_set is not None and query_predictor_output.visible_ratios is not None) else None,
            for_sampling=True
        )
        if tracks is not None:
            motion_predictor_input_batch["input_ids"] = motion_predictor_input_batch["input_ids"][:, :, :tracks.shape[2] + 1]
        
        motion_predictor_outputs = self.motion_predictor.sample(
            **BatchFeature(motion_predictor_input_batch).to(self.device),
            **sampling_kwargs,
            track_predictor_past_key_values=self.track_predictor_past_key_values if sampling_kwargs.get("use_kv_cache") else None
        )
        self.track_predictor_past_key_values = motion_predictor_outputs.track_predictor_past_key_values if sampling_kwargs.get("use_kv_cache") else None
        return motion_predictor_outputs.predictor_output
    
    @torch.no_grad()
    def initialize_online_future_point_prediction(
        self,
        starting_frames,
        query_points=None,
        text_inputs=None,
        camera_motion=None,
        sampling_kwargs={},
        query_predictor_movement_inference_multiplier=1.0,
    ):
        self.reset_caches()
        self.all_frames = np.stack(starting_frames, axis=0)
        self.current_tracks = self.predict_tracks(
            frames=starting_frames,
            query_points=query_points,
            text_inputs=text_inputs,
            camera_motion=camera_motion,
            sampling_kwargs=sampling_kwargs,
            query_predictor_movement_inference_multiplier=query_predictor_movement_inference_multiplier,
        )
        self.current_track_step = 0
        query_points = self.current_tracks[:, :, 0]
        self.tracker_query_features, self.tracker_causal_context = self.online_point_tracker.initialize(
            frames=torch.stack([torch.Tensor(i).to(self.device) for i in starting_frames]),
            query_points=torch.cat([torch.zeros_like(query_points[..., :1]), query_points[..., 1:2], query_points[..., :1]], dim=-1)
        )
        for i in range(len(self.tracker_causal_context)):
            self.tracker_causal_context[i] = {k:v.to(self.device) for k,v in self.tracker_causal_context[i].items()}
        return self.current_tracks

    @torch.no_grad()
    def update_online_future_point_prediction(
        self,
        new_frames,
        condition_all_frames=False,
        text_inputs=None,
        camera_motion=None,
        predict_horizon=True,
        sampling_kwargs={},
        error_based_edit_magnitude_kwargs=None,
        query_predictor_movement_inference_multiplier=1.0,
    ):
        if self.current_tracks.shape[2] > self.max_context_length:
            self.current_track_step -= self.current_tracks.shape[2] - self.max_context_length
            self.current_tracks = self.current_tracks[:, :, self.current_tracks.shape[2] - self.max_context_length:]
        
        new_tracks, new_visibles, self.tracker_causal_context = self.online_point_tracker.predict(
            frames=torch.stack([torch.Tensor(i).to(self.device) for i in new_frames]),
            query_features=self.tracker_query_features,
            causal_context=self.tracker_causal_context
        )
        self.current_track_step += 1
        if predict_horizon:
            update_sampling_kwargs = deepcopy(sampling_kwargs)
            # start predicting at self.current_track_step + 1, but we need to perform kv cache update at self.current_track_step
            update_sampling_kwargs["start_timestep"] = self.current_track_step + 1
            update_sampling_kwargs["num_timesteps_to_sample"] = 1
            if error_based_edit_magnitude_kwargs is not None:
                update_num_steps, update_noising_start_timestep = error_based_edit_magnitude(
                    predicted_track_positions=self.current_tracks[:, :, self.current_track_step:self.current_track_step + 1],
                    true_track_positions=new_tracks,
                    **error_based_edit_magnitude_kwargs,
                )
                update_sampling_kwargs["update_num_steps"] = update_num_steps
                update_sampling_kwargs["update_noising_start_timestep"] = update_noising_start_timestep
            
        if self.current_tracks.shape[2] <= self.current_track_step:
            self.current_tracks = torch.cat((self.current_tracks, new_tracks), dim=2)
        else:
            future_shift = new_tracks - self.current_tracks[:, :, self.current_track_step:self.current_track_step + 1]
            self.current_tracks[:, :, self.current_track_step:self.current_track_step + 1] = new_tracks
            self.current_tracks[:, :, self.current_track_step + 1:] = self.current_tracks[:, :, self.current_track_step + 1:] + future_shift
        if predict_horizon:
            if condition_all_frames:
                self.all_frames = np.concatenate((self.all_frames, np.stack(new_frames, axis=0)), axis=1)
                self.all_frames = self.all_frames[:, -self.max_context_length:]
            query_points = self.current_tracks[:, :, 0].cpu()
            tracks = self.current_tracks[:, :, 1:].cpu()

            query_points = torch.cat((torch.zeros_like(query_points[..., :1]), query_points), dim=-1)
            output_tracks = self.predict_tracks(
                frames=self.all_frames,
                query_points=query_points,
                tracks=tracks,
                text_inputs=text_inputs,
                camera_motion=camera_motion,
                sampling_kwargs=update_sampling_kwargs,
                query_predictor_movement_inference_multiplier=query_predictor_movement_inference_multiplier,
            )
            self.current_tracks = output_tracks
        self.current_visibles = new_visibles
        
        return self.current_tracks[:, :, self.current_track_step:]