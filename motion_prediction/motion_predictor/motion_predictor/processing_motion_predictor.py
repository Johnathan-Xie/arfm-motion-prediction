import torch
import numpy as np

from motion_predictor.utils import apply_framewise_function_batched
from transformers import AutoProcessor, AutoTokenizer
from .configuration_motion_predictor import MotionPredictorConfig
import torch.nn.functional as F

from sam2.utils.transforms import SAM2Transforms

class Sam2Processor:
    def __init__(self, image_size=1024):
        self.transforms = SAM2Transforms(resolution=image_size, mask_threshold=0.0)
    
    def __call__(self, x):
        return self.transforms.forward_batch(x)

class ImageProcessorWrapper:
    def __init__(self, image_model_name, image_size=None):
        super().__init__()
        if "sam2" in image_model_name:
            self.image_processor = Sam2Processor(image_size or 1024)
        elif any(keyword in image_model_name for keyword in ["clip", "siglip2"]):
            self.image_processor = AutoProcessor.from_pretrained(image_model_name).image_processor
        else:
            self.image_processor = AutoProcessor.from_pretrained(image_model_name)
   
    def __call__(self, x):
        processor_output = self.image_processor(x)
        if isinstance(self.image_processor, Sam2Processor):
            return processor_output
        else:
            return torch.Tensor(np.array(processor_output["pixel_values"]))

class VideoFrameProcessor:
    def __init__(
        self,
        image_model_name,
        frame_sample_rate,
        sam2_image_size,
    ):
        self.image_processor = ImageProcessorWrapper(image_model_name, image_size=sam2_image_size)        
        self.frame_sample_rate = frame_sample_rate
    
    def __call__(self, videos, max_frames=None, truncate_max=True, start_indices=None):
        if self.frame_sample_rate is not None:
            video_frames = []
            for video in videos:
                frames, fps = video["frames"], video["fps"]
                hop_rate = int(fps) // self.frame_sample_rate
                video_frames.append(frames[::hop_rate])
        else:
            video_frames = [v["frames"] for v in videos]
        
        if start_indices is not None:
            video_frames = [video_frames[i][start_indices[i]:] for i in range(len(video_frames))]
        
        if truncate_max:
            max_frames = min(max_frames or 1000000, *[len(i) for i in video_frames])
        if max_frames is not None:
            video_frames = [i[:max_frames] for i in video_frames]
        processed_frames = apply_framewise_function_batched(lambda x: self.image_processor(x), video_frames)
        return torch.stack(processed_frames)

class TrackProcessor:
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.height = config.height
        self.width = config.width
        self.max_track_length = config.max_track_length
        self.track_subsample_count = config.track_subsample_count
        self.max_height_shift = config.max_height_shift
        self.max_width_shift = config.max_width_shift
        self.continuous_predictor = config.continuous_predictor
        self.track_dimensionality = config.track_dimensionality
        self.prepend_query_points = config.prepend_query_points
        
    def __call__(
        self,
        query_points,
        tracks=None,
        visibles=None,
        start_indices=None,
        visible_min_ratio=None,
        visible_ratios=None,
        movements=None,
        movement_weighting_temperature=None,
        mask_non_visible_tracks=False,
        total_points_needed_before_selection=None,
    ):
        # Total points needed before selection should only be used during inference time
        # During inference we use pseudo tracks which are just the query points repeated for max track length
        # Also pseudo visibles are just always visible for the entire track length
        have_real_tracks = tracks is not None
        have_real_visibles = visibles is not None
        if visible_min_ratio is None:
            visible_min_ratio = self.config.visible_min_ratio
        if movement_weighting_temperature is None:
            movement_weighting_temperature = self.config.movement_weighting_temperature
        query_points = [torch.Tensor(i)[..., 1:self.track_dimensionality + 1] for i in query_points]
        if not have_real_tracks:
            tracks = [i.unsqueeze(-2).expand(-1, self.max_track_length, -1) for i in query_points]
            if movements is not None:
                movements = [i for i in movements]
        else:
            tracks = [torch.Tensor(i)[..., :self.track_dimensionality] for i in tracks]
        if have_real_visibles and mask_non_visible_tracks:
            visibles = [torch.BoolTensor(i) for i in visibles]
        else:
            visibles = [torch.ones(i.shape[:-1], dtype=torch.bool) for i in tracks]
            
        if start_indices is not None:
            assert tracks is not None
            for batch_idx in range(len(tracks)):
                if start_indices[batch_idx] > 0:
                    query_points[batch_idx] = tracks[batch_idx][:, start_indices[batch_idx] - 1]
                    tracks[batch_idx] = tracks[batch_idx][:, start_indices[batch_idx]:]
                    visibles[batch_idx] = visibles[batch_idx][:, start_indices[batch_idx]:]
        
        max_track_length = self.max_track_length
        max_num_tracks = self.track_subsample_count
        # Making padding, truncating, making label mask and attention mask
        attention_mask = []
        label_mask = []
        track_shapes = [t.shape for t in tracks]
        for batch_idx in range(len(tracks)):
            track = tracks[batch_idx]
            num_tracks, track_length, track_dimensionality = track_shapes[batch_idx]
            if track_length < max_track_length:
                # +1 for 1s and -1 for 0s indicates we will be prepending the query point and cutting the last track off
                current_attention_mask = torch.cat((torch.ones((num_tracks, track_length + 1)), torch.zeros((num_tracks, max_track_length - track_length - 1))), dim=1)
                attention_mask.append(current_attention_mask)
                current_label_mask = torch.ones_like(visibles[batch_idx])
                if mask_non_visible_tracks:
                    current_label_mask = current_label_mask & visibles[batch_idx]
                label_mask_padding = torch.zeros((current_label_mask.shape[0], max_track_length - current_label_mask.shape[1],), dtype=torch.bool)
                current_label_mask = torch.cat((current_label_mask, label_mask_padding), dim=1)
                label_mask.append(current_label_mask)
                track_padding = -torch.ones((num_tracks, max_track_length - track_length, track_dimensionality), device=tracks[batch_idx].device)
                tracks[batch_idx] = torch.cat((track, track_padding), dim=1)
                
                visibles_padding = torch.zeros((num_tracks, max_track_length - visibles[batch_idx].shape[1]), dtype=torch.bool)
                visibles[batch_idx] = torch.cat((visibles[batch_idx], visibles_padding), dim=1)
            else:
                tracks[batch_idx] = track[:, :max_track_length]
                visibles[batch_idx] = visibles[batch_idx][:, :max_track_length]

                current_attention_mask = torch.ones(visibles[batch_idx].shape)
                attention_mask.append(current_attention_mask)
                current_label_mask = torch.ones_like(visibles[batch_idx])
                
                if mask_non_visible_tracks:
                    current_label_mask = current_label_mask & visibles[batch_idx]
                label_mask.append(current_label_mask)
            if num_tracks < max_num_tracks:
                if total_points_needed_before_selection is not None:
                    support_points = total_points_needed_before_selection - num_tracks
                    y = torch.randint(0, self.height, (support_points, 1))
                    x = torch.randint(0, self.width, (support_points, 1))
                    support_points = torch.cat((y, x), dim=-1)
                    query_points[batch_idx] = torch.cat((query_points[batch_idx], support_points), dim=0)

                    track_padding = -torch.ones((max_num_tracks - num_tracks, max_track_length, track_dimensionality))
                    tracks[batch_idx] = torch.cat((tracks[batch_idx], track_padding), dim=0)

                    visibles_padding = torch.ones((max_num_tracks - num_tracks, max_track_length), dtype=torch.bool)
                    visibles[batch_idx] = torch.cat((visibles[batch_idx], visibles_padding), dim=0)

                    attention_mask_padding = torch.ones((max_num_tracks - num_tracks, max_track_length))
                    attention_mask[batch_idx] = torch.cat((attention_mask[batch_idx], attention_mask_padding), dim=0)
                    
                    label_mask_padding = torch.zeros((max_num_tracks - num_tracks, max_track_length), dtype=torch.bool)
                    label_mask[batch_idx] = torch.cat((label_mask[batch_idx], label_mask_padding), dim=0)
                else:
                    track_padding = -torch.ones((max_num_tracks - num_tracks, max_track_length, track_dimensionality))
                    tracks[batch_idx] = torch.cat((tracks[batch_idx], track_padding), dim=0)
                    query_points_padding = -torch.ones((max_num_tracks - num_tracks), 2)
                    query_points[batch_idx] = torch.cat((query_points[batch_idx], query_points_padding), dim=0)

                    visibles_padding = torch.zeros((max_num_tracks - num_tracks, max_track_length), dtype=torch.bool)
                    visibles[batch_idx] = torch.cat((visibles[batch_idx], visibles_padding), dim=0)
                    
                    attention_mask_padding = torch.zeros((max_num_tracks - num_tracks, max_track_length))
                    attention_mask[batch_idx] = torch.cat((attention_mask[batch_idx], attention_mask_padding), dim=0)
                    
                    label_mask_padding = torch.zeros((max_num_tracks - num_tracks, max_track_length), dtype=torch.bool)
                    label_mask[batch_idx] = torch.cat((label_mask[batch_idx], label_mask_padding), dim=0)
        
        all_track_indices = [torch.arange(len(i)) for i in tracks]
        # Removing tracks with portion of track visible less than visible_min_ratio
        if visible_min_ratio > 0 and (have_real_visibles or visible_ratios is not None):
            for batch_idx in range(len(query_points)):
                if have_real_visibles:
                    sample_visible_ratios = visibles[batch_idx].float().mean(dim=1)
                else:
                    sample_visible_ratios = visible_ratios[batch_idx]
                visibles_mask = sample_visible_ratios > visible_min_ratio
                # Ensure we keep at least track subsample count
                if visibles_mask.sum() < self.track_subsample_count:
                    indices = sample_visible_ratios.topk(k=self.track_subsample_count).indices
                else:
                    indices = visibles_mask.nonzero(as_tuple=True)

                tracks[batch_idx] = tracks[batch_idx][indices]
                label_mask[batch_idx] = label_mask[batch_idx][indices]
                visibles[batch_idx] = visibles[batch_idx][indices]
                query_points[batch_idx] = query_points[batch_idx][indices]
                attention_mask[batch_idx] = attention_mask[batch_idx][indices]
                all_track_indices[batch_idx] = all_track_indices[batch_idx][indices]
                if movements is not None:
                    movements[batch_idx] = movements[batch_idx][indices]
            
        if self.prepend_query_points:
            full_tracks = [torch.cat((query_points[batch_idx].unsqueeze(1), tracks[batch_idx]), dim=1) for batch_idx in range(len(tracks))]
            visibles = [torch.cat((torch.ones((visibles[batch_idx].shape[0], 1), dtype=visibles[batch_idx].dtype), visibles[batch_idx]), dim=-1) for batch_idx in range(len(tracks))]
        else:
            full_tracks = tracks
        # Random or movement weighted sampling
        for batch_idx in range(len(tracks)):
            # No need to subsample
            if tracks[batch_idx].shape[0] <= self.track_subsample_count:
                continue
            num_tracks, track_length, track_dimensionality = track_shapes[batch_idx]
            if movement_weighting_temperature > 0 and (have_real_tracks or movements is not None):
                visibles_mask = visibles[batch_idx][:, :-1] & visibles[batch_idx][:, 1:]
                if have_real_tracks:
                    sample_movements = ((full_tracks[batch_idx][:, :-1] - full_tracks[batch_idx][:, 1:]) ** 2) * visibles_mask.unsqueeze(-1).type(torch.float)
                    sample_movements = sample_movements.sum((-1, -2)) / (visibles_mask.sum(dim=1) + 1)
                else:
                    sample_movements = movements[batch_idx]
                logits = sample_movements * movement_weighting_temperature
                logits = logits.clip(0, 10)
                probs = F.softmax(logits, dim=0).numpy()
            else:
                probs = None
            
            if num_tracks < max_num_tracks and total_points_needed_before_selection is not None:
                actual_tracks = np.arange(num_tracks)
                support_tracks = np.random.choice(len(tracks[batch_idx]) - num_tracks, size=(self.track_subsample_count - num_tracks,), replace=False, p=probs) + num_tracks
                track_indices = np.concatenate((actual_tracks, support_tracks), axis=0).astype(np.uint32)
            else:
                track_indices = np.random.choice(len(tracks[batch_idx]), size=(self.track_subsample_count,), replace=False, p=probs)
            
            full_tracks[batch_idx] = full_tracks[batch_idx][track_indices]

            query_points[batch_idx] = query_points[batch_idx][track_indices]
            label_mask[batch_idx] = label_mask[batch_idx][track_indices]
            attention_mask[batch_idx] = attention_mask[batch_idx][track_indices]
            all_track_indices[batch_idx] = all_track_indices[batch_idx][track_indices]

        full_tracks = torch.stack(full_tracks)
        attention_mask = torch.stack(attention_mask)
        label_mask = torch.stack(label_mask)
        all_track_indices = torch.stack(all_track_indices)
        # Discretizing if predictor is not continuous (probably going to be deprecated)
        if not self.continuous_predictor:
            full_tracks[..., 0] = full_tracks[..., 0].clip(0, self.height - 1)
            full_tracks[..., 1] = full_tracks[..., 1].clip(0, self.width - 1)
            full_tracks = full_tracks.round().to(torch.long)
        inputs, labels = full_tracks[:, :, :-1], full_tracks[:, :, 1:]
        
        # Reducing track movement based on maximum shift allowed. Can be useful for catching potentially erroneous tracks
        if self.max_height_shift is not None:
            label_mask = label_mask & ((full_tracks[:, :, 1:, 0] - full_tracks[:, :, :-1, 0]).abs() < self.max_height_shift)

        if self.max_width_shift is not None:
            label_mask = label_mask & ((full_tracks[:, :, 1:, 1] - full_tracks[:, :, :-1, 1]).abs() < self.max_width_shift)

        # This is for inference where we only keep query points
        if not have_real_tracks:
            inputs = inputs[:, :, :1]
        return inputs, labels, attention_mask, label_mask, all_track_indices

class MotionPredictorProcessor:
    def __init__(self, config):
        self.config = config
        self.video_processor = VideoFrameProcessor(
            image_model_name=config.image_model_name,
            frame_sample_rate=config.frame_sample_rate,
            sam2_image_size=config.sam2_image_size,
        )
        if config.global_image_model_name is not None:
            self.global_video_processor = VideoFrameProcessor(
                image_model_name=config.global_image_model_name,
                frame_sample_rate=config.frame_sample_rate,
                sam2_image_size=config.sam2_image_size,
            )
        else:
            self.global_video_processor = None
        self.track_processor = TrackProcessor(config)
        if config.text_encoder_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_name)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        return MotionPredictorProcessor(MotionPredictorConfig.from_pretrained(pretrained_model_name_or_path))

    def modify_to_create_sampling_inputs(self, inputs):
        keys = ["pixel_values", "global_pixel_values", "attention_mask", "input_ids", "camera_motion", "track_rate", "text_input_ids", "text_attention_mask"]
        batch = {k:inputs.get(k) for k in inputs if k in keys if inputs.get(k) is not None}
        
        return batch

    def __call__(
        self,
        videos=None,
        tracks=None,
        query_points=None,
        max_frames=None,
        visibles=None,
        camera_motion=None,
        text_inputs=None,
        start_indices=None,
        video_start_indices=None,
        visible_min_ratio=None,
        movement_weighting_temperature=None,
        mask_non_visible_tracks=False,
        total_points_needed_before_selection=None,
        for_sampling=False,
        movements=None,
        visible_ratios=None
    ):
        output = dict()
        
        if videos is not None:
            videos = [{"frames": i} if not isinstance(i, dict) else i for i in videos]
            output["pixel_values"] = self.video_processor(videos, max_frames=max_frames, start_indices=video_start_indices or start_indices)
            if self.global_video_processor is not None:
                output["global_pixel_values"] = self.global_video_processor(videos, max_frames=max_frames, start_indices=video_start_indices or start_indices)
            else:
                output["global_pixel_values"] = output["pixel_values"]
        if tracks is not None or query_points is not None:
            (
                output["input_ids"],
                output["labels"],
                output["attention_mask"],
                output["label_mask"],
                all_track_indices
            ) = self.track_processor(
                query_points=query_points,
                tracks=tracks,
                visibles=visibles,
                start_indices=start_indices,
                visible_min_ratio=visible_min_ratio,
                movement_weighting_temperature=movement_weighting_temperature,
                mask_non_visible_tracks=mask_non_visible_tracks,
                total_points_needed_before_selection=total_points_needed_before_selection,
                movements=movements,
                visible_ratios=visible_ratios,
            )
        if text_inputs is not None:
            text_input_ids, text_attention_mask = [], []
            for text in text_inputs:
                if text is not None and self.config.text_encoder_name is not None:
                    tokenizer_output = self.tokenizer(
                        [text],
                        padding="max_length",
                        truncation=True,
                        max_length=self.config.text_encoder_max_seq_length,
                        return_tensors="pt"
                    )
                    text_input_ids.append(tokenizer_output["input_ids"][0])
                    text_attention_mask.append(tokenizer_output["attention_mask"][0])
                elif self.config.text_encoder_name is not None:
                    text_input_ids.append(torch.full((self.config.text_encoder_max_seq_length, ), fill_value=-1))
                    text_attention_mask.append(torch.ones((self.config.text_encoder_max_seq_length, )))
                else:
                    text_input_ids.append(torch.full((1, ), fill_value=-1))
                    text_attention_mask.append(torch.ones((1,)))
                output["text_input_ids"] = torch.stack(text_input_ids)
                output["text_attention_mask"] = torch.stack(text_attention_mask)
        else:
            output["text_input_ids"] = torch.full((len(query_points), self.config.text_encoder_max_seq_length), fill_value=-1)
            output["text_attention_mask"] = torch.ones((len(query_points), self.config.text_encoder_max_seq_length))
        if camera_motion is not None:
            mapping = {False: 0, True: 1}
            output["camera_motion"] = torch.LongTensor([mapping[i] if i is not None else 2 for i in camera_motion])
        else:
            output["camera_motion"] = torch.LongTensor([2] * len(videos))

        output["track_rate"] = torch.FloatTensor([float(i["fps"]) if i.get("fps") is not None else self.config.default_track_rate for i in videos])
        if for_sampling:
            output = self.modify_to_create_sampling_inputs(output)
        return output
    
AutoProcessor.register(MotionPredictorConfig, MotionPredictorProcessor)
