import torch
import numpy as np

from transformers import AutoProcessor, AutoTokenizer
from .configuration_query_predictor import QueryPredictorConfig
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

class QueryPredictorTrackProcessor:
    def __init__(
        self,
        config,
    ):
        self.height = config.height
        self.width = config.width
        self.track_dimensionality = config.track_dimensionality
        self.track_subsample_count = config.track_subsample_count
        self.movement_mean = config.movement_mean
        self.movement_std = config.movement_std
        self.max_track_length = config.max_track_length
        self.prepend_query_points = config.prepend_query_points

    def __call__(
        self,
        query_points,
        tracks=None,
        visibles=None,
    ):        
        have_real_tracks = tracks is not None
        have_real_visibles = visibles is not None

        query_points = [torch.Tensor(i)[..., 1:self.track_dimensionality + 1] for i in query_points]
        if not have_real_tracks:
            tracks = [i.unsqueeze(-2).expand(-1, 2, -1) for i in query_points]
        else:
            tracks = [torch.Tensor(i)[..., :self.track_dimensionality] for i in tracks]
        if have_real_visibles:
            visibles = [torch.BoolTensor(i) for i in visibles]
        else:
            visibles = [torch.ones(i.shape[:-1], dtype=torch.bool) for i in tracks]
        
        max_num_tracks = self.track_subsample_count
        # Making padding, truncating, making label mask and attention mask
        attention_mask = []
        label_mask = []

        track_shapes = [t.shape for t in tracks]
        max_track_length = self.max_track_length

        for batch_idx in range(len(query_points)):
            num_tracks = len(query_points[batch_idx])
            num_tracks, track_length, track_dimensionality = track_shapes[batch_idx]
            if track_length > max_track_length:
                tracks[batch_idx] = tracks[batch_idx][:, :max_track_length]
                visibles[batch_idx] = visibles[batch_idx][:, :max_track_length]

            if num_tracks < max_num_tracks:
                if tracks is not None and visibles is not None:
                    track_padding = -torch.ones((max_num_tracks - num_tracks, track_length, track_dimensionality), device=tracks[batch_idx].device)
                    tracks[batch_idx] = torch.cat((tracks[batch_idx], track_padding), dim=0)

                    visibles_padding = torch.zeros((max_num_tracks - num_tracks, track_length), dtype=torch.bool, device=visibles[batch_idx].device)
                    visibles[batch_idx] = torch.cat((visibles[batch_idx], visibles_padding), dim=0)
                
                query_points_padding = -torch.ones((max_num_tracks - num_tracks, 2), device=query_points[batch_idx].device)
                query_points[batch_idx] = torch.cat((query_points[batch_idx], query_points_padding), dim=0)
                
                attention_mask.append(torch.cat((torch.ones((num_tracks)), torch.zeros((max_num_tracks - num_tracks))), dim=0))
                
                label_mask.append(torch.cat((torch.ones((num_tracks), dtype=torch.bool), torch.zeros((max_num_tracks - num_tracks), dtype=torch.bool)), dim=0))
            else:
                if tracks is not None and visibles is not None:
                    tracks[batch_idx] = tracks[batch_idx][:max_num_tracks]
                    visibles[batch_idx] = visibles[batch_idx][:max_num_tracks]

                attention_mask.append(torch.ones((max_num_tracks, )))
                label_mask.append(torch.ones((max_num_tracks, ), dtype=torch.bool))
        
        if self.prepend_query_points:
            full_tracks = [torch.cat((query_points[batch_idx].unsqueeze(1), tracks[batch_idx]), dim=1) for batch_idx in range(len(tracks))]
            visibles = [torch.cat((torch.ones((visibles[batch_idx].shape[0], 1), dtype=visibles[batch_idx].dtype), visibles[batch_idx]), dim=-1) for batch_idx in range(len(tracks))]
        else:
            full_tracks = tracks
        
        if have_real_tracks and have_real_visibles:
            movements = []
            visible_ratios = []
            for idx in range(len(full_tracks)):
                visibles_mask = visibles[batch_idx][:, :-1] & visibles[batch_idx][:, 1:]
                sample_movements = ((full_tracks[batch_idx][:, :-1] - full_tracks[batch_idx][:, 1:]) ** 2) * visibles_mask.unsqueeze(-1).type(torch.float)
                sample_movements = (sample_movements.sum((-1, -2)) / (visibles_mask.sum(dim=1) + 1) - self.movement_mean) / self.movement_std
                
                movements.append(sample_movements)
                visible_ratios.append(visibles[batch_idx].type(torch.float).mean(dim=-1))
            movements = torch.stack(movements)
            visible_ratios = torch.stack(visible_ratios)
        else:
            movements = None
            visible_ratios = None

        attention_mask = torch.stack(attention_mask)
        label_mask = torch.stack(label_mask)
        query_points = torch.stack(query_points)
        query_points[..., :2] = query_points[..., :2]

        return query_points, attention_mask, label_mask, movements, visible_ratios

class QueryPredictorProcessor:
    def __init__(self, config):
        self.config = config
        self.image_processor = ImageProcessorWrapper(config.image_model_name, image_size=config.encoder_image_size)
        self.track_dimensionality = config.track_dimensionality
        self.query_processor_track_processor = QueryPredictorTrackProcessor(config)
        if config.text_encoder_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_name)
        self.canonical_track_rate = config.canonical_track_rate
        self.predict_visible_ratios = config.predict_visible_ratios

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        return QueryPredictorProcessor(QueryPredictorConfig.from_pretrained(pretrained_model_name_or_path))

    def __call__(
        self,
        images=None,
        tracks=None,
        query_points=None,
        visibles=None,
        text_inputs=None,
        camera_motion=None,
        track_rate=None,
    ):
        # Start indices augmentation needs to be taken care of by ProcessorWrapper
        output = dict()
        if images is not None:
            output["pixel_values"] = self.image_processor(np.stack(images))
        (
            output["query_points"],
            output["attention_mask"],
            output["label_mask"],
            movements,
            visible_ratios
        ) = self.query_processor_track_processor(
            query_points=query_points,
            tracks=tracks,
            visibles=visibles,
        )
        if track_rate is not None and self.canonical_track_rate is not None:
            movements = movements / ((self.canonical_track_rate / torch.FloatTensor(track_rate).unsqueeze(1)) ** 2)
        if movements is not None and visible_ratios is not None:
            if self.predict_visible_ratios:
                output["labels"] = torch.stack([movements, visible_ratios], dim=-1)
            else:
                output["labels"] = movements.unsqueeze(-1)
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
            output["camera_motion"] = torch.LongTensor([2] * len(query_points))
        
        return output