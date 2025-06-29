from torch import nn

import torch
import numpy as np
import math
from .utils import apply_framewise_function_batched

from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from transformers.activations import ACT2FN

from sam2.sam2_image_predictor import SAM2ImagePredictor

from transformers import AutoModel, AutoTokenizer

@dataclass
class VideoEncoderOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    video_feature_map: Optional[torch.FloatTensor] = None
    image_features: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

def get_position_to_embeddings(
    num_positions,
    positions,
    concatenate_position=True,
    num_bands=None,
    spacing="linear",
):
    if num_bands is None:
        num_bands = math.ceil(num_positions / 2)
    
    if spacing == "linear":
        scales = torch.pi * (torch.linspace(0, 0.5, num_bands, device=positions.device)).unsqueeze(0) * positions.unsqueeze(1)
    elif spacing == "square":
        scales = torch.pi * ((torch.linspace(1, torch.sqrt(torch.tensor(num_positions / 2), device=positions.device), num_bands) ** 2) / num_positions).unsqueeze(0) * positions.unsqueeze(1)
    elif spacing == "log":
        scales = torch.pi * (torch.logspace(1, torch.log2(torch.tensor(num_positions / 2), device=positions.devicex), num_bands, base=2) / num_positions).unsqueeze(0) * positions.unsqueeze(1)
    else:
        raise ValueError(f"Unknown spacing type {spacing}")
    embeddings = torch.cat([torch.sin(scales), torch.cos(scales)], dim=-1)
    if concatenate_position:
        embeddings = torch.cat([positions.unsqueeze(1) / num_positions, embeddings], dim=-1)
    return embeddings

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class SinusoidalEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dropout_prob=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        if dropout_prob > 0:
            self.token_drop_embedding = nn.Parameter(torch.randn((frequency_embedding_size,)))
        self.frequency_embedding_size = frequency_embedding_size
        self.dropout_prob = dropout_prob

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def embedding_drop(self, t_freq, force_drop_indices=None):
        """
        Drops tokens to enable classifier-free guidance.
        """
        if force_drop_indices is None:
            drop_ids = torch.rand(t_freq.shape[0], device=t_freq.device) < self.dropout_prob
        else:
            drop_ids = force_drop_indices == 1
        t_freq[drop_ids] = self.token_drop_embedding
        return t_freq

    def forward(self, t, force_drop_indices=None):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if self.dropout_prob > 0 or force_drop_indices is not None:
            t_freq = self.embedding_drop(t_freq, force_drop_indices)
        
        t_emb = self.mlp(t_freq)
        return t_emb

class DiscreteEmbedder(nn.Module):
    """
    Embeds discrete values into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_indices=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_indices is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_indices == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, force_drop_indices=None):
        if self.dropout_prob > 0 or force_drop_indices is not None:
            labels = self.token_drop(labels, force_drop_indices)
        embeddings = self.embedding_table(labels)
        return embeddings
        

class FourierPositionalEncoding2D(nn.Module):
    def __init__(
        self,
        height=256,
        width=256,
        **embedding_kwargs,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.height_embeddings = nn.Parameter(get_position_to_embeddings(height, torch.arange(height), **embedding_kwargs), requires_grad=False)
        self.width_embeddings = nn.Parameter(get_position_to_embeddings(width, torch.arange(width), **embedding_kwargs), requires_grad=False)
        self.embedding_kwargs = embedding_kwargs
        
    def position_to_embedding(self, position):
        if isinstance(position, torch.LongTensor):
            width_embeddings = self.width_embeddings[position[..., 0]]
            height_embeddings = self.height_embeddings[position[..., 1]]
            return torch.cat((height_embeddings, width_embeddings), dim=-1)
        else:
            original_position_shape = position.shape[:-1]
            new_position = position.clone()
            new_position = new_position.view(-1, position.shape[-1])
            width_embeddings = get_position_to_embeddings(self.width, new_position[:, 0], **self.embedding_kwargs)
            height_embeddings = get_position_to_embeddings(self.height, new_position[:, 1], **self.embedding_kwargs)
            embeddings = torch.cat((height_embeddings, width_embeddings), dim=-1)
            embeddings = embeddings.view(original_position_shape + (-1, ))
        return embeddings
    
    def embedding_to_position_similarity(self, embeddings):
        height_similarity = torch.matmul(embeddings[..., :embeddings.shape[-1] // 2], self.height_embeddings.T) / self.height_embeddings.shape[-1]
        width_similarity = torch.matmul(embeddings[..., embeddings.shape[-1] // 2:], self.width_embeddings.T) / self.width_embeddings.shape[-1]
        return height_similarity, width_similarity

class gMLP(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, intermediate_size)
        self.up_proj = nn.Linear(input_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, output_size)
        self.act_fn = ACT2FN["gelu"]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class ShiftScaleFourierPredictor(nn.Module):
    def __init__(
        self,
        feature_size,
        height=256,
        width=256,
        max_height_shift=None,
        max_width_shift=None,
        concatenate_original_position=True,
        norm_feature=True,
        predictor_type="linear",
        track_dimensionality=2,
        **embedding_kwargs,
    ):
        super().__init__()
        self.fourier_embeddings = FourierPositionalEncoding2D(height, width, **embedding_kwargs)
        self.embedding_size = self.fourier_embeddings.height_embeddings.shape[-1] + self.fourier_embeddings.width_embeddings.shape[-1]
        self.height = height
        self.width = width
        self.max_height_shift = max_height_shift
        self.max_width_shift = max_width_shift
        # extra 2 for shift
        if concatenate_original_position:
            feature_size += self.embedding_size
        
        if norm_feature:
            self.norm = nn.LayerNorm(feature_size)
        else:
            self.norm = nn.Identity()
        if predictor_type == "linear":
            self.predictor = nn.Linear(feature_size, self.embedding_size + track_dimensionality)
        elif predictor_type == "gmlp":
            self.predictor = gMLP(feature_size, feature_size * 3, self.embedding_size + track_dimensionality)
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
        self.concatenate_original_position = concatenate_original_position
        
    def forward(self, input_features, original_positions, **kwargs):
        # Last dimension being of shape 2 probably means they're absolute positions rather than embeddings
        if self.concatenate_original_position:
            input_features = torch.cat((input_features, self.fourier_embeddings.position_to_embedding(original_positions)), dim=-1)
        input_features = self.norm(input_features)
        
        shift_scale_pred = self.predictor(input_features)
        shift, scale = shift_scale_pred[..., :2], shift_scale_pred[..., 2:]
        
        shifted_positions = original_positions + shift
        shifted_embeddings = self.fourier_embeddings.position_to_embedding(shifted_positions.reshape(-1, shifted_positions.shape[-1]))
        shifted_embeddings = shifted_embeddings.view(shifted_positions.shape[:-1] + (-1, ))
        scaled_embeddings = shifted_embeddings * scale
        height_logits, width_logits = self.fourier_embeddings.embedding_to_position_similarity(scaled_embeddings)
        if self.max_height_shift is not None:
            original_heights = original_positions[..., 1]

            height_mask = torch.arange(self.height, device=height_logits.device)[None, None, None].expand_as(height_logits)

            min_heights = (original_heights.unsqueeze(-1) - self.max_height_shift).clip(min=0)
            max_heights = (original_heights.unsqueeze(-1) + self.max_height_shift).clip(max=self.height - 1)
            height_mask = (height_mask < min_heights) | (height_mask > max_heights)
            height_logits[height_mask] = torch.finfo(height_logits.dtype).min
        
        if self.max_width_shift is not None:
            original_widths = original_positions[..., 0]

            width_mask = torch.arange(self.width, device=height_logits.device)[None, None, None].expand_as(width_logits)

            min_widths = (original_widths.unsqueeze(-1) - self.max_width_shift).clip(min=0)
            max_widths = (original_widths.unsqueeze(-1) + self.max_width_shift).clip(max=self.width - 1)
            width_mask = (width_mask < min_widths) | (width_mask > max_widths)
            width_logits[width_mask] = torch.finfo(width_logits.dtype).min
            
        return height_logits, width_logits
    
class ShiftFourierPredictor(nn.Module):
    def __init__(
        self,
        feature_size,
        height=256,
        width=256,
        concatenate_original_position=True,
        norm_feature=True,
        predictor_type="linear",
        track_dimensionality=2,
        **embedding_kwargs,
    ):
        super().__init__()
        self.fourier_embeddings = FourierPositionalEncoding2D(height, width, **embedding_kwargs)
        self.embedding_size = self.fourier_embeddings.height_embeddings.shape[-1] + self.fourier_embeddings.width_embeddings.shape[-1]
        self.height = height
        self.width = width
        # extra 2 for shift
        if concatenate_original_position:
            feature_size += self.embedding_size
        
        if norm_feature:
            self.norm = nn.LayerNorm(feature_size)
        else:
            self.norm = nn.Identity()
        if predictor_type == "linear":
            self.predictor = nn.Linear(feature_size, track_dimensionality)
        elif predictor_type == "gmlp":
            self.predictor = gMLP(feature_size, feature_size * 3, track_dimensionality)
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
        self.concatenate_original_position = concatenate_original_position
        
    def forward(self, input_features, original_positions, **kwargs):
        if self.concatenate_original_position:
            input_features = torch.cat((input_features, self.fourier_embeddings.position_to_embedding(original_positions)), dim=-1)
        input_features = self.norm(input_features)
        
        shift = self.predictor(input_features)
        
        return shift


class OutputPredictor(nn.Module):
    def __init__(
        self,
        feature_size,
        output_size=2,
        height=256,
        width=256,
        concatenate_original_position=True,
        norm_feature=False,
        predictor_type="linear",
        **embedding_kwargs,
    ):
        super().__init__()
        self.fourier_embeddings = FourierPositionalEncoding2D(height, width, **embedding_kwargs)
        self.embedding_size = self.fourier_embeddings.height_embeddings.shape[-1] + self.fourier_embeddings.width_embeddings.shape[-1]
        self.height = height
        self.width = width
        # extra 2 for shift
        if concatenate_original_position:
            feature_size += self.embedding_size
        
        if norm_feature:
            self.norm = nn.LayerNorm(feature_size)
        else:
            self.norm = nn.Identity()
        if predictor_type == "linear":
            self.predictor = nn.Linear(feature_size, output_size)
        elif predictor_type == "gmlp":
            self.predictor = gMLP(feature_size, feature_size * 3, output_size)
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
        self.concatenate_original_position = concatenate_original_position
        
    def forward(self, input_features, original_positions, global_conditioning=None, **kwargs):
        if isinstance(global_conditioning, torch.Tensor):
            input_features = torch.cat([input_features, global_conditioning.unsqueeze(1).expand(-1, input_features.shape[1], -1, -1)], dim=-1)
        if self.concatenate_original_position:
            input_features = torch.cat((input_features, self.fourier_embeddings.position_to_embedding(original_positions)), dim=-1)
        input_features = self.norm(input_features)
        output = self.predictor(input_features)
        
        return output



class ImageModelFeatureExtraction(nn.Module):
    def __init__(self, image_model_name, sam2_image_size=512, sam2_original_image_size=1024, freeze_image_model=True):
        super().__init__()
        model_type = None
        if "sam2" in image_model_name:
            self.image_model = SAM2ImagePredictor.from_pretrained(image_model_name)
            self._bb_feat_sizes = self.image_model._bb_feat_sizes
            if sam2_image_size != sam2_original_image_size:
                downsize_ratios = [((sam2_original_image_size / i[0]), (sam2_original_image_size / i[1])) for i in self._bb_feat_sizes]
                self._bb_feat_sizes = [(int(math.ceil(sam2_image_size / i[0])), int(math.ceil(sam2_image_size / i[1]))) for i in downsize_ratios]
            self.image_model_hidden_size = 32 + 64 + 256
            self.image_model = self.image_model.model
            model_type = "sam2"
        elif "clip" in image_model_name:
            self.image_model = AutoModel.from_pretrained(image_model_name).vision_model
            self.image_model_hidden_size = self.image_model.config.hidden_size
            model_type = "clip"
        elif "siglip2" in image_model_name:
            self.image_model = AutoModel.from_pretrained(image_model_name).vision_model
            self.image_model_hidden_size = self.image_model.config.hidden_size
            self.model_type = "siglip2"
        else:
            self.image_model = AutoModel.from_pretrained(image_model_name)
            self.image_model_hidden_size = self.image_model.config.hidden_size
            model_type = "other"

        self.model_type = model_type
        if freeze_image_model:
            for p in self.image_model.parameters():
                p.requires_grad = False
        
    def forward(self, x):
        if self.model_type == "sam2":
            backbone_out = self.image_model.forward_image(x)
            _, vision_feats, _, _ = self.image_model._prepare_backbone_features(backbone_out)            
            feats = [
                    feat.permute(1, 2, 0).view(x.shape[0], -1, *feat_size).permute(0, 2, 3, 1)
                    for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
                ][::-1]
            return feats
        else:
            model_output = self.image_model(x)
            feature_map = model_output.last_hidden_state
            # remove cls token, siglip2 no cls token in hidden state
            if self.model_type != "siglip2":
                feature_map = feature_map[:, 1:]
            # make grid
            feature_map_height = x.shape[-2] // self.image_model.config.patch_size
            feature_map_width = x.shape[-1] // self.image_model.config.patch_size
            feature_map = feature_map.view(feature_map.shape[0], feature_map_height, feature_map_width, -1)
            return feature_map


def image_features_to_sequence(frame_features):
    frame_features = frame_features.moveaxis(1, 3)
    sequence = frame_features.reshape((-1, ) + frame_features.shape[-2:])
    return sequence

def sequence_to_frame_features(sequence, feature_map_shape):
    original_batch_size = sequence.shape[0] // (feature_map_shape[0] * feature_map_shape[1])
    frame_features = sequence.view((original_batch_size, feature_map_shape[0], feature_map_shape[1]) + sequence.shape[1:])
    frame_features = frame_features.moveaxis(3, 1)
    return frame_features

class ImageAggregationCausalVideoModel(nn.Module):
    def __init__(
        self,
        image_model_name,
        freeze_image_model=True,
        time_dimension_model_config=None,
        encoder_pooler_type="none",
        sam2_image_size=512,
        image_model_batch_size=None,
    ):
        super().__init__()        
        self.image_model_batch_size = image_model_batch_size
        self.image_model = ImageModelFeatureExtraction(
            image_model_name,
            freeze_image_model=freeze_image_model,
            sam2_image_size=sam2_image_size,
        )
        self.image_model_hidden_size = self.image_model.image_model_hidden_size
        if time_dimension_model_config is not None:
            if self.image_model_hidden_size != time_dimension_model_config.hidden_size:
                self.image_to_encoder_projection = nn.Linear(self.image_model_hidden_size, time_dimension_model_config.hidden_size)
            else:
                self.image_to_encoder_projection = nn.Identity()
            self.time_dimension_model = AutoModel.from_config(time_dimension_model_config)
        else:
            self.time_dimension_model = None
        
        if encoder_pooler_type == "mean":
            self.pooler = lambda features: features.mean(1)
        elif encoder_pooler_type == "last":
            self.pooler = lambda features: features[:, -1]
        elif encoder_pooler_type == "none":
            self.pooler = lambda features: features
        else:
            raise ValueError(f"Unknown encoder_pooler_type: {encoder_pooler_type}")
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
    ):
        image_features = apply_framewise_function_batched(self.image_model, pixel_values, processing_batch_size=self.image_model_batch_size)
        if self.time_dimension_model is not None:
            feature_map_shape = image_features.shape[2:4]
            feature_sequence = image_features_to_sequence(image_features)
            time_dimension_outputs = self.time_dimension_model(inputs_embeds=feature_sequence)
            feature_sequence = time_dimension_outputs.last_hidden_state
            image_features = sequence_to_frame_features(feature_sequence, feature_map_shape=feature_map_shape)
            
            pooled_features = self.pooler(image_features)
            return VideoEncoderOutput(
                video_feature_map=pooled_features,
                image_features=image_features,
                past_key_values=time_dimension_outputs.past_key_values
            )
        else:
            pooled_features = self.pooler(image_features)
            return VideoEncoderOutput(
                video_feature_map=pooled_features,
                image_features=image_features,
            )

class HFTextEncoder(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        output_size,
        pooled_output_size,
        max_seq_length,
        freeze_text_encoder=True,
        conditioning_drop_prob=0,
    ):
        super().__init__()
        self.text_encoder_name = text_encoder_name
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name, torch_dtype=torch.float32)
        if any(model in self.text_encoder_name for model in ("clip", "siglip2")):
            self.text_encoder = self.text_encoder.text_model
        
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        self.projection = nn.Linear(self.text_encoder.config.hidden_size, output_size)
        self.pooled_projection = nn.Linear(self.text_encoder.config.hidden_size, pooled_output_size)
        self.conditioning_drop_prob = conditioning_drop_prob

        self.embedding_mask_tokens = nn.Parameter(torch.randn((max_seq_length, output_size)))
        self.pooled_mask_token = nn.Parameter(torch.randn((output_size,)))
        
    def token_drop(self, text_embedding, pooled_text_embedding, attention_mask, force_drop_indices=None):
        if force_drop_indices is None:
            drop_indices = torch.rand((text_embedding.shape[0],)) > self.conditioning_drop_prob
        else:
            drop_indices = force_drop_indices == 1

        text_embedding[drop_indices] = self.embedding_mask_tokens
        pooled_text_embedding[drop_indices] = self.pooled_mask_token
        # We repeat the dropped token many times so that we can use the space as pooling tokens
        attention_mask_for_dropped = torch.ones_like(attention_mask[0])
        attention_mask[drop_indices] = attention_mask_for_dropped
        
        return text_embedding, pooled_text_embedding, attention_mask

    def forward(self, input_ids, attention_mask, force_drop_indices=None):
        # return batch_size, embedding_seq_len, hidden_size
        model_input_ids = input_ids.clone()
        model_input_ids[input_ids == -1] = self.tokenizer.pad_token_id
        if any(model in self.text_encoder_name for model in ("bert", "clip", "siglip2")):
            model_output = self.text_encoder(input_ids=model_input_ids, attention_mask=attention_mask)
            text_embedding = model_output.last_hidden_state.to(dtype=torch.float32)
            pooled_text_embedding = model_output.pooler_output.to(dtype=torch.float32)
        else:
            raise ValueError(f"Unknown text encoder type {type(self.text_encoder)}")

        text_embedding = self.projection(text_embedding).to(dtype=torch.float32)
        pooled_text_embedding = self.pooled_projection(pooled_text_embedding).to(dtype=torch.float32)
        if (self.training and self.conditioning_drop_prob) or (force_drop_indices is not None):
            text_embedding, pooled_text_embedding, attention_mask = self.token_drop(text_embedding, pooled_text_embedding, attention_mask)
        
        text_embedding, pooled_text_embedding, attention_mask = self.token_drop(
            text_embedding, pooled_text_embedding, attention_mask, force_drop_indices=(input_ids == -1).all(dim=1)
        )
        return text_embedding, pooled_text_embedding, attention_mask


def get_batch_timestep_indices(points, visible_frame_timesteps=None):
    batch_size, timesteps, num_points_per_sample, _ = points.shape
    batch_indices = torch.arange(batch_size, device=points.device).repeat_interleave(timesteps * num_points_per_sample)
    timestep_indices = torch.arange(timesteps, device=points.device).repeat(batch_size).repeat_interleave(num_points_per_sample)
    # common case where there are more point time steps than visible frame timesteps and we just take last frame
    if visible_frame_timesteps is not None and visible_frame_timesteps < timesteps:
        timestep_indices = timestep_indices.clip(max=visible_frame_timesteps - 1)
    return batch_indices, timestep_indices

def pointwise_nearest_interpolation(grid_features, upsampled_size, points, batch_indices, timestep_indices):
    width_indices, height_indices = points[..., 0].flatten().clip(0, upsampled_size[0] - 1), points[..., 1].flatten().clip(0, upsampled_size[1] - 1)
    height_indices = (height_indices // (upsampled_size[0] / grid_features.shape[-3])).long()
    width_indices = (width_indices // (upsampled_size[1] / grid_features.shape[-2])).long()
    selected_features = grid_features[batch_indices, timestep_indices, height_indices, width_indices]
    return selected_features

def spacetime_pointwise_interpolation(grid_features, upsampled_size, points, interpolation_method="nearest"):
    num_visible_frames = grid_features.shape[1]
    batch_indices, timestep_indices = get_batch_timestep_indices(points, visible_frame_timesteps=num_visible_frames)
    if interpolation_method == "nearest":
        selected_features = pointwise_nearest_interpolation(grid_features, upsampled_size, points, batch_indices, timestep_indices)
    else:
        raise ValueError(f"Unknown interpolation_method {interpolation_method}")
    selected_features = selected_features.view(points.shape[:-1] + (grid_features.shape[-1], ))
    return selected_features
        

def create_grid_points(height, width, height_num_points, width_num_points):
    """Sample grid points with (time, height, width) order."""
    height_offset = height / height_num_points / 2
    width_offset = width / width_num_points / 2
    
    y = np.linspace(height_offset, height - height_offset, height_num_points).repeat(width_num_points)
    x = np.tile(np.linspace(width_offset, width - width_offset, width_num_points), height_num_points)
    points = torch.Tensor(np.stack((y, x), axis=-1))
    return points


def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(torch.pi / 2 * t) + 1))