import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

from ema_pytorch import EMA
from torchdiffeq import odeint

from transformers import PreTrainedModel
from motion_predictor.prediction_head import DenoisingJointTrackPredictor
from .configuration_query_predictor import QueryPredictorConfig

from motion_predictor.modeling_utils import (
    ImageModelFeatureExtraction,
    DiscreteEmbedder,
    HFTextEncoder,
    OutputPredictor,
    cosmap,
)
@dataclass
class QueryPredictorOutput(ModelOutput):
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
    loss: Optional[torch.FloatTensor] = None
    predictor_output: Optional[torch.FloatTensor] = None
    labels: Optional[torch.FloatTensor] = None
    pooled_text_embedding: Optional[torch.FloatTensor] = None
    image_features: Optional[torch.FloatTensor] = None
    movements: Optional[torch.FloatTensor] = None
    visible_ratios: Optional[torch.FloatTensor] = None

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.weight * (x * Nx) + self.bias + x

class ConvNeXtV2Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class QueryPredictorPretrainedModel(PreTrainedModel):
    config_class = QueryPredictorConfig

class QueryPredictor(QueryPredictorPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.image_feature_extractor = ImageModelFeatureExtraction(
            config.image_model_name,
            freeze_image_model=config.freeze_image_model,
            sam2_image_size=config.encoder_image_size,
        )
        self.upsampler = nn.Upsample(size=(config.height, config.width), mode="bilinear")
        if config.image_encoder_hidden_size != config.conv_block_hidden_size:
            self.image_features_to_conv_block_projection = nn.Linear(config.image_encoder_hidden_size, config.conv_block_hidden_size)
        else:
            self.image_features_to_conv_block_projection = nn.Identity()
        
        self.conv_blocks = nn.Sequential(
            *[ConvNeXtV2Block(dim=config.conv_block_hidden_size, drop_path=config.drop_path_prob) for _ in range(config.num_blocks)]
        )
        if config.text_encoder_name is not None:
            self.text_encoder = HFTextEncoder(
                text_encoder_name=config.text_encoder_name,
                output_size=config.denoising_predictor_config.hidden_size if config.denoising_predictor_config is not None else config.predictor_hidden_size,
                max_seq_length=self.config.text_encoder_max_seq_length,
                pooled_output_size=config.denoising_predictor_config.hidden_size if config.denoising_predictor_config is not None else config.predictor_hidden_size,
                conditioning_drop_prob=config.text_conditioning_drop_prob,
                freeze_text_encoder=config.freeze_text_encoder,
            )
        else:
            self.text_encoder = None
        if config.use_camera_motion_conditioning:
            self.camera_motion_embedder = DiscreteEmbedder(
                num_classes=2,
                hidden_size=config.denoising_predictor_config.hidden_size if config.denoising_predictor_config is not None else config.predictor_hidden_size,
                dropout_prob=config.camera_motion_conditioning_drop_prob
            )
        else:
            self.camera_motion_embedder = None
        # Outputs are movement and visibility ratio
        if config.predictor_type == "rectified_flow":
            self.predictor = DenoisingJointTrackPredictor(
                feature_size=config.conv_block_hidden_size,
                config=config.denoising_predictor_config,
                out_channels=config.out_channels
            )
        else:
            feature_size = config.conv_block_hidden_size
            if config.text_encoder_name is not None or config.use_camera_motion_conditioning:
                feature_size += config.predictor_hidden_size
            self.predictor = OutputPredictor(
                feature_size=feature_size,
                output_size=config.out_channels,
                height=config.height,
                width=config.width,
                predictor_type=config.predictor_type
            )
        self.height = config.height
        self.width = config.width
    
    def construct_features(self, feature_map, query_points):
        # multiple feature maps
        width_indices = query_points[..., 0].round().long().clip(0, self.config.width - 1)
        height_indices = query_points[..., 1].round().long().clip(0, self.config.height - 1)
        
        batch_indices = torch.arange(feature_map.shape[0])[:, None].expand_as(height_indices).flatten()
        
        selected_features = feature_map[batch_indices, height_indices.flatten(), width_indices.flatten()]
        selected_features = selected_features.view(height_indices.shape + (-1, ))
                
        return selected_features

    def forward(
        self,
        query_points: torch.FloatTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        camera_motion: Optional[torch.LongTensor] = None,
        noised: Optional[torch.FloatTensor] = None,
        times: Optional[torch.FloatTensor] = None,
        image_features: Optional[torch.FloatTensor] = None,
        pooled_text_embedding: Optional[torch.FloatTensor] = None,
        text_conditioning_force_drop_indices: Optional[torch.LongTensor] = None,
        camera_motion_conditioning_force_drop_indices: Optional[torch.LongTensor] = None,
    ):
        if image_features is None:
            image_features = [i.permute(0, 3, 1, 2) for i in self.image_feature_extractor(pixel_values)]
            for i in range(len(image_features)):
                image_features[i] = self.upsampler(image_features[i])
            image_features = torch.cat(image_features, dim=1)
            image_features = self.conv_blocks(image_features).permute(0, 2, 3, 1)
        
        if pooled_text_embedding is None and text_input_ids is not None and self.text_encoder is not None:
            _, pooled_text_embedding, text_attention_mask = self.text_encoder(
                text_input_ids,
                text_attention_mask,
                force_drop_indices=text_conditioning_force_drop_indices,
            )
        elif self.text_encoder is None:
            pooled_text_embedding = None
        
        if camera_motion is not None and self.camera_motion_embedder is not None:
            camera_motion_embedding = self.camera_motion_embedder(
                camera_motion,
                force_drop_indices=camera_motion_conditioning_force_drop_indices
            )
        else:
            camera_motion_embedding = None
        
        # Sent to adaLN modulation
        if any(i is not None for i in [pooled_text_embedding, camera_motion_embedding]):
            global_conditioning = sum([i for i in [pooled_text_embedding, camera_motion_embedding] if i is not None]).unsqueeze(1)
        else:
            global_conditioning = None
        
        input_features = self.construct_features(image_features, query_points)
        predictor_output = self.predictor(
            input_features=input_features.unsqueeze(2),
            original_positions=query_points.unsqueeze(2),
            noised=noised,
            times=times,
            attention_mask=attention_mask.unsqueeze(-1),
            global_conditioning=global_conditioning
        )
        return QueryPredictorOutput(
            predictor_output=predictor_output,
            pooled_text_embedding=pooled_text_embedding,
            image_features=image_features,
        )

class QueryPredictorForRegression(QueryPredictorPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.query_predictor = QueryPredictor(config)
        self.use_consistency = False
    
    def sample(
        self, *args, **kwargs
    ):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        query_points: torch.FloatTensor,
        labels: torch.FloatTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        camera_motion: Optional[torch.LongTensor] = None,
        label_mask: Optional[torch.BoolTensor] = None,
    ):        
        query_predictor_outputs = self.query_predictor(
            query_points=query_points,
            pixel_values=pixel_values,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            attention_mask=attention_mask,
            camera_motion=camera_motion,
        )
        predictor_output = query_predictor_outputs.predictor_output.squeeze(-2)
        loss = None
        if labels is not None:
            loss = F.mse_loss(predictor_output[label_mask], labels[label_mask])

        return QueryPredictorOutput(
            loss=loss,
            labels=labels,
            predictor_output=predictor_output,
            movements=(predictor_output[..., 0] * self.config.movement_std + self.config.movement_mean) * self.config.movement_inference_multiplier,
            visible_ratios=predictor_output[..., 1] if self.config.predict_visible_ratios else None,
            image_features=query_predictor_outputs.image_features,
            pooled_text_embedding=query_predictor_outputs.pooled_text_embedding,
        )

class QueryPredictorForRectifiedFlow(QueryPredictorPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.query_predictor = QueryPredictor(config)
        
        if config.noise_schedule_type == "cos":
            self.noise_schedule = cosmap
        
        self.use_consistency = config.rectified_flow_use_consistency
        if self.use_consistency:
            self.consistency_delta_time = config.consistency_delta_time
            self.consistency_velocity_match_alpha = config.consistency_velocity_match_alpha
            self.consistency_loss_weight = config.consistency_loss_weight
            self.ema_model = EMA(
                self.query_predictor,
                beta=config.rectified_flow_ema_beta,
                update_after_step=config.rectified_flow_ema_update_every,
                include_online_model=False,
            )
        # caches
        self.image_features = None
        self.pooled_text_embedding = None

    def get_noised_times_flows(self, labels, noise, times, consistency_prediction=False):
        if self.use_consistency and not consistency_prediction:
            times = times - self.consistency_delta_time
        times = self.noise_schedule(times)
        noised = times * labels + (1 - times) * noise
        flow = labels - noise

        return noised, times, flow

    @torch.no_grad()
    def predict_flow(
        self,
        query_points: torch.FloatTensor,
        noised: torch.FloatTensor,
        times: torch.FloatTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        camera_motion: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        video_conditioning_cfg_scale: Optional[float] = None,
        camera_motion_conditioning_cfg_scale: Optional[float] = None,
        text_conditioning_cfg_scale: Optional[float] = None,
    ):
        self.eval()
        model = self.ema_model if self.use_consistency else self.query_predictor
        # Add camera motion and text later probably
        batch_size = query_points.shape[0]
        drop_all = torch.ones((batch_size, ), device=query_points.device, dtype=torch.long)
        drop_none = torch.zeros((batch_size, ), device=query_points.device, dtype=torch.long)
        # Passing None means conditioning without cfg, passing 0 means drop conditioning
        # "Unconditional" isn't really unconditional but rather conditioning on everything not using cfg (cfg_scale passed as None)
        # This operates under the assumption that conditioning on both is the same as summing an equal weighting of conditioning on each individually
        cfg_scales = {
            "video": video_conditioning_cfg_scale,
            "camera_motion": camera_motion_conditioning_cfg_scale,
            "text": text_conditioning_cfg_scale,
        }
        cfg_scales = {k:v for k,v in cfg_scales.items() if v is not None}
        if any(i is not None for i in cfg_scales.values()):
            input_dict = dict(
                query_points=query_points,
                pixel_values=pixel_values,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                pooled_text_embedding=self.pooled_text_embedding,
                attention_mask=attention_mask,
                image_features=self.image_features,
                camera_motion=camera_motion,
                noised=noised,
                times=times,
            )
            input_dict.update({k + "_conditioning_drop_indices": drop_all for k in cfg_scales})
            unconditional_outputs = model(
                **input_dict,
            )
            unconditional_flow = unconditional_outputs.predictor_output
            guided_flow = unconditional_outputs.predictor_output.clone()
            
            for key, cfg_scale in cfg_scales.items():
                if cfg_scale is not None and cfg_scale != 0:
                    input_dict[key + "_conditioning_drop_indices"] = drop_none
                    key_conditional_outputs = model(**input_dict)
                    input_dict[key + "_conditioning_drop_indices"] = drop_all
            
                    if key == "text":
                        self.pooled_text_embedding = key_conditional_outputs.pooled_text_embedding
                    guided_flow += cfg_scale * (key_conditional_outputs.predictor_output - unconditional_flow)
            return QueryPredictorOutput(
                predictor_output=guided_flow,
            )
        else:
            query_predictor_outputs = model(
                query_points=query_points,
                pixel_values=pixel_values,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                pooled_text_embedding=self.pooled_text_embedding,
                attention_mask=attention_mask,
                image_features=self.image_features,
                camera_motion=camera_motion,
                noised=noised,
                times=times,
            )
            if self.image_features is None:
                self.image_features = query_predictor_outputs.image_features

            self.pooled_text_embedding = query_predictor_outputs.pooled_text_embedding

        return query_predictor_outputs

    @torch.no_grad()
    def sample(
        self,
        query_points: torch.FloatTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        track_rate: Optional[torch.FloatTensor] = None,
        camera_motion: Optional[torch.LongTensor] = None,
        num_steps: Optional[int] = 16,
        solver: Optional[str] = "euler",
        video_conditioning_cfg_scale: Optional[float] = None,
        camera_motion_conditioning_cfg_scale: Optional[float] = None,
        track_rate_conditioning_cfg_scale: Optional[float] = None,
        text_conditioning_cfg_scale: Optional[float] = None,
        apply_noise_schedule: Optional[bool] = False,
    ):
        self.eval()
        self.image_features = None
        self.pooled_text_embedding = None
        
        noise = torch.randn(
            (query_points.shape[0], query_points.shape[1], 1, self.config.out_channels),
            device=query_points.device,
            dtype=query_points.dtype
        )
        times = torch.linspace(0, 1, num_steps, device=query_points.device)
        if apply_noise_schedule:
            times = self.noise_schedule(times)
        
        def ode_fn(t, x):
            outputs = self.predict_flow(
                query_points=query_points,
                noised=x,
                times=torch.Tensor([t]).to(query_points.device)[:, None, None, None],
                pixel_values=pixel_values,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                camera_motion=camera_motion,
                attention_mask=attention_mask,
                video_conditioning_cfg_scale=video_conditioning_cfg_scale,
                camera_motion_conditioning_cfg_scale=camera_motion_conditioning_cfg_scale,
                text_conditioning_cfg_scale=text_conditioning_cfg_scale,
            )
            return outputs.predictor_output
        
        trajectory = odeint(ode_fn, noise, times, method=solver)
        predicted = trajectory[-1]
        return QueryPredictorOutput(
            movements=(predicted[..., 0, 0] * self.config.movement_std + self.config.movement_mean) * self.config.movement_inference_multiplier,
            visible_ratios=predicted[..., 0, 1] if self.config.predict_visible_ratios else None,
        )
    def forward(
        self,
        query_points: torch.FloatTensor,
        labels: torch.FloatTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        camera_motion: Optional[torch.LongTensor] = None,
        label_mask: Optional[torch.BoolTensor] = None,
    ):
        labels = labels.unsqueeze(-2)
        noise = torch.randn_like(labels)
        times = torch.rand(labels.shape[0], device=labels.device)[:, None, None, None]
        noised, times, flow = self.get_noised_times_flows(labels, noise, times)
        
        query_predictor_outputs = self.query_predictor(
            query_points=query_points,
            pixel_values=pixel_values,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            attention_mask=attention_mask,
            camera_motion=camera_motion,
            noised=noised,
            times=times,
            image_features=self.image_features,
            pooled_text_embedding=self.pooled_text_embedding,
        )

        flow_prediction = query_predictor_outputs.predictor_output
        data_prediction = noised - flow_prediction * (1 - times)

        if self.use_consistency and labels is not None:
            ema_noised, ema_times, ema_flow = self.get_noised_times_flows(labels, noise, times, consistency_prediction=True)
            ema_query_predictor_outputs = self.ema_model(
                query_points=query_points,
                pixel_values=pixel_values,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                attention_mask=attention_mask,
                camera_motion=camera_motion,
                noised=noised,
                times=times,
                image_features=self.image_features,
                pooled_text_embedding=self.pooled_text_embedding,
            )
            ema_flow_prediction = ema_query_predictor_outputs.predictor_output
            ema_data_prediction = ema_noised - ema_flow_prediction * (1 - ema_times)

        loss = None
        if labels is not None:
            loss = F.mse_loss(flow_prediction[label_mask], flow[label_mask])

            if self.use_consistency:
                data_match_loss = F.mse_loss(data_prediction[label_mask], ema_data_prediction[label_mask])
                velocity_match_loss = F.mse_loss(flow_prediction[label_mask], ema_flow_prediction[label_mask])
                consistency_loss = data_match_loss + velocity_match_loss * self.consistency_velocity_match_alpha
                loss = loss + consistency_loss * self.consistency_loss_weight

        return QueryPredictorOutput(
            loss=loss,
            labels=labels,
            predictor_output=flow_prediction,
            image_features=query_predictor_outputs.image_features,
            pooled_text_embedding=query_predictor_outputs.pooled_text_embedding,
        )