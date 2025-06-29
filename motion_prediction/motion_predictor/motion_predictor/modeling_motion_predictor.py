from torch import nn

import torch
import torch.nn.functional as F
from .configuration_motion_predictor import MotionPredictorConfig

from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

from ema_pytorch import EMA
from torchdiffeq import odeint

from transformers import AutoModel, PreTrainedModel
from motion_predictor.spacetime_transformer import JointTrackModel
from motion_predictor.prediction_head import DenoisingJointTrackPredictor
from motion_predictor.modeling_utils import (
    ImageAggregationCausalVideoModel,
    DiscreteEmbedder,
    SinusoidalEmbedder,
    HFTextEncoder,
    cosmap,
    ShiftScaleFourierPredictor,
    ShiftFourierPredictor,
    ShiftFourierPredictor,
    spacetime_pointwise_interpolation,
    create_grid_points,
    get_batch_timestep_indices
)

@dataclass
class MotionPredictorOutput(ModelOutput):
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
    track_predictor_hidden_state: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    predictor_output: Optional[torch.FloatTensor] = None
    labels: Optional[torch.FloatTensor] = None
    video_encoder_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    track_predictor_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    video_feature_map: Optional[torch.FloatTensor] = None
    text_embedding: Optional[torch.FloatTensor] = None
    pooled_text_embedding: Optional[torch.FloatTensor] = None

class MotionPredictorPretrainedModel(PreTrainedModel):
    config_class = MotionPredictorConfig

class MotionPredictor(MotionPredictorPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.video_encoder = ImageAggregationCausalVideoModel(
            config.image_model_name,
            freeze_image_model=config.freeze_image_model,
            time_dimension_model_config=config.encoder_config,
            encoder_pooler_type=config.encoder_pooler_type,
            sam2_image_size=config.sam2_image_size,
            image_model_batch_size=config.image_model_batch_size
        )
        if config.global_image_model_name is not None and config.global_image_model_name != config.image_model_name:
            self.global_video_encoder = ImageAggregationCausalVideoModel(
                config.global_image_model_name,
                freeze_image_model=config.freeze_global_image_model,
                image_model_batch_size=config.image_model_batch_size,
            )
        else:
            self.global_video_encoder = None
        
        if config.track_predictor_head_type == "shift_scale":
            self.track_predictor_head = ShiftScaleFourierPredictor(
                feature_size=self.config.track_predictor_config.hidden_size,
                height=config.height,
                width=config.width,
                max_height_shift=config.max_height_shift,
                max_width_shift=config.max_width_shift,
                **config.track_predictor_head_kwargs
            )
        elif config.track_predictor_head_type == "shift":
            self.track_predictor_head = ShiftFourierPredictor(
                feature_size=self.config.track_predictor_config.hidden_size,
                height=config.height,
                width=config.width,
                **config.track_predictor_head_kwargs,
            )
        elif config.track_predictor_head_type == "denoising_predictor":
            self.track_predictor_head = DenoisingJointTrackPredictor(
                feature_size=self.config.track_predictor_config.hidden_size,
                config=config.denoising_predictor_config,
                concatenate_original_position=config.use_absolute_positional_embeddings,
                use_previous_relative_shift_input=config.use_previous_relative_shift_input,
            )
        else:
            raise ValueError(f"Unknown token predictor {config.track_predictor_head_type}")
        if config.text_encoder_name is not None:
            self.text_encoder = HFTextEncoder(
                text_encoder_name=config.text_encoder_name,
                output_size=config.track_predictor_config.hidden_size,
                max_seq_length=self.config.text_encoder_max_seq_length,
                pooled_output_size=config.denoising_predictor_config.hidden_size,
                conditioning_drop_prob=config.text_conditioning_drop_prob,
                freeze_text_encoder=config.freeze_text_encoder,
            )
        else:
            self.text_encoder = None
        if config.use_camera_motion_conditioning:
            self.camera_motion_embedder = DiscreteEmbedder(
                num_classes=2,
                hidden_size=config.denoising_predictor_config.hidden_size,
                dropout_prob=config.camera_motion_conditioning_drop_prob
            )
        else:
            self.camera_motion_embedder = None
        if config.use_track_rate_conditioning:
            self.track_rate_embedder = SinusoidalEmbedder(
                config.denoising_predictor_config.hidden_size,
                dropout_prob=config.track_rate_conditioning_drop_prob
            )
        else:
            self.track_rate_embedder = None
        
        self.track_predictor = AutoModel.from_config(config.track_predictor_config)
        self.position_encoding_norm_to_feature_norm_ratio = config.position_encoding_norm_to_feature_norm_ratio
        
        self.feature_map_upsampler = nn.Upsample(size=(self.config.height, self.config.width), mode=self.config.feature_map_upsample_method)
        
        input_feature_shape = self.track_predictor_head.embedding_size if config.use_absolute_positional_embeddings else 0
        if self.config.video_encoder_hidden_size is not None:
            video_feature_shape = self.video_encoder.image_model.image_model_hidden_size
        elif self.config.encoder_config is not None:
            video_feature_shape = self.config.encoder_config.hidden_size
        else:
            raise ValueError("Could not find video feature hidden_size, encoder_config and video_encoder_hidden_size not set")
        
        input_feature_shape += video_feature_shape
        if not self.config.late_noise_conditioning:
            input_feature_shape += self.config.extra_features_size
            
        if input_feature_shape != self.config.track_predictor_config.hidden_size:
            self.feature_to_predictor_projection = nn.Linear(input_feature_shape, self.config.track_predictor_config.hidden_size)
        else:
            self.feature_to_predictor_projection = nn.Identity()
        if self.global_video_encoder is not None:
            global_video_feature_shape = self.global_video_encoder.image_model_hidden_size
            global_input_feature_shape = input_feature_shape - video_feature_shape + global_video_feature_shape
            self.global_feature_to_predictor_projection = nn.Linear(global_input_feature_shape, self.config.track_predictor_config.hidden_size)
        self.late_noise_conditioning = config.late_noise_conditioning

        if self.config.encoder_pooler_type == "none":
            self.visibility_indicator_embeddings = nn.Embedding(2, video_feature_shape)
            if self.global_video_encoder is not None:
                self.global_visibility_indicator_embeddings = nn.Embedding(2, global_video_feature_shape)
        else:
            self.visibility_indicator_embeddings = None
        if config.video_conditioning_drop_prob > 0:
            self.video_conditioning_dropped_embedding = nn.Parameter(torch.randn((video_feature_shape, )))
            if self.global_video_encoder is not None:
                self.global_video_conditioning_dropped_embedding = nn.Parameter(torch.randn((global_video_feature_shape, )))
        
    # Add support for different upsampling types
    def select_video_features(self, feature_map, input_ids):
        # space time indexing
        if self.config.encoder_pooler_type == "none":
            return spacetime_pointwise_interpolation(
                grid_features=feature_map,
                upsampled_size=(self.config.height, self.config.width),
                points=input_ids[..., :2],
                interpolation_method=self.config.feature_map_upsample_method
            )
        else:
            if self.config.feature_map_upsample_method == "bilinear":
                width_indices = input_ids[..., 0].round().long().clip(0, self.config.width - 1)
                height_indices = input_ids[..., 1].round().long().clip(0, self.config.height - 1)
                
                upsampled_feature_map = self.feature_map_upsampler(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                height_indices = height_indices.clip(0, upsampled_feature_map.shape[1] - 1)
                width_indices = width_indices.clip(0, upsampled_feature_map.shape[2] - 1)

                batch_indices = torch.arange(upsampled_feature_map.shape[0])[:, None, None].expand_as(height_indices).flatten()
                selected_video_features = upsampled_feature_map[batch_indices, height_indices.flatten(), width_indices.flatten()]
                selected_video_features = selected_video_features.view(height_indices.shape + (-1, ))
                return selected_video_features
            else:
                raise ValueError(f"Unknown upsampling method {self.config.feature_map_upsample_method}")

    # Refactor to make the video model take care of this logic?
    def construct_features(self, feature_map, input_ids, noised, times, global_feature_map=None, video_conditioning_force_drop_indices=None):
        
        position_embeddings = self.track_predictor_head.fourier_embeddings.position_to_embedding(input_ids)
        if not self.config.use_absolute_positional_embeddings:
            position_embeddings = position_embeddings[..., :0]
        # multiple feature maps
        if isinstance(feature_map, list):
            selected_video_features = [self.select_video_features(fm, input_ids) for fm in feature_map]
            selected_video_features = torch.cat(selected_video_features, dim=-1)
        else:
            selected_video_features = self.select_video_features(feature_map, input_ids)
        
        if (self.config.video_conditioning_drop_prob > 0 and self.training) or video_conditioning_force_drop_indices is not None:
            if video_conditioning_force_drop_indices is None:
                video_conditioning_force_drop_indices = (
                    torch.rand(selected_video_features.shape[0], device=selected_video_features.device) < self.config.video_conditioning_drop_prob
                )
            else:
                video_conditioning_force_drop_indices = video_conditioning_force_drop_indices == 1
            if global_feature_map is not None:
                global_feature_map[video_conditioning_force_drop_indices] = self.global_video_conditioning_dropped_embedding

        if self.visibility_indicator_embeddings is not None:
            batch_indices, timestep_indices = get_batch_timestep_indices(input_ids)
            if isinstance(feature_map, list):
                num_visible_frames = feature_map[0].shape[1]
            else:
                num_visible_frames = feature_map.shape[1]

            visibility = (timestep_indices < min(num_visible_frames, input_ids.shape[-2])).long()
            if video_conditioning_force_drop_indices is not None:
                for batch_idx in range(len(video_conditioning_force_drop_indices)):
                    if video_conditioning_force_drop_indices[batch_idx] == 1:
                        visibility[batch_indices == batch_idx] = 0
            selected_visibility_indicator_embeddings = self.visibility_indicator_embeddings(visibility)
            selected_visibility_indicator_embeddings = selected_visibility_indicator_embeddings.view(input_ids.shape[:-1] + (-1, ))
            selected_video_features = selected_video_features + selected_visibility_indicator_embeddings
            
        if noised is not None and times is not None and not self.late_noise_conditioning:
            input_features = torch.cat((noised, times, position_embeddings, selected_video_features), dim=-1)
        else:
            input_features = torch.cat((position_embeddings, selected_video_features), dim=-1)
        
        input_features = self.feature_to_predictor_projection(input_features)
        if global_feature_map is not None:
            global_feature_map_positions = create_grid_points(self.config.height, self.config.width, global_feature_map.shape[2], global_feature_map.shape[3])
            global_feature_map_position_embeddings = self.track_predictor_head.fourier_embeddings.position_to_embedding(global_feature_map_positions)

            visible_visibility_embeddings = self.global_visibility_indicator_embeddings(torch.tensor(1, device=global_feature_map.device))
            non_visibile_visibility_embeddings = self.global_visibility_indicator_embeddings(torch.tensor(0, device=global_feature_map.device))
            global_feature_map = global_feature_map.flatten(2, 3)
            global_feature_map = global_feature_map + visible_visibility_embeddings

            if input_ids.shape[2] > global_feature_map.shape[1]:
                non_visible_global_feature_map = global_feature_map[:, -1:].repeat_interleave(input_ids.shape[2] - global_feature_map.shape[1], dim=1)
                non_visible_global_feature_map = non_visible_global_feature_map + non_visibile_visibility_embeddings
                global_feature_map = torch.cat((global_feature_map, non_visible_global_feature_map), dim=1)
            
            global_feature_map_position_embeddings = global_feature_map_position_embeddings[None, None, ...].expand(global_feature_map.shape[0], global_feature_map.shape[1], -1, -1)
            if not self.config.use_absolute_positional_embeddings:
                global_feature_map_position_embeddings = global_feature_map_position_embeddings[..., :0]

            global_feature_map = torch.cat((global_feature_map, global_feature_map_position_embeddings.to(global_feature_map.device)), dim=-1)
            
            global_feature_map = self.global_feature_to_predictor_projection(global_feature_map)
            global_input_features = global_feature_map.transpose(1, 2)
        else:
            global_input_features = None

        return input_features, global_input_features

    def forward(
        self,
        input_ids,
        pixel_values: Optional[torch.FloatTensor] = None,
        global_pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
        video_encoder_past_key_values: Optional[torch.FloatTensor] = None,
        track_predictor_past_key_values: Optional[torch.FloatTensor] = None,
        track_predictor_cache_position: Optional[torch.LongTensor] = None,
        track_predictor_position_ids: Optional[torch.LongTensor] = None,
        track_predictor_hidden_state: Optional[torch.FloatTensor] = None,
        video_feature_map: Optional[torch.FloatTensor] = None,
        global_video_feature_map: Optional[torch.FloatTensor] = None,
        text_embedding: Optional[torch.FloatTensor] = None,
        pooled_text_embedding: Optional[torch.FloatTensor] = None,
        camera_motion: Optional[torch.LongTensor] = None,
        track_rate: Optional[torch.FloatTensor] = None,
        noised: Optional[torch.FloatTensor] = None,
        times: Optional[torch.FloatTensor] = None,
        use_kv_cache: Optional[bool] = False,
        video_conditioning_force_drop_indices: Optional[torch.LongTensor] = None,
        camera_motion_conditioning_force_drop_indices: Optional[torch.LongTensor] = None,
        text_conditioning_force_drop_indices: Optional[torch.LongTensor] = None,
        track_rate_conditioning_force_drop_indices: Optional[torch.LongTensor] = None,
    ):
        video_encoder_outputs = None
        if video_feature_map is None:
            video_encoder_outputs = self.video_encoder(pixel_values)
            video_feature_map = video_encoder_outputs.video_feature_map
        if global_video_feature_map is None and self.global_video_encoder is not None:
            global_video_encoder_outputs = self.global_video_encoder(global_pixel_values)
            global_video_feature_map = global_video_encoder_outputs.video_feature_map

        if (text_embedding is None or pooled_text_embedding is None) and text_input_ids is not None and self.text_encoder is not None:
            text_embedding, pooled_text_embedding, text_attention_mask = self.text_encoder(
                text_input_ids,
                text_attention_mask,
                force_drop_indices=text_conditioning_force_drop_indices,
            )
        elif self.text_encoder is None:
            text_embedding = None
            pooled_text_embedding = None
        if camera_motion is not None and self.camera_motion_embedder is not None:
            camera_motion_embedding = self.camera_motion_embedder(
                camera_motion,
                force_drop_indices=camera_motion_conditioning_force_drop_indices
            )
        else:
            camera_motion_embedding = None
        if track_rate is not None and self.track_rate_embedder is not None:
            track_rate_embedding = self.track_rate_embedder(
                track_rate / self.config.track_rate_conditioning_max_track_rate,
            )
        else:
            track_rate_embedding = None
        
        # Sent to adaLN modulation
        if any(i is not None for i in [pooled_text_embedding, camera_motion_embedding, track_rate_embedding]):
            global_conditioning = sum([i for i in [pooled_text_embedding, camera_motion_embedding, track_rate_embedding] if i is not None]).unsqueeze(1)
        else:
            global_conditioning = 0
        track_predictor_outputs = None
        
        global_input_feature_length = global_video_feature_map.shape[2] * global_video_feature_map.shape[3] if global_video_feature_map is not None else 0
        attention_mask_text_embedding_length = text_embedding.shape[1] if text_embedding is not None else 0
        hidden_state_text_embedding_length = text_embedding.shape[1] if text_embedding is not None else 0
        if track_predictor_hidden_state is None:
            input_features, global_input_features = self.construct_features(
                feature_map=video_feature_map,
                global_feature_map=global_video_feature_map,
                input_ids=input_ids,
                noised=noised,
                times=times,
                video_conditioning_force_drop_indices=video_conditioning_force_drop_indices
            )
            if global_input_features is not None:
                input_features = torch.cat((global_input_features, input_features), dim=1)
                #print(attention_mask.shape, global_input_features.shape, input_features.shape)
                attention_mask = torch.cat((torch.ones((attention_mask.shape[0], global_input_features.shape[1], input_features.shape[2]), device=attention_mask.device), attention_mask), dim=1)
            
            if text_embedding is not None:
                expanded_text_embedding = text_embedding[:, None, :, :].expand(-1, input_features.shape[1], -1, -1)
                # Checks if the kv_cache already has the text embedding in which case we don't need to concatenate as it will be in the kv cache
                if not use_kv_cache or track_predictor_past_key_values is None or (track_predictor_past_key_values is not None and track_predictor_past_key_values.get_seq_length()) == 0:
                    input_features = torch.cat((expanded_text_embedding, input_features), dim=2)
                else:
                    hidden_state_text_embedding_length = 0
                    
                # concatenated text embedding is not pooled
                if text_embedding.shape[1] > 1:
                    expanded_text_attention_mask = text_attention_mask[:, None, :].expand(-1, input_features.shape[1], -1).clone()
                    # We don't want to attend at all to spatial positions where the track is missing
                    if attention_mask is not None:
                        expanded_text_attention_mask[attention_mask.sum(dim=2) == 0] = 0
                    # Used during sampling to ensure we have an attention mask for the past encoded tracks after the text embedding values
                    if track_predictor_past_key_values is not None and track_predictor_past_key_values.get_seq_length() > attention_mask_text_embedding_length:
                        track_attention_mask_padding = attention_mask[:, :, -1:].clone().expand(-1, -1, track_predictor_past_key_values.get_seq_length() - attention_mask_text_embedding_length)
                        attention_mask = torch.cat([expanded_text_attention_mask, track_attention_mask_padding, attention_mask], dim=2)
                    else:
                        attention_mask = torch.cat([expanded_text_attention_mask, attention_mask], dim=2)
                else:
                    attention_mask = torch.cat([attention_mask[:, :, :1].clone(), attention_mask], dim=2)
            if not self.config.track_predictor_has_spatial_condition:
                input_features = input_features.flatten(0, 1)
                if attention_mask is not None:
                    attention_mask = attention_mask.flatten(0, 1)
            # should be renamed to feature fusion module or something
            track_predictor_outputs = self.track_predictor(
                inputs_embeds=input_features,
                attention_mask=attention_mask,
                past_key_values=track_predictor_past_key_values,
                use_cache=use_kv_cache,
                cache_position=track_predictor_cache_position,
                position_ids=track_predictor_position_ids,
            )
            track_predictor_hidden_state = track_predictor_outputs.last_hidden_state
            # if we pass a track predictor hidden state cache then we don't need to index the attention mask otherwise we do
            attention_mask = attention_mask[:, global_input_feature_length:, attention_mask_text_embedding_length:]
        else:
            hidden_state_text_embedding_length = 0

        predictor_output = self.track_predictor_head(
            input_features=track_predictor_hidden_state[:, global_input_feature_length:, hidden_state_text_embedding_length:],
            original_positions=input_ids,
            noised=noised,
            times=times,
            attention_mask=attention_mask,
            global_conditioning=global_conditioning
        )

        return MotionPredictorOutput(
            track_predictor_hidden_state=track_predictor_hidden_state,
            predictor_output=predictor_output,
            track_predictor_past_key_values=track_predictor_outputs.past_key_values if track_predictor_outputs is not None else track_predictor_past_key_values,
            video_encoder_past_key_values=video_encoder_outputs.past_key_values if video_encoder_outputs is not None else video_encoder_past_key_values,
            hidden_states=track_predictor_outputs.hidden_states if track_predictor_outputs is not None else None,
            attentions=track_predictor_outputs.attentions if track_predictor_outputs is not None else None,
            video_feature_map=video_feature_map,
            text_embedding=text_embedding,
            pooled_text_embedding=pooled_text_embedding,
        )

TIME_DEBUG = False

class MotionPredictorForRectifiedFlow(MotionPredictorPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.motion_predictor = MotionPredictor(config)
        if config.noise_schedule_type == "cos":
            self.noise_schedule = cosmap
        elif config.noise_schedule_type == "left_weighted":
            self.noise_schedule = left_weighted
        elif config.noise_schedule_type == "beta":
            self.noise_schedule = beta_distribution
        
        self.use_consistency = config.rectified_flow_use_consistency
        self.relative_prediction = config.rectified_flow_relative_prediction
        if self.use_consistency:
            self.consistency_delta_time = config.consistency_delta_time
            self.consistency_velocity_match_alpha = config.consistency_velocity_match_alpha
            self.consistency_loss_weight = config.consistency_loss_weight
            self.ema_model = EMA(
                self.motion_predictor,
                beta=config.rectified_flow_ema_beta,
                update_after_step=config.rectified_flow_ema_update_every,
                include_online_model=False,
            )
        # caches
        self.video_feature_map = None
        self.track_predictor_past_key_values = None
        self.track_predictor_hidden_state = None
        self.text_embedding = None
        self.pooled_text_embedding = None
        self.encoded_text = False
    
    def reset_ema_model(self):
        self.ema_model = EMA(
            self.motion_predictor,
            beta=self.config.rectified_flow_ema_beta,
            update_after_step=self.config.rectified_flow_ema_update_every,
            include_online_model=False,
        )

    def reset_caches(self):
        self.video_feature_map = None
        self.track_predictor_past_key_values = None
        self.track_predictor_hidden_state = None
        self.text_embedding = None
        self.pooled_text_embedding = None
        self.encoded_text = False
    
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
        input_ids: torch.FloatTensor,
        noised: torch.FloatTensor,
        times: torch.FloatTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        global_pixel_values: Optional[torch.FloatTensor] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        camera_motion: Optional[torch.LongTensor] = None,
        track_rate: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        use_kv_cache: Optional[bool] = False,
        track_predictor_cache_position: Optional[torch.LongTensor] = None,
        track_predictor_position_ids: Optional[torch.LongTensor] = None,
        video_conditioning_cfg_scale: Optional[float] = None,
        camera_motion_conditioning_cfg_scale: Optional[float] = None,
        text_conditioning_cfg_scale: Optional[float] = None,
        track_rate_conditioning_cfg_scale: Optional[float] = None,
    ):
        self.eval()
        model = self.ema_model if self.use_consistency else self.motion_predictor
        # Add camera motion and text later probably
        batch_size = input_ids.shape[0]
        drop_all = torch.ones((batch_size, ), device=input_ids.device, dtype=torch.long)
        drop_none = torch.zeros((batch_size, ), device=input_ids.device, dtype=torch.long)
        # Passing None means conditioning without cfg, passing 0 means drop conditioning
        # "Unconditional" isn't really unconditional but rather conditioning on everything not using cfg (cfg_scale passed as None)
        # This operates under the assumption that conditioning on both is the same as summing an equal weighting of conditioning on each individually
        cfg_scales = {
            "video": video_conditioning_cfg_scale,
            "camera_motion": camera_motion_conditioning_cfg_scale,
            "text": text_conditioning_cfg_scale,
            "track_rate": track_rate_conditioning_cfg_scale,
        }
        cfg_scales = {k:v for k,v in cfg_scales.items() if v is not None}
        if any(i is not None for i in cfg_scales.values()):
            if type(self.track_predictor_hidden_state) is not dict:
                self.track_predictor_hidden_state = {k:self.track_predictor_hidden_state for k in list(cfg_scales.keys()) + ["unconditional",]}
            if type(self.track_predictor_past_key_values) is not dict:
                self.track_predictor_past_key_values = {k:self.track_predictor_past_key_values for k in list(cfg_scales.keys()) + ["unconditional",]}

            input_dict = dict(
                input_ids=input_ids,
                pixel_values=pixel_values,
                global_pixel_values=global_pixel_values,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                text_embedding=self.text_embedding,
                pooled_text_embedding=self.pooled_text_embedding,
                attention_mask=attention_mask,
                video_feature_map=self.video_feature_map,
                use_kv_cache=use_kv_cache,
                camera_motion=camera_motion,
                track_rate=track_rate,
                noised=noised,
                times=times,
                track_predictor_position_ids=track_predictor_position_ids,
                track_predictor_cache_position=track_predictor_cache_position,
            )
            input_dict.update({k + "_conditioning_force_drop_indices": drop_all for k in cfg_scales})
            input_dict.update(dict(
                track_predictor_hidden_state=self.track_predictor_hidden_state.get("unconditional"),
                track_predictor_past_key_values=self.track_predictor_past_key_values.get("unconditional"),
            ))
            unconditional_outputs = model(
                **input_dict,
            )
            unconditional_flow = unconditional_outputs.predictor_output
            guided_flow = unconditional_outputs.predictor_output.clone()
            
            self.track_predictor_hidden_state["unconditional"] = unconditional_outputs.track_predictor_hidden_state
            if unconditional_outputs.track_predictor_past_key_values is not None and use_kv_cache:
                self.track_predictor_past_key_values["unconditional"] = unconditional_outputs.track_predictor_past_key_values
            for key, cfg_scale in cfg_scales.items():
                if cfg_scale is not None and cfg_scale != 0:
                    input_dict[key + "_conditioning_force_drop_indices"] = drop_none
                    input_dict.update(dict(
                        track_predictor_hidden_state=self.track_predictor_hidden_state.get(key),
                        track_predictor_past_key_values=self.track_predictor_past_key_values.get(key),
                    ))
                    key_conditional_outputs = model(**input_dict)
                    self.track_predictor_hidden_state[key] = key_conditional_outputs.track_predictor_hidden_state
                    if key_conditional_outputs.track_predictor_past_key_values is not None and use_kv_cache:
                        self.track_predictor_past_key_values[key] = key_conditional_outputs.track_predictor_past_key_values
                    
                    input_dict[key + "_conditioning_force_drop_indices"] = drop_all
                    
                    if key == "text":
                        self.text_embedding = key_conditional_outputs.text_embedding
                        self.pooled_text_embedding = key_conditional_outputs.pooled_text_embedding
                    guided_flow += cfg_scale * (key_conditional_outputs.predictor_output - unconditional_flow)
            return MotionPredictorOutput(
                predictor_output=guided_flow,
            )
        else:
            motion_predictor_outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                global_pixel_values=global_pixel_values,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                text_embedding=self.text_embedding,
                pooled_text_embedding=self.pooled_text_embedding,
                attention_mask=attention_mask,
                track_predictor_past_key_values=self.track_predictor_past_key_values,
                track_predictor_hidden_state=self.track_predictor_hidden_state,
                video_feature_map=self.video_feature_map,
                use_kv_cache=use_kv_cache,
                camera_motion=camera_motion,
                track_rate=track_rate,
                noised=noised,
                times=times,
                track_predictor_position_ids=track_predictor_position_ids,
                track_predictor_cache_position=track_predictor_cache_position,
            )
            if self.video_feature_map is None:
                self.video_feature_map = motion_predictor_outputs.video_feature_map
            if motion_predictor_outputs.track_predictor_past_key_values is not None and use_kv_cache:
                self.track_predictor_past_key_values = motion_predictor_outputs.track_predictor_past_key_values
            self.track_predictor_hidden_state = motion_predictor_outputs.track_predictor_hidden_state
            self.text_embedding = motion_predictor_outputs.text_embedding
            self.pooled_text_embedding = motion_predictor_outputs.pooled_text_embedding
        return motion_predictor_outputs

    @torch.no_grad()
    def sample_forward_pass(
        self,
        input_ids: torch.FloatTensor,
        past_noised: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        global_pixel_values: Optional[torch.FloatTensor] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        camera_motion: Optional[torch.LongTensor] = None,
        track_rate: Optional[torch.FloatTensor] = None,
        update_noising_start_timestep: Optional[float] = 0,
        num_steps: Optional[int] = 8,
        use_kv_cache: Optional[bool] = False,
        solver: Optional[str] = "euler",
        num_timesteps_to_sample: Optional[int] = 1,
        video_conditioning_cfg_scale: Optional[torch.LongTensor] = None,
        camera_motion_conditioning_cfg_scale: Optional[float] = None,
        text_conditioning_cfg_scale: Optional[float] = None,
        track_rate_conditioning_cfg_scale: Optional[float] = None,
        apply_noise_schedule: Optional[bool] = False,
        track_predictor_cache_position: Optional[int] = None,
        noised: Optional[torch.FloatTensor] = None,
        track_predictor_position_ids: Optional[torch.LongTensor] = None
    ):
        self.eval()
        self.track_predictor_hidden_state = None
        
        if noised is None:
            noised = torch.randn(
                (input_ids.shape[0], input_ids.shape[1], num_timesteps_to_sample, self.config.track_dimensionality),
                device=input_ids.device,
                dtype=input_ids.dtype
            )
        times = torch.linspace(update_noising_start_timestep, 1, num_steps + 1, device=input_ids.device)[:-1]
        
        if apply_noise_schedule:
            times = self.noise_schedule(times)
        
        def ode_fn(t, x):
            if past_noised is not None:
                full_noised = torch.cat((past_noised[0], x), dim=2)
            else:
                full_noised = x
            outputs = self.predict_flow(
                input_ids=input_ids,
                noised=full_noised,
                times=torch.Tensor([t]).to(input_ids.device)[:, None, None, None],
                pixel_values=pixel_values,
                global_pixel_values=global_pixel_values,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                camera_motion=camera_motion,
                track_rate=track_rate,
                attention_mask=attention_mask,
                use_kv_cache=use_kv_cache,
                track_predictor_position_ids=track_predictor_position_ids,
                track_predictor_cache_position=track_predictor_cache_position,
                video_conditioning_cfg_scale=video_conditioning_cfg_scale,
                camera_motion_conditioning_cfg_scale=camera_motion_conditioning_cfg_scale,
                text_conditioning_cfg_scale=text_conditioning_cfg_scale,
                track_rate_conditioning_cfg_scale=track_rate_conditioning_cfg_scale,
            )
            if past_noised is not None:
                return outputs.predictor_output[:, :, -num_timesteps_to_sample:]
            else:
                return outputs.predictor_output
        
        trajectory = odeint(ode_fn, noised, times, method=solver)
        return trajectory
    
    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.FloatTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        global_pixel_values: Optional[torch.FloatTensor] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        track_predictor_past_key_values: Optional[torch.FloatTensor] = None,
        video_feature_map: Optional[torch.FloatTensor] = None,
        text_embedding: Optional[torch.FloatTensor] = None,
        pooled_text_embedding: Optional[torch.FloatTensor] = None,
        camera_motion: Optional[torch.LongTensor] = None,
        track_rate: Optional[torch.FloatTensor] = None,
        max_track_length: Optional[int] = None,
        num_timesteps_to_sample: Optional[int] = None,
        num_steps: Optional[int] = 16,
        start_timestep: Optional[int] = None,
        update_noising_start_timestep: Optional[float] = 0,
        update_num_steps: Optional[int] = 16,
        update_timestep_size: Optional[int] = 1,
        use_kv_cache: Optional[bool] = False,
        solver: Optional[str] = "euler",
        video_conditioning_cfg_scale: Optional[float] = None,
        camera_motion_conditioning_cfg_scale: Optional[float] = None,
        text_conditioning_cfg_scale: Optional[float] = None,
        track_rate_conditioning_cfg_scale: Optional[float] = None,
        apply_noise_schedule: Optional[bool] = False,
        kv_cache_update_size: Optional[int] = 0,
    ):
        self.eval()
        # reset video_feature_map, track_predictor_past_key_values
        self.video_feature_map = video_feature_map
        self.track_predictor_past_key_values = track_predictor_past_key_values
        self.text_embedding = text_embedding
        self.pooled_text_embedding = pooled_text_embedding
        self.encoded_text = False

        if max_track_length is None:
            if num_timesteps_to_sample is not None:
                max_track_length = num_timesteps_to_sample + input_ids.shape[2] - 1
            else:
                max_track_length = self.config.max_track_length
        
        if start_timestep is None:
            start_timestep = input_ids.shape[2]
        new_input_ids = torch.empty((input_ids.shape[0], input_ids.shape[1], max_track_length + 1, input_ids.shape[3]), device=input_ids.device, dtype=input_ids.dtype)
        new_input_ids[:, :, :input_ids.shape[2]] = input_ids
        # Past noised of time step 0 is for predicting 1
        if not self.config.late_noise_conditioning:
            past_noised = torch.empty((num_steps, input_ids.shape[0], input_ids.shape[1], max_track_length, input_ids.shape[3]), device=input_ids.device, dtype=input_ids.dtype)
        
        # Additional conditioning beyond start track point
        if input_ids.shape[2] > 1 and not self.config.late_noise_conditioning:
            step_trajectory = self.sample_forward_pass(
                input_ids=new_input_ids[..., :input_ids.shape[2] - 1, :],
                past_noised=past_noised[..., :0, :],
                pixel_values=pixel_values,
                global_pixel_values=global_pixel_values,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                attention_mask=attention_mask[:, :, :input_ids.shape[2] - 1] if attention_mask is not None else None,
                num_steps=num_steps,
                num_timesteps_to_sample=input_ids.shape[2] - 1,
                use_kv_cache=use_kv_cache,
                camera_motion=camera_motion,
                track_rate=track_rate,
                solver=solver,
                video_conditioning_cfg_scale=video_conditioning_cfg_scale,
                camera_motion_conditioning_cfg_scale=camera_motion_conditioning_cfg_scale,
                text_conditioning_cfg_scale=text_conditioning_cfg_scale,
                track_rate_conditioning_cfg_scale=track_rate_conditioning_cfg_scale,
                apply_noise_schedule=apply_noise_schedule
            )
            past_noised[:, :, :, :input_ids.shape[2] - 1] = step_trajectory
        
        if kv_cache_update_size > 0:
            for timestep in range(start_timestep - kv_cache_update_size, start_timestep):
                _ = self.predict_flow(
                    input_ids=new_input_ids[..., timestep - 1 if use_kv_cache else 0:timestep, :],
                    past_noised=past_noised[..., timestep - 2 if use_kv_cache else 0:timestep - 1, :] if not self.config.late_noise_conditioning else None,
                    pixel_values=pixel_values,
                    global_pixel_values=global_pixel_values,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    camera_motion=camera_motion,
                    track_rate=track_rate,
                    attention_mask=attention_mask[:, :, timestep - 1 if use_kv_cache else 0:timestep] if attention_mask is not None else None,
                    num_steps=update_num_steps,
                    use_kv_cache=use_kv_cache,
                    update_noising_start_timestep=update_noising_start_timestep,
                    num_timesteps_to_sample=update_timestep_size,
                    solver=solver,
                    video_conditioning_cfg_scale=video_conditioning_cfg_scale,
                    camera_motion_conditioning_cfg_scale=camera_motion_conditioning_cfg_scale,
                    text_conditioning_cfg_scale=text_conditioning_cfg_scale,
                    track_rate_conditioning_cfg_scale=track_rate_conditioning_cfg_scale,
                    apply_noise_schedule=apply_noise_schedule,
                    noised=noised,
                    track_predictor_cache_position=torch.LongTensor([track_predictor_cache_position]).to(input_ids.device).expand(input_ids.shape[0]),
                )

        if update_num_steps > 0:
            for timestep in range(start_timestep, input_ids.shape[2], update_timestep_size):
                if update_noising_start_timestep > 0:
                    labels = new_input_ids[:, :, timestep - update_timestep_size + 1:timestep + 1]
                    if self.relative_prediction:
                        labels = labels - new_input_ids[:, :, timestep - update_timestep_size:timestep]
                    noised = torch.randn_like(labels)
                    noised = update_noising_start_timestep * labels + (1 - update_noising_start_timestep) * noised
                else:
                    noised = None
                track_predictor_cache_position = timestep - 1 + self.config.text_encoder_max_seq_length if self.config.text_encoder_name is not None else 0
                step_trajectory = self.sample_forward_pass(
                    input_ids=new_input_ids[..., timestep - 1 if use_kv_cache else 0:timestep, :],
                    past_noised=past_noised[..., timestep - 2 if use_kv_cache else 0:timestep - 1, :] if not self.config.late_noise_conditioning else None,
                    pixel_values=pixel_values,
                    global_pixel_values=global_pixel_values,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    camera_motion=camera_motion,
                    track_rate=track_rate,
                    attention_mask=attention_mask[:, :, timestep - 1 if use_kv_cache else 0:timestep] if attention_mask is not None else None,
                    num_steps=update_num_steps,
                    use_kv_cache=use_kv_cache,
                    update_noising_start_timestep=update_noising_start_timestep,
                    num_timesteps_to_sample=update_timestep_size,
                    solver=solver,
                    video_conditioning_cfg_scale=video_conditioning_cfg_scale,
                    camera_motion_conditioning_cfg_scale=camera_motion_conditioning_cfg_scale,
                    text_conditioning_cfg_scale=text_conditioning_cfg_scale,
                    track_rate_conditioning_cfg_scale=track_rate_conditioning_cfg_scale,
                    apply_noise_schedule=apply_noise_schedule,
                    noised=noised,
                    track_predictor_cache_position=torch.LongTensor([track_predictor_cache_position]).to(input_ids.device).expand(input_ids.shape[0]),
                )
                self.encoded_text = True
                if not self.config.late_noise_conditioning:
                    past_noised[:, :, :, timestep - update_timestep_size:timestep] = step_trajectory
                final_prediction = step_trajectory[-1]
                #final_prediction[..., :2] = final_prediction[..., :2].flip(-1)
                if self.relative_prediction:
                    final_prediction = new_input_ids[:, :, timestep - update_timestep_size:timestep] + final_prediction

                new_input_ids[:, :, timestep:timestep + update_timestep_size] = final_prediction
        
        for timestep in range(input_ids.shape[2], max_track_length + 1):
            step_trajectory = self.sample_forward_pass(
                input_ids=new_input_ids[..., timestep - 1 if use_kv_cache else 0:timestep, :],
                past_noised=past_noised[..., timestep - 2 if use_kv_cache else 0:timestep - 1, :] if not self.config.late_noise_conditioning else None,
                pixel_values=pixel_values,
                global_pixel_values=global_pixel_values,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                camera_motion=camera_motion,
                track_rate=track_rate,
                use_kv_cache=use_kv_cache,
                attention_mask=attention_mask[:, :, timestep - 1 if use_kv_cache else 0:timestep] if attention_mask is not None else None,
                num_steps=num_steps,
                solver=solver,
                video_conditioning_cfg_scale=video_conditioning_cfg_scale,
                camera_motion_conditioning_cfg_scale=camera_motion_conditioning_cfg_scale,
                text_conditioning_cfg_scale=text_conditioning_cfg_scale,
                track_rate_conditioning_cfg_scale=track_rate_conditioning_cfg_scale,
                apply_noise_schedule=apply_noise_schedule,
            )
            self.encoded_text = True
            if not self.config.late_noise_conditioning:
                past_noised[:, :, :, timestep - 1:timestep] = step_trajectory
            final_prediction = step_trajectory[-1]
            #final_prediction[..., :2] = final_prediction[..., :2].flip(-1)
            if self.relative_prediction:
                final_prediction = new_input_ids[:, :, timestep - 1:timestep] + final_prediction

            new_input_ids[:, :, timestep:timestep + 1] = final_prediction
        #if use_kv_cache:
        #    return new_input_ids
        return MotionPredictorOutput(
            predictor_output=new_input_ids,
            track_predictor_past_key_values=self.track_predictor_past_key_values
        )
            

    def forward(
        self,
        input_ids: torch.FloatTensor,
        labels: torch.FloatTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        global_pixel_values: Optional[torch.FloatTensor] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        text_attention_mask: Optional[torch.FloatTensor] = None,
        text_embedding: Optional[torch.FloatTensor] = None,
        pooled_text_embedding: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        camera_motion: Optional[torch.LongTensor] = None,
        track_rate: Optional[torch.FloatTensor] = None,
        video_encoder_past_key_values: Optional[torch.FloatTensor] = None,
        track_predictor_hidden_state: Optional[torch.FloatTensor] = None,
        track_predictor_past_key_values: Optional[torch.FloatTensor] = None,
        video_feature_map: Optional[torch.FloatTensor] = None,
        label_mask: Optional[torch.BoolTensor] = None,
    ):
        if TIME_DEBUG:
            import time
            torch.cuda.synchronize()
            start_time = time.time()
        
        if self.relative_prediction:
            labels = labels - input_ids
        noise = torch.randn_like(labels)
        times = torch.rand(labels.shape[0], device=labels.device)[:, None, None, None]
        noised, times, flow = self.get_noised_times_flows(labels, noise, times)
        
        if TIME_DEBUG:
            torch.cuda.synchronize()
            print(f"{self.device}, processing time: {time.time() - start_time}")
            start_time = time.time()
        motion_predictor_outputs = self.motion_predictor(
            input_ids=input_ids,
            pixel_values=pixel_values,
            global_pixel_values=global_pixel_values,
            text_input_ids=text_input_ids,
            attention_mask=attention_mask,
            text_attention_mask=text_attention_mask,
            video_encoder_past_key_values=video_encoder_past_key_values,
            track_predictor_hidden_state=track_predictor_hidden_state,
            track_predictor_past_key_values=track_predictor_past_key_values,
            video_feature_map=video_feature_map,
            text_embedding=text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            camera_motion=camera_motion,
            track_rate=track_rate,
            noised=noised,
            times=times,
        )
        if TIME_DEBUG:
            torch.cuda.synchronize()
            print(f"model time: {time.time() - start_time}")
            start_time = time.time()
        
        flow_prediction = motion_predictor_outputs.predictor_output
        data_prediction = noised - flow_prediction * (1 - times)
        
        if self.use_consistency and labels is not None:
            ema_noised, ema_times, ema_flow = self.get_noised_times_flows(labels, noise, times, consistency_prediction=True)
            ema_motion_predictor_outputs = self.ema_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                global_pixel_values=global_pixel_values,
                text_input_ids=text_input_ids,
                attention_mask=attention_mask,
                text_attention_mask=text_attention_mask,
                video_encoder_past_key_values=video_encoder_past_key_values,
                track_predictor_hidden_state=track_predictor_hidden_state,
                track_predictor_past_key_values=track_predictor_past_key_values,
                text_embedding=text_embedding,
                pooled_text_embedding=pooled_text_embedding,
                camera_motion=camera_motion,
                track_rate=track_rate,
                noised=noised,
                times=times,
            )
            ema_flow_prediction = ema_motion_predictor_outputs.predictor_output
            ema_data_prediction = ema_noised - ema_flow_prediction * (1 - ema_times)
        
        if TIME_DEBUG:
            torch.cuda.synchronize()
            print(f"ema_model time: {time.time() - start_time}")
            start_time = time.time()

        loss = None
        if labels is not None:
            # Add this functionality later perhaps
            if self.config.timestep_exponential_decay_loss_factor is not None:
                timestep_exponential_decay_scaling = 2 ** (-torch.arange(flow_prediction.shape[2]) * self.config.timestep_exponential_decay_loss_factor / (self.config.max_track_length))
                timestep_exponential_decay_scaling = timestep_exponential_decay_scaling / timestep_exponential_decay_scaling.mean()
                
                timestep_exponential_decay_scaling = timestep_exponential_decay_scaling.to(flow_prediction.device)
                timestep_exponential_decay_scaling = timestep_exponential_decay_scaling[None, None, :, None]
            else:
                timestep_exponential_decay_scaling = torch.ones_like(flow_prediction)
            
            loss = (F.mse_loss(flow_prediction, flow, reduction="none") * timestep_exponential_decay_scaling)[label_mask].mean()
            
            if self.use_consistency:
                data_match_loss = (F.mse_loss(data_prediction, ema_data_prediction, reduction="none") * timestep_exponential_decay_scaling)[label_mask].mean()
                velocity_match_loss = (F.mse_loss(flow_prediction, ema_flow_prediction, reduction="none") * timestep_exponential_decay_scaling)[label_mask].mean()
                consistency_loss = data_match_loss + velocity_match_loss * self.consistency_velocity_match_alpha
                loss = loss + consistency_loss * self.consistency_loss_weight
                
        if TIME_DEBUG:
            torch.cuda.synchronize()
            print(f"loss time: {time.time() - start_time}")
            start_time = time.time()
        
        return MotionPredictorOutput(
            loss=loss,
            predictor_output=flow_prediction,
            track_predictor_hidden_state=motion_predictor_outputs.track_predictor_hidden_state,
            track_predictor_past_key_values=motion_predictor_outputs.track_predictor_past_key_values,
            video_encoder_past_key_values=motion_predictor_outputs.video_encoder_past_key_values,
            hidden_states=motion_predictor_outputs.hidden_states,
            attentions=motion_predictor_outputs.attentions,
            video_feature_map=motion_predictor_outputs.video_feature_map,
            text_embedding=motion_predictor_outputs.text_embedding,
            pooled_text_embedding=motion_predictor_outputs.pooled_text_embedding,
        )