from torch import nn

import torch
import math

from .configuration_prediction_head import SpatialTrackConfig

from typing import Optional, Tuple
from transformers.modeling_outputs import BaseModelOutput
from transformers.activations import ACT2FN

from typing import Union

from transformers import PreTrainedModel
from motion_predictor.modeling_utils import FourierPositionalEncoding2D, SinusoidalEmbedder

class SpatialTrackRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        SpatialTrackRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class SpatialTrackMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class SpatialTrackAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: SpatialTrackConfig):
        super().__init__()
        self.config = config

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class SpatialTrackLayer(nn.Module):
    def __init__(self, config: SpatialTrackConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = SpatialTrackAttention(config=config)

        self.mlp = SpatialTrackMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=config.rms_norm_eps)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        )
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        adaln_conditioning: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        if adaln_conditioning is not None:
            track_length = hidden_states.shape[0] // adaln_conditioning.shape[0]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                [torch.repeat_interleave(i, track_length, dim=0) for i in self.adaln_modulation(adaln_conditioning).chunk(6, dim=-1)]
            )
            
        hidden_states = self.input_layernorm(hidden_states)
        if adaln_conditioning is not None:
            hidden_states = modulate(hidden_states, shift_msa, scale_msa)
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if adaln_conditioning is not None:
            hidden_states = gate_msa * hidden_states
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if adaln_conditioning is not None:
            hidden_states = modulate(hidden_states, shift_mlp, scale_mlp)
        
        hidden_states = self.mlp(hidden_states)
        if adaln_conditioning is not None:
            hidden_states = gate_mlp * hidden_states
        
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

class DenoisingJointTrackPredictorPreTrainedModel(PreTrainedModel):
    config_class = SpatialTrackConfig
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class NoisePredictorHead(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(
        self,
        hidden_states,
        adaln_conditioning=None,
    ):
        if adaln_conditioning is not None:
            shift, scale = self.adaln_modulation(adaln_conditioning).chunk(2, dim=-1)
            hidden_states = modulate(self.norm_final(hidden_states), shift, scale)
        hidden_states = self.linear(hidden_states)
        return hidden_states

class DenoisingJointTrackPredictor(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`SpatialTrackLayer`]

    Args:
        config: JointTrackConfig
    """

    def __init__(
        self,
        feature_size,
        config: SpatialTrackConfig,
        height=256,
        width=256,
        concatenate_original_position=True,
        out_channels=None,
        use_previous_relative_shift_input=False,
        **embedding_kwargs
    ):
        super().__init__()
        self.fourier_embeddings = FourierPositionalEncoding2D(height, width, **embedding_kwargs)
        self.embedding_size = self.fourier_embeddings.height_embeddings.shape[-1] + self.fourier_embeddings.width_embeddings.shape[-1]
        if concatenate_original_position:
            feature_size += self.embedding_size
        self.use_previous_relative_shift_input = use_previous_relative_shift_input
        if use_previous_relative_shift_input:
            feature_size += config.track_dimensionality
        feature_size += out_channels or config.track_dimensionality # noised
        self.time_embedder = SinusoidalEmbedder(config.hidden_size)
        self.feature_projection = nn.Linear(feature_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [SpatialTrackLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.predictor = NoisePredictorHead(config.hidden_size, out_channels or config.track_dimensionality)
        self.concatenate_original_position = concatenate_original_position
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaln modulation layers in DiT blocks:
        for layer in self.layers:
            nn.init.constant_(layer.adaln_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaln_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.predictor.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.predictor.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.predictor.linear.weight, 0)
        nn.init.constant_(self.predictor.linear.bias, 0)

    def forward(
        self,
        input_features: torch.FloatTensor,
        noised: torch.FloatTensor,
        times: torch.FloatTensor,
        original_positions: torch.LongTensor,
        global_conditioning: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        
        if self.use_previous_relative_shift_input:
            relative_shift = original_positions[..., 1:, :] - original_positions[..., :-1, :]
            relative_shift = torch.cat(torch.zeros_like(relative_shift[..., :1, :]), relative_shift, dim=-2)
        # During sampling the input ids will keep accumulating but we only provide noised for the current prediction timestep
        if noised.shape[-2] != input_features.shape[-2]:
            input_features = input_features[..., -noised.shape[-2]:, :]
        if noised.shape[-2] != original_positions.shape[-2]:
            original_positions = original_positions[..., -noised.shape[-2]:, :]

        if self.use_previous_relative_shift_input and noised.shape[-2] != relative_shift.shape[-2]:
            relative_shift = relative_shift[..., -noised.shape[-2]:, :]
        if noised.shape[-2] != attention_mask.shape[-2]:
            attention_mask = attention_mask[:, :, -noised.shape[-2]:]
        if self.concatenate_original_position:
            input_features = torch.cat((input_features, self.fourier_embeddings.position_to_embedding(original_positions)), dim=-1)
        if self.use_previous_relative_shift_input:
            input_features = torch.cat((input_features, relative_shift), dim=-1)
        
        input_features = torch.cat((noised, input_features), dim=-1)
        adaln_conditioning = self.time_embedder(times.squeeze(1).squeeze(1))
        
        if global_conditioning is not None:
            adaln_conditioning = adaln_conditioning + global_conditioning
        hidden_states = self.feature_projection(input_features)
        hidden_states = hidden_states.permute(0, 2, 1, 3)

        pre_flatten_shape = hidden_states.shape
        hidden_states = hidden_states.flatten(0, 1)
        
        if attention_mask is not None:
            spatial_attention_mask = attention_mask.permute(0, 2, 1).flatten(0, 1)
            spatial_attention_mask = spatial_attention_mask[:, None, :, None].expand(-1, -1, -1, spatial_attention_mask.shape[-1])
        else:
            spatial_attention_mask = None
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=spatial_attention_mask,
                output_attentions=output_attentions,
                adaln_conditioning=adaln_conditioning,
                apply_rotary_pos_emb=False
            )

            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = hidden_states.view(pre_flatten_shape)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        flow_prediction = self.predictor(hidden_states, adaln_conditioning.unsqueeze(1))

        return flow_prediction