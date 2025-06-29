from transformers import PretrainedConfig, AutoConfig

class SpatialTrackConfig(PretrainedConfig):

    model_type = "spatial_track"
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=2046,
        num_hidden_layers=4,
        num_attention_heads=12,
        num_key_value_heads=12,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        attention_bias=True,
        attention_dropout=0.0,
        mlp_bias=True,
        head_dim=None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads

AutoConfig.register("spatial_track", SpatialTrackConfig)