from transformers import PretrainedConfig, AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from motion_predictor.prediction_head import SpatialTrackConfig

def load_config_dict(input_dict):
    if isinstance(input_dict, dict):
        config_class = CONFIG_MAPPING[input_dict["model_type"]]
        return config_class.from_dict(input_dict)
    return input_dict

class QueryPredictorConfig(PretrainedConfig):

    model_type = "query_predictor"
    keys_to_ignore_at_inference = ["past_key_values"]
    def __init__(
        self,
        image_model_name="facebook/sam2.1-hiera-tiny",
        freeze_image_model=True,
        text_encoder_name="openai/clip-vit-base-patch32",
        freeze_text_encoder=True,
        text_encoder_max_seq_length=64,
        text_conditioning_drop_prob=0.1,
        conv_block_hidden_size=352,
        num_blocks=4,
        encoder_image_size=512,
        height=256,
        width=256,
        track_subsample_count=1000,
        drop_path_prob=0.0,
        consistency_delta_time=1e-3,
        consistency_velocity_match_alpha=1e-5,
        consistency_loss_weight=1.0,
        rectified_flow_use_consistency=True,
        rectified_flow_ema_beta=0.9999,
        rectified_flow_ema_update_every=100,
        noise_schedule_type="cos",
        track_dimensionality=2,
        depth_track_multiplier=1.0,
        predictor_hidden_size=512,
        denoising_predictor_config=SpatialTrackConfig(
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=8,
            num_key_value_heads=8,
        ),
        use_camera_motion_conditioning=False,
        camera_motion_conditioning_drop_prob=0.1,
        image_encoder_hidden_size=352,
        out_channels=2,
        predictor_type="rectified_flow",
        movement_inference_multiplier=1.0,
        canonical_track_rate=None,
        predict_visible_ratios=True,
        prepend_query_points=True,
        max_track_length=50,
        movement_mean=2.5,
        movement_std=12.0,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.image_model_name = image_model_name
        self.image_encoder_hidden_size = image_encoder_hidden_size
        self.freeze_image_model = freeze_image_model
        self.text_encoder_name = text_encoder_name
        self.freeze_text_encoder = freeze_text_encoder
        self.text_encoder_max_seq_length = text_encoder_max_seq_length
        self.text_conditioning_drop_prob = text_conditioning_drop_prob
        self.conv_block_hidden_size = conv_block_hidden_size
        self.num_blocks = num_blocks
        self.encoder_image_size = encoder_image_size
        self.height = height
        self.width = width
        
        self.track_subsample_count = track_subsample_count
        self.drop_path_prob = drop_path_prob

        self.rectified_flow_use_consistency = rectified_flow_use_consistency
        self.consistency_delta_time = consistency_delta_time
        self.consistency_velocity_match_alpha = consistency_velocity_match_alpha
        self.consistency_loss_weight = consistency_loss_weight
        self.rectified_flow_ema_beta = rectified_flow_ema_beta
        self.rectified_flow_ema_update_every = rectified_flow_ema_update_every
        self.noise_schedule_type = noise_schedule_type

        self.track_dimensionality = track_dimensionality
        self.depth_track_multiplier = depth_track_multiplier
        
        self.out_channels = out_channels
        self.denoising_predictor_config = load_config_dict(denoising_predictor_config)
        if self.denoising_predictor_config is not None:
            self.denoising_predictor_config.track_dimensionality = track_dimensionality
            self.denoising_predictor_config.out_channels = out_channels
            self.denoising_predictor_config.hidden_size = predictor_hidden_size
        
        self.predictor_hidden_size = predictor_hidden_size
        self.use_camera_motion_conditioning = use_camera_motion_conditioning
        self.camera_motion_conditioning_drop_prob = camera_motion_conditioning_drop_prob
        self.predictor_type = predictor_type
        self.movement_inference_multiplier = movement_inference_multiplier
        self.canonical_track_rate = canonical_track_rate
        self.predict_visible_ratios = predict_visible_ratios
        self.prepend_query_points = prepend_query_points
        self.max_track_length = max_track_length
        self.movement_mean = movement_mean
        self.movement_std = movement_std

AutoConfig.register("query_predictor", QueryPredictorConfig)