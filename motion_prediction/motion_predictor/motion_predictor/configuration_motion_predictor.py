from transformers import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers import AutoConfig
from motion_predictor.prediction_head import SpatialTrackConfig
from motion_predictor.spacetime_transformer import JointTrackConfig

def load_config_dict(input_dict):
    if isinstance(input_dict, dict):
        config_class = CONFIG_MAPPING[input_dict["model_type"]]
        return config_class.from_dict(input_dict)
    return input_dict

class MotionPredictorConfig(PretrainedConfig):
    model_type = "motion_predictor"

    def __init__(
        self,
        image_model_name="facebook/dinov2-base",
        global_image_model_name=None,
        image_model_batch_size=128,
        freeze_global_image_model=True,
        freeze_image_model=True,
        frame_sample_rate=None,
        encoder_config=None,
        encoder_pooler_type="mean",
        track_predictor_has_spatial_condition=False,
        max_track_length=50,
        track_subsample_count=100,
        track_predictor_head_type="shift_scale",
        position_encoding_norm_to_feature_norm_ratio=1.0,
        width=256,
        height=256,
        max_height_shift=None,
        max_width_shift=None,
        feature_map_upsample_method="bilinear",
        track_predictor_head_kwargs=dict(
            concatenate_original_position=True,
            norm_feature=True,
            predictor_type="linear",
        ),
        continuous_predictor=False,
        consistency_delta_time=1e-3,
        consistency_velocity_match_alpha=1e-5,
        consistency_loss_weight=1.0,
        rectified_flow_use_consistency=True,
        rectified_flow_ema_beta=0.9999,
        rectified_flow_ema_update_every=100,
        rectified_flow_relative_prediction=True,
        extra_features_size=0, # noised shift and time
        track_predictor_config=JointTrackConfig(
            vocab_size=0,
            hidden_size=768,
            intermediate_size=2046,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=1024,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=True,
            attention_dropout=0.0,
            mlp_bias=True,
        ),
        late_noise_conditioning=False,
        denoising_predictor_config=SpatialTrackConfig(num_hidden_layers=4),
        noise_schedule_type="cos",
        text_encoder_name=None,
        freeze_text_encoder=True,
        text_conditioning_drop_prob=0.1,
        text_encoder_max_seq_length=32,
        use_camera_motion_conditioning=False,
        camera_motion_conditioning_drop_prob=0.1,
        use_track_rate_conditioning=False,
        track_rate_conditioning_max_track_rate=60,
        track_dimensionality=2,
        depth_track_multiplier=1.0,
        video_encoder_hidden_size=None,
        global_video_encoder_hidden_size=768,
        num_global_image_tokens=49,
        sam2_image_size=1024,
        video_conditioning_drop_prob=0.0,
        track_rate_conditioning_drop_prob=0.0,
        movement_weighting_temperature=0.5,
        visible_min_ratio=0.5,
        default_track_rate=30,
        timestep_exponential_decay_loss_factor=None,
        use_absolute_positional_embeddings=True,
        use_previous_relative_shift_input=False,
        prepend_query_points=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.image_model_name = image_model_name
        self.global_image_model_name = global_image_model_name
        self.global_video_encoder_hidden_size = global_video_encoder_hidden_size
        self.freeze_global_image_model = freeze_global_image_model
        self.freeze_image_model = freeze_image_model
        self.image_model_batch_size = image_model_batch_size
        self.frame_sample_rate = frame_sample_rate
        self.encoder_pooler_type = encoder_pooler_type

        self.encoder_config = load_config_dict(encoder_config)
        
        self.feature_map_upsample_method = feature_map_upsample_method

        self.max_track_length = max_track_length
        self.track_subsample_count = track_subsample_count
        
        self.width = width
        self.height = height
        self.max_height_shift = max_height_shift
        self.max_width_shift = max_width_shift
        self.track_predictor_head_type = track_predictor_head_type
        self.track_predictor_head_kwargs = track_predictor_head_kwargs
        self.position_encoding_norm_to_feature_norm_ratio = position_encoding_norm_to_feature_norm_ratio
        
        self.rectified_flow_use_consistency = rectified_flow_use_consistency
        self.rectified_flow_relative_prediction = rectified_flow_relative_prediction
        self.consistency_delta_time = consistency_delta_time
        self.consistency_velocity_match_alpha = consistency_velocity_match_alpha
        self.consistency_loss_weight = consistency_loss_weight
        self.rectified_flow_ema_beta = rectified_flow_ema_beta
        self.rectified_flow_ema_update_every = rectified_flow_ema_update_every
        self.continuous_predictor = continuous_predictor

        self.extra_features_size = extra_features_size
        self.track_predictor_config = load_config_dict(track_predictor_config)
        self.track_predictor_has_spatial_condition = track_predictor_has_spatial_condition
        self.late_noise_conditioning = late_noise_conditioning
        self.denoising_predictor_config = load_config_dict(denoising_predictor_config)
        self.noise_schedule_type = noise_schedule_type
        self.text_encoder_name = text_encoder_name
        self.freeze_text_encoder = freeze_text_encoder
        self.text_conditioning_drop_prob = text_conditioning_drop_prob
        self.text_encoder_max_seq_length = text_encoder_max_seq_length
        self.use_camera_motion_conditioning = use_camera_motion_conditioning
        self.camera_motion_conditioning_drop_prob = camera_motion_conditioning_drop_prob
        self.use_track_rate_conditioning = use_track_rate_conditioning
        self.track_rate_conditioning_max_track_rate = track_rate_conditioning_max_track_rate
        self.track_rate_conditioning_drop_prob = track_rate_conditioning_drop_prob
        self.video_conditioning_drop_prob = video_conditioning_drop_prob
        self.num_global_image_tokens = num_global_image_tokens

        self.track_dimensionality = track_dimensionality
        if self.denoising_predictor_config is not None:
            self.denoising_predictor_config.track_dimensionality = track_dimensionality
        self.depth_track_multiplier = depth_track_multiplier
        self.video_encoder_hidden_size = video_encoder_hidden_size
        self.sam2_image_size = sam2_image_size
        self.movement_weighting_temperature = movement_weighting_temperature
        self.visible_min_ratio = visible_min_ratio
        self.default_track_rate = default_track_rate
        self.timestep_exponential_decay_loss_factor = timestep_exponential_decay_loss_factor
        self.use_absolute_positional_embeddings = use_absolute_positional_embeddings
        self.use_previous_relative_shift_input = use_previous_relative_shift_input
        self.prepend_query_points = prepend_query_points

AutoConfig.register("motion_predictor", MotionPredictorConfig)
