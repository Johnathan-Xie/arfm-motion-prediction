model="jxie/autorf-zero_shot-motion_predictor" # Enter path to model
main_process_port=25904
dataset="jxie/UCF_101-100_samples-100_points-50_movement_min-mwt0.5" # Swap with some other dataset on hub or local
optim_name=adamw_torch
num_gpus=1
nodes=1
max_track_count=100
min_track_count=100
mask_camera_motion_indicator=false
split="test"
evaluation_num_frames=1
provide_tracks_for_visible_frames=true
thresholds="4 8 16 32 64"

# Disable heartbeat monitor
TORCH_NCCL_ENABLE_MONITORING=0
export RUN_NAME="$(basename $model)-$(basename $dataset)"
export TOKENIZERS_PARALLELISM=false

accelerate launch --config_file ddp_config.yaml --num_processes $num_gpus --main_process_port $main_process_port \
    run_evaluation_motion_predictor.py \
    --output_dir evaluation/$RUN_NAME \
    --model_name_or_path $model \
    --do_eval \
    --dataset $dataset \
    --validation_split $split \
    --per_device_eval_batch_size 1 \
    --seed 101 \
    --ddp_timeout 3000 \
    --dataloader_num_workers 4 \
    --mask_non_visible_tracks \
    --min_track_count $min_track_count \
    --mask_camera_motion_indicator $mask_camera_motion_indicator \
    --write_visualizations \
    --evaluation_num_frames $evaluation_num_frames \
    --provide_tracks_for_visible_frames $provide_tracks_for_visible_frames \
    --sampling_apply_noise_schedule false \
    --thresholds $thresholds \
    --visualization_horizon 16 
    # done
echo "Done"
