model=configs/motion_predictor-config
main_process_port=25901
# This dataset will require around 2TB of storage. If you want to test quickly on just a small dataset you may use this: data_configs/ucf_101.json
dataset=data_configs/human_motion_robotics_zero_shot.json
mixed_precision=no # You can replace with fp16, but training may diverge at around 200k steps
optim_name=adamw_torch
num_gpus=1 # Swap based on number of gpus you have
nodes=1
num_train_epochs=1000 # We only train for around 3 epochs, but the schedule is constant so you can run for as long as you wish
movement_weighting_temperature=0.5
track_skip_max=1
track_skip_min=1
visible_min_ratio=0.5
max_track_count=100
min_track_count=100
evaluation_num_frames=1
evaluation_provide_tracks_for_visible_frames=true
lr=5e-5
weight_decay=0
global_batch_size=16
warmup_steps=0

# Disable heartbeat monitor
export TORCH_NCCL_ENABLE_MONITORING=0

export per_device_batch_size=$((global_batch_size / num_gpus))
echo using batch size $per_device_batch_size per gpu
# IMPORTANT CHANGE WANDB STUFF BELOW or set REPORT_TO to none
export HF_REPORT_TO=wandb
#export WANDB_ENTITY=
#export WANDB_PROJECT=

export WANDB__SERVICE_WAIT=300
export RUN_NAME="$(basename $model)-$(basename $dataset)"
export RUN_NAME="${RUN_NAME:0:255}"
export TOKENIZERS_PARALLELISM=false
#export HF_HOME= # default location is usually ok, but you can use this to change

accelerate launch --config_file ddp_config.yaml --num_processes $num_gpus --main_process_port $main_process_port --mixed_precision $mixed_precision \
    run_training_motion_predictor.py \
    --model_name_or_path $model \
    --do_train \
    --do_eval \
    --dataset $dataset \
    --train_split train \
    --validation_split test \
    --output_dir runs/$RUN_NAME \
    --run_name $RUN_NAME \
    --per_device_train_batch_size $per_device_batch_size \
    --per_device_eval_batch_size 1 \
    --warmup_steps $warmup_steps \
    --lr_scheduler_type constant \
    --logging_steps 100 \
    --save_steps 10000 \
    --learning_rate $lr \
    --weight_decay $weight_decay \
    --num_train_epochs $num_train_epochs \
    --overwrite_output_dir \
    --seed 101 \
    --ddp_timeout 3000 \
    --dataloader_num_workers 8 \
    --movement_weighting_temperature $movement_weighting_temperature \
    --visible_min_ratio $visible_min_ratio \
    --min_track_count $min_track_count \
    --report_to $HF_REPORT_TO \
    --visualization_interval 10000 \
    --sampling_evaluation_interval 10000 \
    --visualization_validation_indices 0,1,2 \
    --track_skip_max $track_skip_max \
    --track_skip_min $track_skip_min \
    --mask_non_visible_tracks \
    --dataloader_pin_memory false \
    --evaluation_provide_tracks_for_visible_frames $provide_tracks_for_visible_frames \
    --evaluation_num_frames $evaluation_num_frames \
    --train_max_frames 1 \
    --save_total_limit 5 \
    --random_start

echo "Done"
