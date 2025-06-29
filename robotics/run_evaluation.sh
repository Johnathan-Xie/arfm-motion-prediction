export CALVIN_ROOT=calvin
export num_gpus=4
# abc_d evaluation: configs/evaluation-calvin-abc_d-zeroshot.json
# abcd_d 10 percent evaluation: configs/evaluation-calvin-abcd_d-10percent-zeroshot.json
export config_path=configs/evaluation-calvin-abcd_d-10percent-zeroshot.json
# CUDA_VISIBLE_DEVICES=0 python3
# accelerate launch --config_file accelerate_config.yaml --main_process_port 25902 --num_processes $num_gpus evaluate_calvin.py \

accelerate launch --config_file ddp_config.yaml --main_process_port 25902 --num_processes $num_gpus evaluate_calvin.py \
    --config_path $config_path \