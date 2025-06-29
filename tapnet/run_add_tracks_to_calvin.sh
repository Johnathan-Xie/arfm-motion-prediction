source ~/.bashrc
cd /viscam/projects/generalized-motion-prediction/tapnet
conda activate tapnet_env

python3 add_tracks_to_calvin.py --input_path /scr/jwxie/task_ABCD_D --output_path /scr/jwxie/task_ABCD_D_10percent_with_tracks --relative_annotation_path "lang_annotations/auto_lang_ann_10percent.npy"