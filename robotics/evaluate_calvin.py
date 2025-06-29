# MIT License

# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Code to evaluate Calvin."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import mediapy as media
import copy
from moviepy.editor import ImageSequenceClip
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

#sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from evaluation import GR1CalvinEvaluation
from utils.calvin_utils import print_and_save
import clip
from PreProcess import PreProcess
import models.vision_transformer as vits
from models.gr1 import GR1 
from argparse import ArgumentParser
from motion_predictor import MotionPredictorPipeline, TrackEncoder
from motion_predictor.viz_utils import paint_future_point_track

logger = logging.getLogger(__name__)

os.environ["FFMPEG_BINARY"] = "auto-detect"
CALVIN_ROOT = os.environ["CALVIN_ROOT"]

def make_env(dataset_path, observation_space, device):
    val_folder = Path(dataset_path) / "validation"
    from evaluation.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def evaluate_policy(model, env, eval_sr_path, eval_result_path, ep_len, num_sequences, num_procs, procs_id, eval_dir=None, visualize=False):
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(num_sequences)
    num_seq_per_procs = num_sequences // num_procs
    eval_sequences = eval_sequences[num_seq_per_procs*procs_id:num_seq_per_procs*(procs_id+1)]

    results = []
    
    eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, visualize, eval_dir, sequence_i, ep_len)
        results.append(result)
        success_list = count_success(results)
        with open(eval_sr_path, "a") as f:
            line =f"{sequence_i}/{num_sequences}: "
            for sr in success_list:
                line += f"{sr:.3f} | "
            sequence_i += 1
            line += "\n"
            f.write(line)
        eval_sequences.set_description(
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
        )
    print_and_save(results, eval_sequences, eval_result_path, None)
    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, visualize, eval_dir, sequence_i, ep_len):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0

    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(env, model, task_checker, subtask, val_annotations, visualize, eval_dir, subtask_i, sequence_i, ep_len)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def write_visualization(visualization_dict, eval_dir, sequence_i, subtask_i, subtask, success):
    predicted_point_tracks = visualization_dict.pop("predicted_future_point_tracks")
    for key in visualization_dict.keys():
        if key == "static":
            if predicted_point_tracks is not None and len(predicted_point_tracks):
                frames = media.resize_video(visualization_dict["static"], (256, 256))
                frames = np.stack(frames)
                predicted_point_tracks = np.stack(predicted_point_tracks)
                visualization = paint_future_point_track(frames, predicted_point_tracks)
                clip = ImageSequenceClip([i for i in visualization], fps=30)
                clip.write_gif(os.path.join(eval_dir, f"{sequence_i}-{subtask_i}-{subtask}-{'success' if success else 'fail'}.gif"), fps=30) 
            else:
                clip = ImageSequenceClip(visualization_dict[key], fps=30)
                clip.write_gif(os.path.join(eval_dir, f"{sequence_i}-{subtask_i}-{subtask}-{'success' if success else 'fail'}.gif"), fps=30)

def rollout(env, model, task_oracle, subtask, val_annotations, visualize, eval_dir, subtask_i, sequence_i, ep_len):
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    if visualize:
        visualization_dict = {
            "static": [],
            "gripper": [],
            "pred_static": [],
            "pred_gripper": [],
            "predicted_future_point_tracks": []
        }
    unfinished = 0
    for step in range(ep_len):
        if unfinished == 0:
            output = model.step(obs, lang_annotation)
            action = output["action_pred"]
            unfinished = action.shape[0]
        obs, _, _, current_info = env.step(action[-unfinished])
        unfinished -= 1
        if visualize:
            visualization_dict["static"].append(copy.deepcopy(obs["rgb_obs"]["rgb_static"]))
            visualization_dict["gripper"].append(copy.deepcopy(obs["rgb_obs"]["rgb_gripper"]))
            visualization_dict["pred_static"].append(copy.deepcopy(output["obs_preds"][0, -1].astype(np.uint8)))
            visualization_dict["pred_gripper"].append(copy.deepcopy(output["obs_hand_preds"][0, -1].astype(np.uint8)))
            if output.get("predicted_tracks") is not None:
                visualization_dict["predicted_future_point_tracks"].append(copy.deepcopy(output["predicted_tracks"]))
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if visualize:
                write_visualization(visualization_dict, eval_dir, sequence_i, subtask_i, subtask, success=True)
            return True
    if visualize:
        print(colored("fail", "red"), end=" ")
        write_visualization(visualization_dict, eval_dir, sequence_i, subtask_i, subtask, success=False)
    return False


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs.json")
    parser.add_argument("--write_visualizations", default=False, action="store_true")
    parser.add_argument("--flip_tracks", default=False, action="store_true")
    args = parser.parse_args()
    return args

def main():
    # Preparation
    args = parse_args()

    cfg = json.load(open(args.config_path))
    cfg["record_evaluation_video"] = args.write_visualizations
    
    # The timeout here is 3600s to wait for other processes to finish the simulation
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    device = acc.device
    preprocessor = PreProcess(
        cfg["rgb_static_pad"],
        cfg["rgb_gripper_pad"],
        cfg["rgb_shape"],
        cfg["rgb_mean"],
        cfg["rgb_std"],
        device,
    )
    model_clip, _ = clip.load(cfg["clip_backbone"], device=device) 
    model_mae = vits.__dict__["vit_base"](patch_size=16, num_classes=0).to(device)
    checkpoint = torch.load(cfg["mae_ckpt"])
    model_mae.load_state_dict(checkpoint["model"], strict=False)
    if cfg["use_future_tracks"]:
        track_encoder = TrackEncoder(track_length=cfg["track_length"], **cfg["track_encoder_kwargs"]).eval()
    else:
        track_encoder = None
    model = GR1(
        model_clip,
        model_mae,
        rgb_shape=cfg["rgb_shape"],
        patch_size=cfg["patch_size"],
        state_dim=cfg["state_dim"],
        act_dim=cfg["act_dim"],
        hidden_size=cfg["embed_dim"],
        sequence_length=cfg["seq_len"],
        chunk_size=cfg["chunk_size"],
        training_target=["act_pred", "fwd_pred", "fwd_pred_hand"],
        img_feat_dim=cfg["img_feat_dim"],
        patch_feat_dim=cfg["patch_feat_dim"],
        lang_feat_dim=cfg["lang_feat_dim"],
        resampler_params={
            "depth": cfg["resampler_depth"],
            "dim_head": cfg["resampler_dim_head"],
            "heads": cfg["resampler_heads"],
            "num_latents": cfg["resampler_num_latents"],
            "num_media_embeds": cfg["resampler_num_media_embeds"],
        },
        without_norm_pixel_loss=cfg["without_norm_pixel_loss"],
        skip_frame=cfg["skip_frame"],
        use_hand_rgb=True,
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_inner=4*cfg["embed_dim"],
        activation_function=cfg["activation_function"],
        n_positions=cfg["n_positions"],
        resid_pdrop=cfg["dropout"],
        attn_pdrop=cfg["dropout"],
        use_future_tracks=cfg["use_future_tracks"],
        track_encoder=track_encoder
    ).to(device)  # for fused optimizer

    if cfg.get("checkpoint_path") is not None:
        model.load_state_dict(torch.load(cfg["checkpoint_path"])["state_dict"], strict=False)
        acc.print("load ", cfg["checkpoint_path"] )
    if cfg["compile_model"]:
        model = torch.compile(model)
    model = acc.prepare(model, device_placement=[True])
    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"], 
        "depth_obs": [], 
        "state_obs": ["robot_obs"], 
        "actions": ["rel_actions"], 
        "language": ["language"]}
    eval_dir = cfg["save_path"]+f"eval{torch.cuda.current_device()}/"
    os.makedirs(eval_dir, exist_ok=True)
    env = make_env("./fake_dataset", observation_space, device)
    motion_prediction_pipeline_keys = ["point_tracker_path", "query_predictor_path", "motion_predictor_path", "query_predictor_movement_inference_multiplier"]
    motion_predicion_pipeline_kwargs = {k:cfg.get(k) for k in motion_prediction_pipeline_keys}
    if all(v is not None for v in motion_predicion_pipeline_kwargs.values()):
        motion_predictor_pipeline = MotionPredictorPipeline(
            **motion_predicion_pipeline_kwargs,
            device=acc.device
        )
    else:
        motion_predictor_pipeline = None
    motion_predictor_sampling_kwargs_default = {
        "num_timesteps_to_sample": 15,
        "num_steps": 16,
        "update_num_steps": 4,
        "update_noising_start_timestep": 0.8,
        "use_kv_cache": False
    }
    evaluation_pipeline = GR1CalvinEvaluation(
        model,
        cfg,
        preprocessor,
        device,
        motion_predictor_pipeline=motion_predictor_pipeline,
        motion_predictor_sampling_kwargs=cfg.get("motion_predictor_sampling_kwargs") or motion_predictor_sampling_kwargs_default,
        reset_point_prediction_on_new_goal=cfg.get("reset_point_prediction_on_new_goal") or False
    )
    model.eval()
    avg_reward = torch.tensor(evaluate_policy(
        evaluation_pipeline, 
        env,
        cfg["save_path"]+"success_rate.txt", 
        cfg["save_path"]+"result.txt", 
        cfg["ep_len"],
        cfg["num_sequences"],
        acc.num_processes,
        acc.process_index,
        eval_dir,
        visualize=cfg["record_evaluation_video"],
    )).float().mean().to(device)
    acc.wait_for_everyone()
    avg_reward = acc.gather_for_metrics(avg_reward).mean()
    if acc.is_main_process:
        print("average success rate ", avg_reward)

if __name__ == "__main__":
    main()
