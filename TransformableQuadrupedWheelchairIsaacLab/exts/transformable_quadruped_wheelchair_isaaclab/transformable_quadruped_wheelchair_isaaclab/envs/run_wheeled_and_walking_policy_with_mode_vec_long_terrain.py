# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.task = "TQW-Change-Mode-Rule-v0"
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time
import json
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path
import csv
from collections import defaultdict

from rsl_rl.runners import OnPolicyRunner
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import numpy as np

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion
from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.two_modes_env_cfg import (
    DEFAULT_ACTION_JOINTS_VEL,
    WALKING_ACTION_JOINTS_POS,
    WHEELED_ACTION_JOINTS_POS,
    WALKING_OFFSET,
    WHEEL_OFFSET,
    WALKING_MODE_PREFERRED_ANGLES,
    WHEELED_MODE_PREFERRED_ANGLES
)
from transformable_quadruped_wheelchair_isaaclab.envs.gym_wrappers.save_trajectory import MultiEnvDistillWrapper
from transformable_quadruped_wheelchair_isaaclab.envs.utils.utils import (
    load_policy_torch,
    run_policy_torch,
    post_process_walking_mode,
    unwrap_env,
    get_joint_names,
    get_joint_idxs_offsets,
    get_joint_idxs_velues,
    apply_offset,
    apply_fixed_value,
    action_idx
)

def load_joint_meta():
    current_file = os.path.abspath(__file__)
    base_dir = os.path.dirname(current_file)
    
    joint_index_map_path = os.path.join(base_dir, "meta", "joint_index_map.json")
    walking_joints_path   = os.path.join(base_dir, "meta", "walking_joints.json")
    wheeled_joints_path   = os.path.join(base_dir, "meta", "wheeled_joints.json")
    
    with open(joint_index_map_path, "r", encoding="utf-8") as f:
        joint_index_map_data = json.load(f)
    with open(walking_joints_path, "r", encoding="utf-8") as f:
        walking_joints_data = json.load(f)
    with open(wheeled_joints_path, "r", encoding="utf-8") as f:
        wheeled_joints_data = json.load(f)
    
    return joint_index_map_data, walking_joints_data, wheeled_joints_data

dwell_time = 5.0  
blend_time = 3.0  
cycle_time = 2 * dwell_time + 2 * blend_time  

def compute_mu(cycle_pos: float) -> float:
    global dwell_time, blend_time, cycle_time
    if cycle_pos < dwell_time:
        return 1.0

    if cycle_pos < dwell_time + blend_time:
        t_rel = cycle_pos - dwell_time  # 0 <= t_rel < blend_time
        return 1.0 - (t_rel / blend_time)

    if cycle_pos < 2 * dwell_time + blend_time:
        return 0.0

    t_rel = cycle_pos - (2 * dwell_time + blend_time)  # 0 <= t_rel < blend_time
    return t_rel / blend_time

def main(merged_policy):
    global dwell_time, blend_time, cycle_time

    WALKING_MODE_ONLY = True # True or False

    joint_index_map_data, walking_joints_data, wheeled_joints_data = load_joint_meta()
    print(f"joint_index_map_data: {joint_index_map_data}")
    print(f"walking_joints_data: {walking_joints_data}")
    print(f"wheeled_joints_data: {wheeled_joints_data}")
    
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )

    device = args_cli.device
 
    env_cfg.events.apply_action.params["scale"] = 1.0
    env_cfg.scene.num_envs = 100

    env = gym.make(
        args_cli.task, 
        cfg=env_cfg, 
        render_mode="rgb_array" if args_cli.video else None
    )
    
    obs, _ = env.reset()

    num_envs      = env_cfg.scene.num_envs          # 例: 100
    GOAL_THRESH   = 60.0                            # ゴール判定 Δy
    CSV_PATH      = SCRIPT_DIR / "env_results.csv"  # 保存先
    results       = {}   # env_id -> dict(Δy, time, goal)

    action_dim   =  env.action_space.shape[0] 
    prev_actions = torch.zeros(env_cfg.scene.num_envs, action_dim, device=args_cli.device)

    # 1) 物理時間刻み dt を取得
    if hasattr(env.unwrapped, "step_dt"):
        dt = env.unwrapped.step_dt
    elif hasattr(env.unwrapped, "dt"):
        dt = env.unwrapped.dt
    elif hasattr(env.unwrapped, "cfg") and hasattr(env.unwrapped.cfg, "sim") and hasattr(env.unwrapped.cfg.sim, "dt"):
        dt = env.unwrapped.cfg.sim.dt
    else:
        dt = 0.02

    episode_steps = torch.zeros(env_cfg.scene.num_envs, dtype=torch.long, device="cpu")
    
    THRESH = 5.0

    sim = env.unwrapped.sim 

    root_pos_all = env.unwrapped.scene["robot"].data.root_pos_w   # (n_envs, 3)
    start_y = root_pos_all.clone()[:, 1]

    sim_running = True

    while sim_running and simulation_app.is_running():
        with torch.inference_mode():
            episode_steps += 1

            obs_policy = obs["policy"].clone()
            root_pos_all = env.unwrapped.scene["robot"].data.root_pos_w
            current_y    = root_pos_all[:, 1]

            bad_prev = (prev_actions.abs() > THRESH).any(dim=1)          # shape (n_envs,)
            if bad_prev.any():
                obs_policy[bad_prev, 68:88] = 0.0  

            if WALKING_MODE_ONLY:
                obs_policy[:, -2] = 0.0
                obs_policy[:, -1] = 1.0

            mode_mask = (obs_policy[:, -2] > 0.5) & (obs_policy[:, -1] < 0.5)

            actions = merged_policy(obs_policy) 
            obs, r, term, trunc, infos = env.step(actions)
            prev_actions = actions.detach().clone()

            root_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]   # (x, y, z) world

            EYE_OFFSET     = torch.tensor([5.0, 5.0, 3.0],  device=root_pos.device)   # +5,+5,+3 m
            TARGET_OFFSET  = torch.tensor([0.0, 0.0, -0.0], device=root_pos.device)  # 50 cm 下

            cam_eye    = root_pos + EYE_OFFSET
            cam_target = root_pos + TARGET_OFFSET 

            done_mask = (term | trunc).cpu().numpy()
            for idx in np.nonzero(done_mask)[0]:
                dy = (current_y[idx] - start_y[idx]).item()            # Δy
                steps = int(episode_steps[idx].item())                 # ステップ数
                elapsed_time = steps * dt                              # 経過秒

                if idx not in results:
                    if dy < 0.0:
                        start_y[idx]      = current_y[idx]
                        episode_steps[idx] = 0
                        continue

                    results[idx] = dict(
                        delta_y=dy,
                        elapsed_time=elapsed_time if dy > GOAL_THRESH else "",
                        goal=dy > GOAL_THRESH,
                    )

                if dy > GOAL_THRESH:
                    print(f"[Env {idx}] GOAL! Δy : {dy:+.2f} m | time : {elapsed_time:.2f} s | steps : {steps}")
                else:
                    print(f"[Env {idx}] Δy : {dy:+.2f} m")

                start_y[idx]       = current_y[idx]
                episode_steps[idx] = 0

            if len(results) == num_envs:
                sim_running = False         
                break

    env.close()

    delta_all    = [r["delta_y"] for r in results.values()]
    goal_times   = [r["elapsed_time"] for r in results.values() if r["goal"]]
    avg_delta    = np.mean(delta_all) if delta_all else 0.0
    avg_goal_t   = np.mean(goal_times) if goal_times else 0.0
    goal_ratio   = len(goal_times) / num_envs

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["env_id", "delta_y", "elapsed_time", "goal"])  # ヘッダ

        for env_id in range(num_envs):
            row = results.get(env_id, {"delta_y": "", "elapsed_time": "", "goal": ""})
            writer.writerow([env_id, row["delta_y"], row["elapsed_time"], row["goal"]])

        writer.writerow([])                                 # 空行で区切り
        writer.writerow(["平均移動距離", avg_delta])
        writer.writerow(["平均ゴール到達時間", avg_goal_t])
        writer.writerow(["ゴール到達割合", goal_ratio])

    print(f"[INFO] 結果を {CSV_PATH} に保存しました。")

if __name__ == "__main__":
    SCRIPT_FILE = Path(__file__).resolve()
    SCRIPT_DIR  = SCRIPT_FILE.parent
    merged_policy = load_policy_torch(f"{SCRIPT_DIR}\models\walking_and_wheeled_mode_with_mode_vec_20250629.pt")
    main(merged_policy.to(dtype=torch.float32))
    simulation_app.close()
