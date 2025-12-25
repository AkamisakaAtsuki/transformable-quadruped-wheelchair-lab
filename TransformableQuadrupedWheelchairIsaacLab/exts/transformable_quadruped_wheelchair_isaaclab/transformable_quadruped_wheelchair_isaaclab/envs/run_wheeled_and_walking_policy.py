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
args_cli.task = "TQW-Two-Modes-Normal-v0"
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
from collections import deque

from rsl_rl.runners import OnPolicyRunner
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

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

def handle_terminated_envs(
    rewards, terminateds, truncateds,
    reward_sums, recent_returns,
    obs
):
    done_mask = (terminateds | truncateds).cpu().numpy()
    reward_sums += rewards.cpu().numpy()

    for idx in np.nonzero(done_mask)[0]:
        ep = reward_sums[idx]
        recent_returns.append(ep)

        filtered = [r for r in recent_returns if np.isfinite(r) and abs(r) <= 20]
        count = len(filtered)
        if count > 0:
            mean_val = sum(filtered) / count
        else:
            mean_val = float('nan')

        print(f"[Env {idx}] Return={ep:.3f} | Last{count}-mean={mean_val:.3f}")
        reward_sums[idx] = 0.0

def main(policy_wk, policy_wh):
    global dwell_time, blend_time, cycle_time

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
    env_cfg.scene.num_envs = 1

    env = gym.make(
        args_cli.task, 
        cfg=env_cfg, 
        render_mode="rgb_array" if args_cli.video else None
    )

    recent_returns = deque(maxlen=100)
    reward_sums = np.zeros(env_cfg.scene.num_envs, dtype=np.float32)

    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy_wh(obs['policy']) 
            # actions = merged_policy(obs['policy']) 
        obs, r, term, trunc, infos = env.step(actions)

        handle_terminated_envs(r, term, trunc, reward_sums, recent_returns, obs)

    env.close()

if __name__ == "__main__":
    # run the main function
    SCRIPT_FILE = Path(__file__).resolve()
    SCRIPT_DIR  = SCRIPT_FILE.parent
    # policy_wk = load_policy_torch(f"{SCRIPT_DIR}\models\distilled_policy_wk_jit.pt")
    # policy_wk = load_policy_torch(f"{SCRIPT_DIR}\models\walking_mode_restricted_obs.pt")
    # policy_wh = load_policy_torch(f"{SCRIPT_DIR}\models\distilled_policy_wh_jit.pt")
    # policy_wh = load_policy_torch(f"{SCRIPT_DIR}\models\wheeled_mode_restricted_obs.pt")
    # merged_policy = load_policy_torch(f"{SCRIPT_DIR}\models\walking_wheeled_distill_policy.pt")
    policy_wk = load_policy_torch(f"{SCRIPT_DIR}\models\walking_mode_restricted_obs_full.pt") # 制限付きの観測と行動で学習＆前処理と後処理を追加しJIT化したモデル
    policy_wh = load_policy_torch(f"{SCRIPT_DIR}\models\wheeled_mode_restricted_obs_full.pt")
    main(policy_wk, policy_wh)
    simulation_app.close()