# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
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
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.task = "TQW-Two-Modes-with-ModeVector-v0"
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

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

# from isaaclab_rl.rsl_rl import (
#     RslRlOnPolicyRunnerCfg,
#     RslRlOnPolicyRunner,
#     RslRlVecEnvWrapper,
#     export_policy_as_jit, 
#     export_policy_as_onnx,
# )

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

# ------------- ブレンド設定 -------------
dwell_time = 5.0    # 各モード（A, B）を純粋に維持する時間（秒）
blend_time = 3.0    # モード切り替え時にブレンドする時間（秒）
cycle_time = 2 * dwell_time + 2 * blend_time  # =16秒

def compute_mu(cycle_pos: float) -> float:
    global dwell_time, blend_time, cycle_time
    """
    cycle_pos が [0, cycle_time) の範囲で与えられている前提。
    「A(5s) -> blend A->B(3s) -> B(5s) -> blend B->A(3s)」を線形に補完して μ を返す
    μ=1 なら pure A (combined_actions_wk)、μ=0 なら pure B (combined_actions_wh)
    """
    # ① [0, dwell_time): 純粋に A (μ=1)
    if cycle_pos < dwell_time:
        return 1.0

    # ② [dwell_time, dwell_time + blend_time): A->B ブレンド区間
    if cycle_pos < dwell_time + blend_time:
        t_rel = cycle_pos - dwell_time  # 0 <= t_rel < blend_time
        # 線形: t_rel=0 → μ=1, t_rel=blend_time → μ=0
        return 1.0 - (t_rel / blend_time)

    # ③ [dwell_time+blend_time, dwell_time+blend_time+dwell_time): 純粋に B (μ=0)
    if cycle_pos < 2 * dwell_time + blend_time:
        return 0.0

    # ④ [2*dwell_time + blend_time, cycle_time): B->A ブレンド区間
    #    cycle_pos はここで 2*dwell_time+blend_time <= cycle_pos < cycle_time
    t_rel = cycle_pos - (2 * dwell_time + blend_time)  # 0 <= t_rel < blend_time
    # 線形: t_rel=0 → μ=0, t_rel=blend_time → μ=1
    return t_rel / blend_time

def handle_terminated_envs(
    rewards, terminateds, truncateds,
    reward_sums, recent_returns,
    obs
):
    # 1) numpy 化
    done_mask = (terminateds | truncateds).cpu().numpy()
    reward_sums += rewards.cpu().numpy()

    # 2) 終了 env ごとにリターン処理
    for idx in np.nonzero(done_mask)[0]:
        ep = reward_sums[idx]
        recent_returns.append(ep)

        # --- nan/inf + |r|>20 を除いて平均計算 ---
        filtered = [r for r in recent_returns if np.isfinite(r) and abs(r) <= 20]
        count = len(filtered)
        if count > 0:
            mean_val = sum(filtered) / count
        else:
            mean_val = float('nan')

        print(f"[Env {idx}] Return={ep:.3f} | Last{count}-mean={mean_val:.3f}")
        reward_sums[idx] = 0.0

def main(merged_policy):
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
    env_cfg.scene.num_envs = 128

    env = gym.make(
        args_cli.task, 
        cfg=env_cfg, 
        render_mode="rgb_array" if args_cli.video else None
    )

    obs, _ = env.reset()

    action_dim   =  env.action_space.shape[0] 
    prev_actions = torch.zeros(env_cfg.scene.num_envs, action_dim, device=args_cli.device)

    THRESH = 5.0

    recent_returns = deque(maxlen=100)
    reward_sums = np.zeros(env_cfg.scene.num_envs, dtype=np.float32)

    MODE_LABEL = "WH" # None, "WK", "WH"

    while simulation_app.is_running():
        with torch.inference_mode():

            obs_policy = obs["policy"].clone()

            if MODE_LABEL == None:
                pass
            elif MODE_LABEL == "WK":
                obs_policy[:, -2:] = torch.tensor([0.0, 1.0], device=obs_policy.device)
            elif MODE_LABEL == "WH":
                obs_policy[:, -2:] = torch.tensor([1.0, 0.0], device=obs_policy.device)

            bad_prev = (prev_actions.abs() > THRESH).any(dim=1)          # shape (n_envs,)
            if bad_prev.any():
                obs_policy[bad_prev, 68:88] = 0.0  

            actions = merged_policy(obs_policy)
        # obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # handle_terminated_envs(rewards, terminateds, truncateds, reward_sums, recent_returns, env, obs, device)
        obs, r, term, trunc, infos = env.step(actions)

        handle_terminated_envs(r, term, trunc, reward_sums, recent_returns, obs)

        prev_actions = actions.detach().clone()

    env.close()

if __name__ == "__main__":
    # run the main function
    SCRIPT_FILE = Path(__file__).resolve()
    SCRIPT_DIR  = SCRIPT_FILE.parent
    merged_policy = load_policy_torch(f"{SCRIPT_DIR}\models\walking_and_wheeled_mode_with_mode_vec_20250630.pt")
    main(merged_policy)
    simulation_app.close()
