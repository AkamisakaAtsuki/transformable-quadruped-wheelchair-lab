# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
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
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.task = "TQW-Two-Modes-Rl-v0"
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# # launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import json
import time
import torch

from rsl_rl.runners import OnPolicyRunner
import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion


from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion # 追加

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

from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.two_modes_env_cfg import (
    DEFAULT_ACTION_JOINTS_VEL,
    WALKING_ACTION_JOINTS_POS,
    WHEELED_ACTION_JOINTS_POS,
    WALKING_OFFSET,
    WALKING_MODE_PREFERRED_ANGLES
)

def main():
 
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.scene.num_envs = 1

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    joint_names = get_joint_names(env)
    
    joint_index_map = {name: idx for idx, name in enumerate(joint_names)}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    meta_dir = os.path.join(script_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    save_path = os.path.join(meta_dir, "joint_index_map.json")

    # JSON に保存
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(joint_index_map, f, indent=2, ensure_ascii=False)

    print(f"Saved joint index map to {save_path}")


    with open(os.path.join(meta_dir, "walking_joints.json"), "w", encoding="utf-8") as f:
        json.dump(WALKING_ACTION_JOINTS_POS, f, indent=2, ensure_ascii=False)

    # 3) 車輪モード用関節リストを wheeled_joints.json に書き出し
    with open(os.path.join(meta_dir, "wheeled_joints.json"), "w", encoding="utf-8") as f:
        json.dump(WHEELED_ACTION_JOINTS_POS, f, indent=2, ensure_ascii=False)

    print(f"Saved joint index map and mode-specific joint lists to {meta_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
