# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# # add argparse arguments
parser = argparse.ArgumentParser(description="")
# parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.task = "TQW-Two-Modes-Rl-v0"

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# import onnxruntime as ort
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

def load_policy_torch(model_path, device="cuda"):
    policy = torch.jit.load(str(model_path), map_location=device)
    policy.eval()
    return policy

def run_policy_torch(policy, obs_tensor, device="cuda"):
    with torch.no_grad():
        return policy(obs_tensor.to(device))

MODEL_TYPE = "ONNX" # PT, ONNX

def main():
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
    )
    
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None

    # env = gym.make(
    #     args_cli.task, 
    #     cfg=env_cfg, 
    #     render_mode="rgb_array"
    # )

    # obs, _ = env.reset()

    # time.sleep(5)
    # env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()