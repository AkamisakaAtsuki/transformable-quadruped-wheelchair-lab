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

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.scene.num_envs = 1

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # # specify directory for logging experiments
    # log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    # log_root_path = os.path.abspath(log_root_path)
    # print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # if args_cli.use_pretrained_checkpoint:
    #     resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
    #     if not resume_path:
    #         print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
    #         return
    # elif args_cli.checkpoint:
    #     resume_path = retrieve_file_path(args_cli.checkpoint)
    # else:
    #     resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load("C:/Users/admin5050/Downloads/wm_pt/model_99999.pt")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # オリジナルの actor-critic モデルを取り出す
    actor_critic = ppo_runner.alg.actor_critic
    
    try:
        std_param = actor_critic.std
        print("Learned std per action-dim:", std_param.detach().cpu().numpy())
    except:
        pass

    # actor の std（もしくは log_std）パラメータを取得
    # モデルによって名前が異なりますが、多くは `actor.std` または `actor.log_std` です
           # もし std なら
    # log_std の場合は exp して std に変換
    # std_param = actor_critic.actor.log_std.exp()


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
