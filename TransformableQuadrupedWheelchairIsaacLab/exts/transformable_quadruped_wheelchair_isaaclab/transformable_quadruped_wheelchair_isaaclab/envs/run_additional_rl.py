# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

import cli_args

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
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.task = "TQW-Two-Modes-Normal-v0"
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

import os
import time
import torch
import torch.nn as nn
import gymnasium as gym
from pathlib import Path
from datetime import datetime

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
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, 
    RslRlVecEnvWrapper
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion
from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.two_modes_env_cfg import (
    DEFAULT_ACTION_JOINTS_VEL,
    WALKING_ACTION_JOINTS_POS,
    WALKING_OFFSET,
    WALKING_MODE_PREFERRED_ANGLES
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
from transformable_quadruped_wheelchair_isaaclab.utils.rsl_rl.jit_on_policy_runner import JITOnPolicyRunner
from transformable_quadruped_wheelchair_isaaclab.utils.rsl_rl.jit_actor_critic import JITActorCritic
from transformable_quadruped_wheelchair_isaaclab.utils.rsl_rl.algorithms.custom_network_ppo import CustomNetworkPPO
from transformable_quadruped_wheelchair_isaaclab.utils.rsl_rl.modules.custom_network_actor_critic import CustomNetworkActorCritic
from transformable_quadruped_wheelchair_isaaclab.utils.rsl_rl.runners.custom_network_on_policy_runner import CustomNetworkOnPolicyRunner

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    
    SCRIPT_FILE = Path(__file__).resolve()
    SCRIPT_DIR  = SCRIPT_FILE.parent
    policy = load_policy_torch(
        f"{SCRIPT_DIR}\models\distilled_policy_test.pt", 
        device=args_cli.device
    )
    
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )

    device = args_cli.device
 
    env_cfg.events.add_teslabot_mass.params['mass_distribution_params']=(-20.0, -21.0)
    env_cfg.events.apply_action.params["scale"] = 1.0
    env_cfg.scene.num_envs = 128

    # agent_cfg = RslRlOnPolicyRunnerCfg() 
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = device

    # agent_cfg.ppo_actor_critic.class_name     = "JITActorCritic"
    # agent_cfg.ppo_actor_critic.init_noise_std = 0.1
    # agent_cfg.ppo_actor_critic.noise_std_type = "scalar"

    env = gym.make(
        args_cli.task, 
        cfg=env_cfg, 
        render_mode="rgb_array" if args_cli.video else None
    )    

    agent_cfg.experiment_name = "distill_additional_learning"
    # specify directory for logging experiments
    log_root_path = os.path.join(
        "logs", 
        "rsl_rl", 
        agent_cfg.experiment_name
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = RslRlVecEnvWrapper(env)

    custom_actor = policy
    obs, extras = env.get_observations()
    obs_dim = obs.shape[1]
    custom_critic = nn.Sequential(
        nn.Linear(obs_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    ).to(args_cli.device)

    teacher_std_np = [
        1.4931484,  0.4979939,  0.49002692, 0.40329322, 0.4002882,  1.5164722,
        0.5371986,  0.52435076, 0.4868823,  0.489054,   1.5220134,  1.5034223,
        1.5273923,  1.4844325,  1.5027945,  1.5141735,  0.52618355, 0.5476362,
        0.513619,   0.5486213,  0.55168426, 0.48179898, 0.56057125, 0.48737723,
        1.5195277,  1.5161197,  1.5151376,  1.5227453,
    ]

    noise_std_param = nn.Parameter(
        torch.tensor(teacher_std_np, device=args_cli.device)
    )

    # create runner from rsl-rl
    runner = CustomNetworkOnPolicyRunner(
        env, 
        agent_cfg.to_dict(), 
        log_dir=log_dir, 
        device=args_cli.device, 
        custom_actor = custom_actor,
        custom_critic = custom_critic,
        noise_std_param = noise_std_param,
    )

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    

    # cfg: RslRlOnPolicyRunnerCfg = RslRlOnPolicyRunnerCfg()
    # cfg.runner.device = device

    # 通常の実行
    # obs, _ = env.reset()
    # timestep = 0
    # dt = 0.02

    # while simulation_app.is_running():
    #     t0 = time.time()
        
    #     # run everything in inference mode
    #     with torch.inference_mode():
    #         try:
    #             actions, _ = policy(obs['policy'])
    #         except:
    #             actions = policy(obs['policy'])
    #         obs, _, _, _, _ = env.step(actions) 
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    simulation_app.close()
