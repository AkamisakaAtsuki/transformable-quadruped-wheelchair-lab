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

import gymnasium as gym
import os
from pathlib import Path
import time
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

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

def main(policy):
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
    env_cfg.WALKING_MODE = True

    env = gym.make(
        args_cli.task, 
        cfg=env_cfg, 
        render_mode="rgb_array" if args_cli.video else None
    )

    joint_names = get_joint_names(env)
    
    walking_action_joint_pos_idxs, offsets = get_joint_idxs_offsets( 
        env, 
        WALKING_ACTION_JOINTS_POS, 
        WALKING_OFFSET,
        joint_names,
        device
    ) 

    walking_fixed_joint_idxs, values = get_joint_idxs_velues(
        env,
        ['LFUpper2_joint', 'RFUpper2_joint', 'LRUpper2_joint', 'RRUpper2_joint'],
        WALKING_OFFSET,
        joint_names,
        device
    ) 

    obs, _ = env.reset()
    timestep = 0
    dt = 0.02

    distilled_model = True

    while simulation_app.is_running():
        t0 = time.time()
        
        with torch.inference_mode():
            if distilled_model == True:
                try:
                    actions, _ = policy(obs['policy'])
                except:
                    actions = policy(obs['policy'])
                obs, _, _, _, _ = env.step(actions) 
            else:
                _actions = policy(obs['policy'])
                actions = _actions.clone()    

                scale = 0.1
                actions = apply_offset(actions, walking_action_joint_pos_idxs, offsets, scale)
                actions = apply_fixed_value(actions, walking_fixed_joint_idxs, values)

                obs, _, _, _, _ = env.step(actions) 

                action_s_idx, action_e_idx = 68, 96
                obs['policy'][:, action_s_idx: action_e_idx] = _actions

    env.close()


if __name__ == "__main__":
    SCRIPT_FILE = Path(__file__).resolve()
    SCRIPT_DIR  = SCRIPT_FILE.parent
    # policy = load_policy_torch(f"{SCRIPT_DIR}\models\distill_policy_jit.pt")
    # policy = load_policy_torch(f"{SCRIPT_DIR}\models\distilled_policy_kl_std_jit2.pt")
    policy = load_policy_torch(f"{SCRIPT_DIR}\models\distilled_policy.pt")
    # policy = load_policy_torch(f"{SCRIPT_DIR}\models\walking_mode.pt")

    # print(policy._c._method_names())        # e.g. ['forward']
    # print(policy._c._get_method('forward').schema)  # 確認用
    # # 3. バッファ（obs_mean / obs_std）を取り出す
    # print("obs_mean:", policy.obs_mean)      # ちゃんとtraining時の平均が入っているか？
    # print("obs_std: ", policy.obs_std)       # training時のstdが入っているか？
    
    # TorchScript モジュールが持つメソッド名一覧
    # method_names = policy._c._method_names()
    # print("Available methods:", method_names)

    # # 各メソッドのスキーマ（引数と戻り値）を表示
    # for name in method_names:
    #     m = policy._c._get_method(name)
    #     schema = m.schema    # ← 呼び出しではなく属性として取得します
    #     print(f"\n--- Method: {schema.name} ---")
    #     print("Signature:", schema)  # 全体を文字列で表示
    #     print(" Arguments:")
    #     for arg in schema.arguments:
    #         # .name, .type で引数名と型が取れます
    #         print(f"  • {arg.name}: {arg.type}")
    #     print(" Returns:")
    #     for ret in schema.returns:
    #         print(f"  • {ret.name}: {ret.type}")

    main(policy)
    # close sim app
    simulation_app.close()
