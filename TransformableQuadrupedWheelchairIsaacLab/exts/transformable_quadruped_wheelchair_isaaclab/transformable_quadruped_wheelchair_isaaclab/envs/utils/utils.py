# import gymnasium as gym
# import os
# from pathlib import Path
# import time


# from rsl_rl.runners import OnPolicyRunner

# from isaaclab.managers import SceneEntityCfg
# from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
# from isaaclab.utils.assets import retrieve_file_path
# from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

# import isaaclab_tasks  # noqa: F401
# from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion
# from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.two_modes_env_cfg import (
#     DEFAULT_ACTION_JOINTS_VEL,
#     WALKING_ACTION_JOINTS_POS,
#     WALKING_OFFSET,
#     WALKING_MODE_PREFERRED_ANGLES
# )
# from transformable_quadruped_wheelchair_isaaclab.envs.gym_wrappers.save_trajectory import MultiEnvDistillWrapper

import torch
from isaaclab.assets import Articulation

def load_policy_torch(model_path, device="cuda"):
    policy = torch.jit.load(str(model_path), map_location=device)
    policy.eval()
    return policy

def run_policy_torch(policy, obs_tensor, device="cuda"):
    with torch.no_grad():
        return policy(obs_tensor.to(device))

def post_process_walking_mode():
    pass

def unwrap_env(env):
    """OrderEnforcing などのラッパーを全部剥がして、本丸の ManagerBasedEnv を返す"""
    # Gymnasium の env.unwrapped が使えない場合もあるので自前でやる
    raw = env
    # たとえば: OrderEnforcing → TimeLimit → … → ManagerBasedEnv
    while hasattr(raw, "env"):
        raw = raw.env
    return raw

def get_joint_names(env):
    base_env = unwrap_env(env)
    asset: Articulation = base_env.scene["robot"]
    joint_names = asset.joint_names
    return joint_names

def get_joint_idxs_offsets(env, joints, OFFSETS, joint_names, device):
    idxs = []

    base_env = unwrap_env(env)
    offsets = torch.zeros(size=(1, base_env.action_manager.total_action_dim), device=device)
    for _joint in joints:
        _idx = joint_names.index(_joint)
        idxs.append(_idx)

        offsets[:, _idx] = torch.tensor(OFFSETS[_joint], dtype=torch.float)
    
    return idxs, offsets

def get_joint_idxs_velues(env, joints, OFFSETS, joint_names, device):
    idxs = []

    base_env = unwrap_env(env)
    values = torch.zeros(size=(1, base_env.action_manager.total_action_dim), device=device)
    for _joint in joints:
        _idx = joint_names.index(_joint)
        idxs.append(_idx)

        values[:, _idx] = torch.tensor(OFFSETS[_joint], dtype=torch.float)
    
    return idxs, values

def apply_offset(actions, idxs, offsets, scale):
    actions[:, idxs] = actions[:, idxs] * scale + offsets[:, idxs]
    return actions

def apply_fixed_value(actions, idxs, values):
    actions[:, idxs] = values[:, idxs]
    return actions

def action_idx(env): # actionのインデックスを取得する
    base_env = unwrap_env(env)
    
