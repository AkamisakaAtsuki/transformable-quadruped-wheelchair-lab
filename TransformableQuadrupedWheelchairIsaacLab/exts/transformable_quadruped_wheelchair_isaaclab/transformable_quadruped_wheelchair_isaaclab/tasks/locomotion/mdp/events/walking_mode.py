from __future__ import annotations

import os
import time
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path
from typing import TYPE_CHECKING, Dict

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import sample_uniform
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

import numpy as np
from collections import defaultdict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.manage_mode import get_mode, get_stop
from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.models.model_loader import load_policy
from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.base import set_joint_angles, validate_env_and_joint_ids

INIT_FLAG = True

models_dir = "models"
walking_mode_model = "walking_mode.pt"

walking_mode_policy_action_scale = 0.1
previous_actions_walking_mode = None
device = 'cuda'

walking_mode_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), models_dir, walking_mode_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Loading walking mode policy from: {walking_mode_model_path} on device: {device}")

walking_mode_policy = load_policy(walking_mode_model_path, device)
print("[INFO] walking_mode_policy loaded successfully.")

def apply_walking_policy(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_pos_to_fix: Dict[str, float] = None,
    observation_info: Dict[str, float] = None,
    use_mode_flag: bool = False, # モードフラグを使用するか
    mode_num: float = None, # モードフラグを使用する場合に検出するフラグとなる数字は何かを指定
    debug_mode: bool = False,
):

    s_time = time.time()

    global previous_actions_walking_mode, actions_change_mode, walking_mode_policy, device, INIT_FLAG

    _stop = get_stop()
    if debug_mode==False and use_mode_flag==True:
        _mode = get_mode()
        valid_env_ids = torch.where(_mode == mode_num)[0]
       
    else:
        valid_env_ids = env_ids
    
    if INIT_FLAG:
        previous_actions_walking_mode = torch.zeros((env.num_envs, 12), dtype=torch.float32, device=device)
        print(f"env.num_envs: {env.num_envs}")
        
    INIT_FLAG = False

   
    
    if len(valid_env_ids) == 0:
        return


    try:
        observations = env.observation_manager.compute()
        current_observations = torch.tensor(
            observations['policy'], dtype=torch.float32, device=device
        ).clone().detach()
      
    except Exception as e:
        print(f"[Error] Failed to get observations: {e}")
        return

    try:
        if current_observations.shape[1] == 244:
            # actions (Index 6) を置き換え
            action_start_idx = sum([3, 3, 3, 3, 16, 28])  # actions の開始インデックス (上記表から計算)
            action_end_idx = action_start_idx + 1  # 現在の actions は (1,)

            # actions 部分を上書きせず、新しいカラムを挿入
            current_observations = torch.cat([
                current_observations[:, :action_start_idx],  # actions手前まで
                previous_actions_walking_mode,  # (12,) の actions
                current_observations[:, action_end_idx:]  # actions 以降
            ], dim=1)
            
        else:
            print(f"[Error] Unexpected observation shape: {current_observations.shape}")
            return
    except Exception as e:
        print(f"[Error] Failed to adjust observations: {e}")
        return

    try:
        with torch.no_grad():
            actions = walking_mode_policy(current_observations).to(device) 
            # print(f"[INFO] Actions from policy: {actions.shape}")
    except Exception as e:
        print(f"[Error] Failed to infer actions from policy: {e}")
        return

    previous_actions_walking_mode[valid_env_ids] = actions[valid_env_ids].clone().detach()
    
    joint_offsets = {
        "FL_hip_joint": 0.1,
        "FR_hip_joint": -0.1,
        "RL_hip_joint": 0.1,
        "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    }

    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = asset.joint_names

    target_joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
    ]

    for i, joint_name in enumerate(target_joint_names):
        if joint_name in joint_names:
            joint_index = joint_names.index(joint_name)
            
            offset = joint_offsets.get(joint_name, 0.0)

            if validate_env_and_joint_ids(valid_env_ids, torch.tensor([joint_index])):
                
                adjusted_action = actions[valid_env_ids, i].unsqueeze(-1) * walking_mode_policy_action_scale + offset
                asset.set_joint_position_target(
                    target=adjusted_action,
                    joint_ids=joint_index,
                    env_ids=valid_env_ids
                )
        
    if joint_pos_to_fix:
        set_joint_angles(env, valid_env_ids, asset_cfg, joint_pos_to_fix)

    e_time = time.time()

    if debug_mode:
        print(f"[INFO] (apply_learned_policy) Time Delta: {e_time - s_time}")

def walking_mode_manager(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  
    use_mode_flag: bool = False,
    mode_num: float = None,
    debug_mode: bool = False,
):

    s_time = time.time()

    global previous_actions_walking_mode, actions_change_mode, walking_mode_policy, device, INIT_FLAG

    # print(get_mode())

    _stop = get_stop()
    if debug_mode==False and use_mode_flag==True:
        _mode = get_mode()
        valid_env_ids = torch.where(_mode == mode_num)[0]
       
    else:
        valid_env_ids = env_ids
    
    if INIT_FLAG:
        previous_actions_walking_mode = torch.zeros((env.num_envs, 12), dtype=torch.float32, device=device)
        print(f"env.num_envs: {env.num_envs}")
        
    INIT_FLAG = False

    if len(valid_env_ids) == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]

    action_raw = env.action_manager.action[valid_env_ids]
    asset.set_joint_position_target(
        target=action_raw * walking_mode_policy_action_scale,
        env_ids=valid_env_ids
    )
