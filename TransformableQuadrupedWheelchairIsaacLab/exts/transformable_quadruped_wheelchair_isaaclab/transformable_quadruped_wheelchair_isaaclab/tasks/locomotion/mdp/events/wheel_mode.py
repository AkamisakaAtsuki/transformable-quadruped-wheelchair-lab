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
from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.manage_mode import get_mode, get_stop, print_mode
from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.base import set_joint_angles

import numpy as np
from collections import defaultdict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.models.model_loader import load_policy

INIT_FLAG = True

models_dir = "models"
wheel_mode_model = "wheel_mode.pt"

wheel_mode_policy_action_scale = 0.1
previous_actions_wheel_mode = None
device = 'cuda'

wheel_mode_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), models_dir, wheel_mode_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Loading wheel mode policy from: {wheel_mode_model_path} on device: {device}")

wheel_mode_policy = load_policy(wheel_mode_model_path, device)
print("[INFO] wheel_mode_policy loaded successfully.")

def four_wheel_independent_steering(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_offsets: Dict[str, float] = None,
    joint_pos_to_fix: Dict[str, float] = None,
    front_left_steer: str = None,
    front_right_steer: str = None,
    rear_left_steer: str = None,
    rear_right_steer: str = None,
    front_left_wheel: str = None,
    front_right_wheel: str = None,
    rear_left_wheel: str = None,
    rear_right_wheel: str = None,
    use_learned_model: bool = False, # 学習後のモデルで使用
    learning_model: bool = False, # 学習時に選択
    use_mode_flag: bool = False, # モードフラグを使用するか
    mode_num: float = None, # モードフラグを使用する場合に検出するフラグとなる数字は何かを指定
    debug_mode: bool = False,
):
   
    global previous_actions_wheel_mode, actions_change_mode, device, INIT_FLAG


    _stop = get_stop()
    
    if debug_mode==False and use_mode_flag==True:
        _mode = get_mode()
        valid_env_ids = torch.where(_mode == mode_num)[0]
        
    else:
        valid_env_ids = env_ids
    
    if INIT_FLAG and use_learned_model:

        previous_actions_wheel_mode = torch.zeros((env.num_envs, 12), dtype=torch.float32, device=device)
        print("env.num_envs: {env.num_envs}")
        
    INIT_FLAG = False

    if len(valid_env_ids) == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = asset.joint_names
    
    # ジョイントのマッピング（ステア・ホイール）
    steer_joints = [front_left_steer, front_right_steer, rear_left_steer, rear_right_steer]
    wheel_joints = [front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel]
    steer_joint_indices = torch.tensor([joint_names.index(j) for j in steer_joints], device=env_ids.device)
    wheel_joint_indices = torch.tensor([joint_names.index(j) for j in wheel_joints], device=env_ids.device)

    velocity_commands = mdp.generated_commands(env=env, command_name="base_velocity")[valid_env_ids]
    linear_x = velocity_commands[:, 0]
    linear_y = velocity_commands[:, 1]
    angular_z = velocity_commands[:, 2]

    wheel_base = 0.64
    track_width = 0.6
    offsets = torch.tensor([
        [wheel_base / 2, track_width / 2],
        [wheel_base / 2, -track_width / 2],
        [-wheel_base / 2, track_width / 2],
        [-wheel_base / 2, -track_width / 2]
    ], device=linear_x.device)

    # 座標変換
    velocity = linear_x.unsqueeze(1) - angular_z.unsqueeze(1) * offsets[:, 1]
    lateral_velocity = linear_y.unsqueeze(1) + angular_z.unsqueeze(1) * offsets[:, 0]
    angle = -torch.atan2(lateral_velocity, velocity + 1e-6)
    wheel_speeds = torch.sqrt(velocity ** 2 + lateral_velocity ** 2)
    

    if use_learned_model:
        
        try:
            observations = env.observation_manager.compute()
            current_observations = torch.tensor(
                observations['policy'], dtype=torch.float32, device=device
            ).clone().detach()
        except Exception as e:
            print(f"[Error] Failed to get observations: {e}")
            return

       
        try:
            observations = env.observation_manager.compute()
            current_observations = torch.tensor(
                observations['policy'], dtype=torch.float32, device=device
            ).clone().detach()
            
            if current_observations.shape[1] == 244:
                action_start_idx = sum([3, 3, 3, 3, 16, 28]) 
                action_end_idx = action_start_idx + 1 

                current_observations = torch.cat([
                    current_observations[:, :action_start_idx],  
                    previous_actions_wheel_mode, 
                    current_observations[:, action_end_idx:] 
                ], dim=1)
            else:
                print(f"[Error] Unexpected observation shape: {current_observations.shape}")
                return
        except Exception as e:
            print(f"[Error] Failed to get observations: {e}")
            return

        try:
            with torch.no_grad():
                actions = wheel_mode_policy(current_observations).to(device) 
                actions_steering = actions[:, :8]
                actions_wheel = actions[:, 8:]
        except Exception as e:
            print(e)
            return

        target_joint_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"
        ]
        joint_indices = [joint_names.index(joint) for joint in target_joint_names]
        offsets_tensor = torch.tensor(
            [joint_offsets.get(joint, 0.0) for joint in target_joint_names],
            device=device
        )
        adjusted_actions = actions_steering[valid_env_ids] * wheel_mode_policy_action_scale + offsets_tensor.unsqueeze(0)

        all_joint_indices = torch.cat([steer_joint_indices, torch.tensor(joint_indices, device=device)])
        all_targets = torch.cat([angle, adjusted_actions], dim=1)

        asset.set_joint_position_target(
            target=all_targets, 
            joint_ids=all_joint_indices, 
            env_ids=valid_env_ids
        )
        wheel_speeds_adj = wheel_speeds + actions_wheel[valid_env_ids] * 0.25
        asset.set_joint_velocity_target(
            target=wheel_speeds_adj * 55, 
            joint_ids=wheel_joint_indices, 
            env_ids=valid_env_ids
        )
    elif learning_model:
        try:
            observations = env.observation_manager.compute()
            current_observations = torch.tensor(
                observations['policy'], dtype=torch.float32, device=device
            ).clone().detach()
            
            if current_observations.shape[1] == 255:
                action_start_idx = sum([3, 3, 3, 3, 16, 28, 8])  
                action_end_idx = action_start_idx + 4 

                wheel_vel_adj = current_observations[env_ids, action_start_idx:action_end_idx]
            else:
                print(f"[Error] Unexpected observation shape: {current_observations.shape}")
                return
        except Exception as e:
            print(f"[Error] Failed to get observations: {e}")
            return
        
        asset.set_joint_position_target(
            target=angle, 
            joint_ids=steer_joint_indices,
            env_ids=valid_env_ids
        )

        wheel_speeds_adj = wheel_speeds + wheel_vel_adj * 0.25

        asset.set_joint_velocity_target(
            target=wheel_speeds_adj * 55, 
            joint_ids=wheel_joint_indices,
            env_ids=valid_env_ids
        )

    else:
        
        asset.set_joint_position_target(
            target=angle, 
            joint_ids=steer_joint_indices, 
            env_ids=valid_env_ids
        )
        asset.set_joint_velocity_target(
            target=wheel_speeds * 55, 
            joint_ids=wheel_joint_indices, 
            env_ids=valid_env_ids
        )

    if joint_pos_to_fix:
        set_joint_angles(env, valid_env_ids, asset_cfg, joint_pos_to_fix)