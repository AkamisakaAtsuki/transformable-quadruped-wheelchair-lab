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

def validate_env_and_joint_ids(env_ids: torch.Tensor, joint_ids: torch.Tensor):
    """env_ids と joint_ids の検証"""
    if env_ids.numel() == 0:
        return False
        # raise ValueError("[Error] env_ids is empty. Check your environment setup.")
    if joint_ids.numel() == 0:
        # raise ValueError("[Error] joint_ids is empty. Check your joint configuration.")
        return False
    
    return True

def illegal_contact_with_collect_data(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    
    global write_buffer

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    
    state = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
    
    sim_context = SimulationContext.instance()
    sim_time = sim_context.current_time  

    for env_id_int in range(len(state)):
        state_ = state[env_id_int]
        
        if state_ == True:
            data = write_buffer[env_id_int]
            file_path = os.path.join(output_dir, f"env_{env_id_int}_vibration.csv")
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write("time,acc_x,acc_y,acc_z\n")
            with open(file_path, "a") as f:
                f.writelines(data)
                f.write("\n")  # 最後に空白行を追加
            write_buffer[env_id_int] = []

    return state

def time_out_with_collect_data(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    global write_buffer

    state = env.episode_length_buf >= env.max_episode_length
    # if state == True:
    sim_context = SimulationContext.instance()
    sim_time = sim_context.current_time  
    # print(f"{sim_time:.4f} + t:{len(state)}")

    for env_id_int in range(len(state)):
        state_ = state[env_id_int]

        if state_ == True:
            data = write_buffer[env_id_int]
            file_path = os.path.join(output_dir, f"env_{env_id_int}_vibration.csv")
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write("time,acc_x,acc_y,acc_z\n")
            with open(file_path, "a") as f:
                f.writelines(data)
                f.write("\n")  # 最後に空白行を追加
            write_buffer[env_id_int] = []

    return state

def replace_observation_column(observation, info, column_name, new_data):
   
    start_idx = 0
    for key, size in info.items():
        if key == column_name:
            break
        start_idx += size
    
    end_idx = start_idx + info[column_name]

    updated_observation = torch.cat([
        observation[:, :start_idx], 
        new_data,                    
        observation[:, end_idx:]    
    ], dim=1)
    
    return updated_observation

def set_joint_angles(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_angles: Dict[str, float] = None,
):
  
    asset: Articulation = env.scene[asset_cfg.name]

    joint_names = asset.joint_names

    # joint_angles が指定されている場合のみ実行
    if joint_angles:
        for joint_name, angle in joint_angles.items():
            if joint_name in joint_names:  # joint_nameがjoint_namesに存在するか確認
                index_of_joint = joint_names.index(joint_name)
                asset.set_joint_position_target(target=angle, joint_ids=index_of_joint, env_ids=env_ids)
            else:
                print(f"Warning: Joint '{joint_name}' not found in joint_names.")