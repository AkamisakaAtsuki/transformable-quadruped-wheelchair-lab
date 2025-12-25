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

j_pos_idx = None
j_vel_idx = None
j_pos_len = None
j_vel_len = None
j_pos_offset = None

j_16_pos_idx = None
j_16_vel_idx = None
j_16_pos_offset = None

j_wk_pos_idx = None
j_wk_pos_offset = None

j_wh_pos_idx = None
j_wh_vel_idx = None
j_wh_pos_offset = None

_predefined_cache: dict[tuple[str, ...], tuple[torch.Tensor, torch.Tensor]] = {}

j_pre_idx = None
j_pre_target = None

def apply_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_pos_control: List,
    joint_vel_control: List,
    joint_offset: List,
    scale: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):

    global j_pos_idx, j_vel_idx, j_pos_len, j_vel_len, j_pos_offset

    asset: Articulation = env.scene[asset_cfg.name]

    if j_pos_idx == None or j_vel_idx == None or j_pos_offset == None:
        device = env.device  

        j_pos_idx, j_vel_idx = [], []
        j_pos_offset = torch.zeros(size=(1, env.action_manager.total_action_dim), device=device)

        joint_names = asset.joint_names

        for joint_name in joint_pos_control:
            idx = joint_names.index(joint_name)
            j_pos_idx.append(idx)

            j_pos_offset[:, idx] = torch.tensor(joint_offset[joint_name], dtype=torch.float)

        for joint_name in joint_vel_control:
            j_vel_idx.append(joint_names.index(joint_name))

        j_pos_idx = torch.tensor(sorted(j_pos_idx), dtype=torch.long, device=env.device) 
        j_vel_idx = torch.tensor(sorted(j_vel_idx), dtype=torch.long, device=env.device) 
        print(f"j_pos_idx: {j_pos_idx}")
        print(f"j_vel_idx: {j_vel_idx}")
    
    if j_pos_len == None or j_vel_len == None:
        j_pos_len = len(joint_pos_control)
        j_vel_len = len(joint_vel_control)


    action_raw = env.action_manager.action[env_ids]

    asset.set_joint_position_target(
        env_ids=env_ids,
        joint_ids=j_pos_idx,
        target=action_raw[:, j_pos_idx]*scale + j_pos_offset[:, j_pos_idx],
    )
    asset.set_joint_velocity_target(
        env_ids=env_ids,
        joint_ids=j_vel_idx, 
        target=action_raw[:, j_vel_idx] * 100, 
    )

def apply_predefined_joint_angle(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    t_joint_names: list[str],
    t_joint_angles: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
  
    key = tuple(t_joint_names)
    if key not in _predefined_cache:
        asset: Articulation = env.scene[asset_cfg.name]
        joint_names = asset.joint_names
        
        idxs = [joint_names.index(n) for n in t_joint_names]
        idx_tensor = torch.tensor(idxs, dtype=torch.long, device=env.device)
        target_tensor = torch.tensor(t_joint_angles, dtype=torch.float, device=env.device)
        _predefined_cache[key] = (idx_tensor, target_tensor)

    idx_tensor, target_tensor = _predefined_cache[key]
    asset = env.scene[asset_cfg.name]
    asset.set_joint_position_target(
        env_ids=env_ids,
        joint_ids=idx_tensor,
        target=target_tensor,
    )

def apply_action_16_only(  
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_pos_control: List,
    joint_vel_control: List,
    joint_offset: List,
    scale: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):

    global j_16_pos_idx, j_16_vel_idx, j_16_pos_offset

    asset: Articulation = env.scene[asset_cfg.name]

    if j_16_pos_idx == None or j_16_vel_idx == None or j_16_pos_offset == None:
        device = env.device  

        j_16_pos_idx, j_16_vel_idx = [], []
        j_16_pos_offset = torch.zeros(size=(1, 28), device=device)

        joint_names = asset.joint_names

        for joint_name in joint_pos_control:
            idx = joint_names.index(joint_name)
            j_16_pos_idx.append(idx)

            j_16_pos_offset[:, idx] = torch.tensor(joint_offset[joint_name], dtype=torch.float)

        for joint_name in joint_vel_control:
            j_16_vel_idx.append(joint_names.index(joint_name))

        j_16_pos_idx = torch.tensor(sorted(j_16_pos_idx), dtype=torch.long, device=env.device) 
        j_16_vel_idx = torch.tensor(sorted(j_16_vel_idx), dtype=torch.long, device=env.device) 
        print(f"j_16_pos_idx: {j_16_pos_idx}")
        print(f"j_16_vel_idx: {j_16_vel_idx}")

    action_raw = env.action_manager.action[env_ids]

    asset.set_joint_position_target(
        env_ids=env_ids,
        joint_ids=j_16_pos_idx,
        target=action_raw[:, :-4]*scale + j_16_pos_offset[:, j_16_pos_idx],
    )
    asset.set_joint_velocity_target(
        env_ids=env_ids,
        joint_ids=j_16_vel_idx, 
        target=action_raw[:, -4:] * 100, 
    )


def apply_action_wk_mode( 
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_pos_control: List,
    joint_offset: List,
    scale: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):

    global j_wk_pos_idx, j_wk_pos_offset

    asset: Articulation = env.scene[asset_cfg.name]

    if j_wk_pos_idx == None or j_wk_pos_offset == None:
        device = env.device  

        j_wk_pos_idx = []
        j_wk_pos_offset = torch.zeros(size=(1, 28), device=device)

        joint_names = asset.joint_names

        for joint_name in joint_pos_control:
            idx = joint_names.index(joint_name)
            j_wk_pos_idx.append(idx)

            j_wk_pos_offset[:, idx] = torch.tensor(joint_offset[joint_name], dtype=torch.float)

        j_wk_pos_idx = torch.tensor(sorted(j_wk_pos_idx), dtype=torch.long, device=env.device) 
        print(f"j_wk_pos_idx: {j_wk_pos_idx}")


    action_raw = env.action_manager.action[env_ids]

    asset.set_joint_position_target(
        env_ids=env_ids,
        joint_ids=j_wk_pos_idx,
        target=action_raw * scale + j_wk_pos_offset[:, j_wk_pos_idx],
    )
 
def apply_action_wh_mode( 
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_pos_control: List,
    joint_vel_control: List,
    joint_offset: List,
    scale: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):

    global j_wh_pos_idx, j_wh_pos_offset, j_wh_vel_idx

    asset: Articulation = env.scene[asset_cfg.name]

    if j_wh_pos_idx == None or j_wh_pos_offset == None or j_wh_vel_idx == None:
        device = env.device  

        j_wh_pos_idx, j_wh_vel_idx = [], []
        j_wh_pos_offset = torch.zeros(size=(1, 28), device=device)

        joint_names = asset.joint_names

        for joint_name in joint_pos_control:
            idx = joint_names.index(joint_name)
            j_wh_pos_idx.append(idx)

            j_wh_pos_offset[:, idx] = torch.tensor(joint_offset[joint_name], dtype=torch.float)
        
        for joint_name in joint_vel_control:
            j_wh_vel_idx.append(joint_names.index(joint_name))

        j_wh_pos_idx = torch.tensor(sorted(j_wh_pos_idx), dtype=torch.long, device=env.device) 
        j_wh_vel_idx = torch.tensor(sorted(j_wh_vel_idx), dtype=torch.long, device=env.device) 
        print(f"j_wh_pos_idx: {j_wh_pos_idx}")
        print(f"j_wh_vel_idx: {j_wh_vel_idx}")

    action_raw = env.action_manager.action[env_ids]

    asset.set_joint_position_target(
        env_ids=env_ids,
        joint_ids=j_wh_pos_idx,
        target=action_raw[:, :-4] * scale + j_wh_pos_offset[:, j_wh_pos_idx],
    )
    asset.set_joint_velocity_target(
        env_ids=env_ids,
        joint_ids=j_wh_vel_idx, 
        target=action_raw[:, -4:] * 100, 
    )