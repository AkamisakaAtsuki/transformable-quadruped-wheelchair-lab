# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.manage_mode import get_mode, get_stop

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

def my_joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = asset.joint_names

    filtered_joint_ids = [
        idx for idx, name in enumerate(joint_names)
        if any(re.match(pattern, name) for pattern in asset_cfg.joint_names)
    ]
   
    return asset.data.joint_pos[:, filtered_joint_ids] - asset.data.default_joint_pos[:, filtered_joint_ids]

def mode_vector(env: ManagerBasedEnv) -> torch.Tensor:
   
    batch_size = env.num_envs
   
    modes = get_mode()  
    if modes is None:
       
        idx = torch.zeros(batch_size, dtype=torch.long, device=env.device)
    else:
      
        idx = modes.squeeze(-1).long()
    one_hot = F.one_hot(idx, num_classes=2).float()
    return one_hot.to(env.device)