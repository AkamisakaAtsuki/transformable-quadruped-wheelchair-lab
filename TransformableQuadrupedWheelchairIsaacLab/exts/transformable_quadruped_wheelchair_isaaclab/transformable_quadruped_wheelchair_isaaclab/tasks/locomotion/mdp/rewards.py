# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

joint_preferred_l2_ids = None
joint_preferred_l2_prefs = None

def joint_preferred_l2(
    env: ManagerBasedRLEnv,
    preferred_joint_angles: Dict[str, float],
    alpha: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    global joint_preferred_l2_ids, joint_preferred_l2_prefs

    if joint_preferred_l2_ids == None or joint_preferred_l2_prefs == None:
        idxs = []
        prefs = []
    
        asset: Articulation = env.scene[asset_cfg.name]
        joint_names = asset.joint_names

        for name, angle in preferred_joint_angles.items():
            if name not in joint_names:
                raise KeyError(f"Joint '{name}' not found in asset '{asset_cfg.name}'.")
            idxs.append(joint_names.index(name))
            prefs.append(angle)

        joint_preferred_l2_ids = torch.tensor(idxs, dtype=torch.long, device=env.device)       # shape: [K]
        joint_preferred_l2_prefs = torch.tensor(prefs, dtype=torch.float32, device=env.device)  # shape: [K]

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos: torch.Tensor = asset.data.joint_pos         # shape: [E, J]
    selected = joint_pos.index_select(1, joint_preferred_l2_ids)  # [num_envs, K]
    diffs_sq = alpha * (selected - joint_preferred_l2_prefs.unsqueeze(0)) ** 2  # broadcast → [num_envs, K]
   
    return diffs_sq.sum(dim=1)

def action_preferred_l2(
    env: ManagerBasedRLEnv,
    preferred_joint_angles: Dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    global action_preferred_l2_ids, action_preferred_l2_prefs

    if action_preferred_l2_ids == None or action_preferred_l2_prefs == None:
        idxs = []
        prefs = []
    
        asset: Articulation = env.scene[asset_cfg.name]
        joint_names = asset.joint_names

        for name, angle in preferred_joint_angles.items():
            if name not in joint_names:
                raise KeyError(f"Joint '{name}' not found in asset '{asset_cfg.name}'.")
            idxs.append(joint_names.index(name))
            prefs.append(angle)

        action_preferred_l2_ids = torch.tensor(idxs, dtype=torch.long, device=env.device)       # shape: [K]
        action_preferred_l2_prefs = torch.tensor(prefs, dtype=torch.float32, device=env.device)  # shape: [K]

    actions: torch.Tensor = env.action_manager.action
    selected = actions.index_select(1, action_preferred_l2_ids)  # [num_envs, K]
    diffs_sq = (selected - action_preferred_l2_prefs.unsqueeze(0)) ** 2  # broadcast → [num_envs, K]
   
    return diffs_sq.sum(dim=1)