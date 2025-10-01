# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
# from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# action_preferred_l2のグローバル変数
joint_preferred_l2_ids = None
joint_preferred_l2_prefs = None

# ──────────────────────────────────────────────────────────────
# L2 penalty on joint positions
# ──────────────────────────────────────────────────────────────
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
    
    # # print(diffs_sq.sum(dim=1))
    # actions: torch.Tensor = env.action_manager.action
    # joint_names = asset.joint_names
    # jv = joint_pos[:, joint_names.index("RR_hip_joint")]
    # av = actions[:, joint_names.index("RR_hip_joint")] * 0.1
    # print(f"j: {jv} a: {av}")
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
    # print(diffs_sq.sum(dim=1))
    return diffs_sq.sum(dim=1)


# def feet_air_time(
#     env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
# ) -> torch.Tensor:
#     """Reward long steps taken by the feet using L2-kernel.

#     This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
#     that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
#     the time for which the feet are in the air.

#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     """
#     # extract the used quantities (to enable type-hinting)
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
#     last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
#     reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     return reward


# def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Reward long steps taken by the feet for bipeds.

#     This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
#     a time in the air.

#     If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
#     """
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     # compute the reward
#     air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
#     contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
#     in_contact = contact_time > 0.0
#     in_mode_time = torch.where(in_contact, contact_time, air_time)
#     single_stance = torch.sum(in_contact.int(), dim=1) == 1
#     reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
#     reward = torch.clamp(reward, max=threshold)
#     # no reward for zero command
#     reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
#     return reward


# def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     # Penalize feet sliding
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
#     asset = env.scene[asset_cfg.name]
#     body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
#     reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
#     return reward


# def track_lin_vel_xy_yaw_frame_exp(
#     env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
#     lin_vel_error = torch.sum(
#         torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
#     )
#     return torch.exp(-lin_vel_error / std**2)


# def track_ang_vel_z_world_exp(
#     env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset = env.scene[asset_cfg.name]
#     ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
#     return torch.exp(-ang_vel_error / std**2)

