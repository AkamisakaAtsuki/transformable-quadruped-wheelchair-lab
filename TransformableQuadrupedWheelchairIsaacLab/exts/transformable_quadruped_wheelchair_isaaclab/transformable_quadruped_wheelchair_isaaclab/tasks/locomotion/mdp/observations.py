# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.lab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

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
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

# def my_joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     asset: Articulation = env.scene[asset_cfg.name]
#     joint_names = asset.joint_names
#     joint_ids = [joint_names.index(name) for name in asset_cfg.joint_names]

#     return asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]

def my_joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = asset.joint_names

    # 正規表現を使用してjoint_namesをフィルタリング
    filtered_joint_ids = [
        idx for idx, name in enumerate(joint_names)
        if any(re.match(pattern, name) for pattern in asset_cfg.joint_names)
    ]
    
    # 指定した関節の位置情報を計算
    return asset.data.joint_pos[:, filtered_joint_ids] - asset.data.default_joint_pos[:, filtered_joint_ids]

def mode_vector(env: ManagerBasedEnv) -> torch.Tensor:
    """
    get_mode() が None を返すときは全 env をモード0 とみなし、
    それ以外は 0→[1,0], 1→[0,1] のワンホットに変換する。
    """
    batch_size = env.num_envs
    # get_mode が None かどうかをチェック
    modes = get_mode()  # default signature のまま呼ぶなら
    if modes is None:
        # 初期化前は全てモード0 とみなす
        idx = torch.zeros(batch_size, dtype=torch.long, device=env.device)
    else:
        # shape = (batch_size,1) → (batch_size,)
        idx = modes.squeeze(-1).long()
    # 0→[1,0], 1→[0,1] のワンホット
    one_hot = F.one_hot(idx, num_classes=2).float()
    return one_hot.to(env.device)