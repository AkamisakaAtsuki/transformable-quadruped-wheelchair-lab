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
import isaaclab.utils.math as math_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

import numpy as np
from collections import defaultdict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.manage_mode import get_mode, get_stop, print_mode, print_stop, update_mode, update_stop
from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.models.model_loader import load_policy
from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.base import set_joint_angles, validate_env_and_joint_ids
from transformable_quadruped_wheelchair_isaaclab.utils import mass_utils
from transformable_quadruped_wheelchair_isaaclab.utils import force_move_robot
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

robot = None
teslabot = None
next_idx = None
pause_until = None
paused_flag = None
last_update_time = None
mass_restore_time = None 
mode_change_mass_tmp = 0.1
rider_original_mass: dict[int, torch.Tensor] = {} 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def change_mode_rule_before(
    env: ManagerBasedEnv, 
    env_ids: torch.Tensor,
):
    global last_update_time, device
    
    current_time = time.time()

    if last_update_time == None:
        last_update_time = torch.zeros(env.num_envs, dtype=torch.float64)
        for i in range(len(env_ids)):
            last_update_time[i] = current_time

    current_mode = get_mode()[env_ids] 
    current_stop = get_stop()[env_ids] 

    for i, env_id in enumerate(env_ids):
        if current_time - last_update_time[env_id] >= 15 and current_stop[i] == 0:
            update_stop([env_id], torch.tensor([1], device=device))  # 停止
            print(f"環境 {env_id}: 停止状態に入りました。")
            last_update_time[env_id] = current_time  # 更新時刻を更新

        elif current_time - last_update_time[env_id] >= 5 and current_stop[i] == 1:
            update_stop([env_id], torch.tensor([0], device=device))  # 動作再開
            new_mode = 1 - current_mode[i]  # モード切替
            update_mode([env_id], torch.tensor([new_mode], device=device))
            print(f"環境 {env_id}: モードを切り替え、動作を再開しました。")
            last_update_time[env_id] = current_time  # 更新時刻を更新

def change_mode_rule(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_name: str,
    rider_asset_cfg: SceneEntityCfg,
    thresholds: list,
    tolerances: list,
    pause_duration: float,
):
    global next_idx, last_update_time, robot, teslabot, pause_until, paused_flag, mass_restore_time, rider_original_mass, mode_change_mass_tmp

    if next_idx is None:
        n = env.num_envs
        next_idx = torch.zeros(n, dtype=torch.long, device=device)
        last_update_time = torch.zeros(n, dtype=torch.float64, device=device)
        pause_until = torch.zeros(n, dtype=torch.float64, device=device)
        paused_flag = torch.zeros(n, dtype=torch.bool, device=device)
        mass_restore_time = torch.zeros(n, dtype=torch.float64, device=device)
    if robot is None:
        robot = env.scene[asset_name]
    current_time = time.time()
    positions = robot.data.root_pos_w[:, 1]

    for env_id in env_ids.tolist():
        idx = next_idx[env_id].item()
        x = positions[env_id].item()

        if x < thresholds[0] - tolerances[0]:
            next_idx[env_id] = 0
            paused_flag[env_id] = False
            pause_until[env_id] = 0.0
            
            mass_restore_time[env_id] = 0.0
            continue

        if idx >= len(thresholds):
            continue

        target, tol = thresholds[idx], tolerances[idx]

        if not paused_flag[env_id] and abs(x - target) <= tol:
            update_stop([env_id], torch.tensor([1], device=device, dtype=get_stop().dtype))
            pause_until[env_id] = current_time + pause_duration
            paused_flag[env_id] = True
            
            continue

        if paused_flag[env_id] and current_time >= pause_until[env_id]:
            current_mode = get_mode()[env_id].unsqueeze(0)
            new_mode = torch.tensor([1 - current_mode], device=device)
            update_mode([env_id], new_mode)
            
            root_states = robot.data.default_root_state[env_id].clone()
           
            current_pos = robot.data.root_pos_w[env_id].clone()  
            current_pos[2] = 0.4 
            
            current_quat = robot.data.root_quat_w[env_id].clone() 
            
            roll, pitch, yaw = math_utils.euler_xyz_from_quat(current_quat.unsqueeze(0))
            roll = torch.tensor([0], device=device)
            pitch = torch.tensor([0], device=device)
           
            new_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
           
            new_pose = torch.cat([current_pos, new_quat[0]], dim=-1)

            force_move_robot.force_move_robot_uniform(
                env, 
                torch.tensor([env_id], device=device), 
                new_pose.to(device),
                {
                    "x": (-0.5, 0.5), 
                    "y": (-0.5, 0.5), 
                    "yaw": (0, 0)
                },
                {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            )
            force_move_robot.force_move_robot_uniform(
                env, 
                torch.tensor([env_id], device=device), 
                new_pose.to(device),
                {
                    "x": (0, 0), 
                    "y": (0, 0), 
                    "yaw": (0, 0)
                },
                {
                    "x": (0, 0),
                    "y": (0, 0),
                    "z": (0, 0),
                    "roll": (0, 0),
                    "pitch": (0, 0),
                    "yaw": (0, 0),
                },
                rider_asset_cfg,
            )
            mdp.reset_joints_by_scale(
                env, 
                torch.tensor([env_id], device=device),
                (0.0, 0.0),
                (0.0, 0.0)
            )

           
            zero_root_vel = torch.zeros_like(robot.data.root_vel_w[env_id])
           
            robot.write_root_velocity_to_sim(zero_root_vel.unsqueeze(0), env_ids=torch.tensor([env_id], device=device))
          
            joint_pos = robot.data.default_joint_pos[env_id].clone()
       
            joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_id])
            robot.write_joint_state_to_sim(joint_pos.unsqueeze(0), joint_vel.unsqueeze(0), env_ids=torch.tensor([env_id], device=device))
   
            env.scene.reset(torch.tensor([env_id], device=device))
            env.observation_manager.reset(torch.tensor([env_id], device=device))
            env.action_manager.reset(torch.tensor([env_id], device=device))
            env.event_manager.reset(torch.tensor([env_id], device=device))
            env.recorder_manager.reset(torch.tensor([env_id], device=device))
            env.scene.write_data_to_sim()
            env.sim.forward()
       
            next_idx[env_id] += 1
            paused_flag[env_id] = False
            last_update_time[env_id] = current_time
            
            mass_restore_time[env_id] = current_time + 2
            continue

        if mass_restore_time[env_id] > 0 and current_time >= mass_restore_time[env_id]:
            update_stop([env_id], torch.tensor([0], device=device, dtype=get_stop().dtype))
            if env_id in rider_original_mass:
                print(f"[Env {env_id}] rider mass restored to original.")
                mass_restore_time[env_id] = 0.0
                del rider_original_mass[env_id]