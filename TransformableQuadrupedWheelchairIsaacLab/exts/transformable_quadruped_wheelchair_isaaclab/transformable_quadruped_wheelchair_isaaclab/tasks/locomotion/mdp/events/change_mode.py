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

# thresholds = [-24, -16, -8, 0, 8, 16, 24]    # 変形ポイント x 座標リスト
# tolerances = [1, 1, 1, 1, 1, 1, 1]  # 各閾値の許容誤差
robot = None
teslabot = None
next_idx = None
pause_until = None
paused_flag = None
last_update_time = None
mass_restore_time = None  # モード切替後、元の質量に戻すタイミング
mode_change_mass_tmp = 0.1
rider_original_mass: dict[int, torch.Tensor] = {}  # 各環境ごとに元の質量を保存
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def change_mode_rule(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
# ):
#     global last_update_time

#     current_time = time.time()
#     if current_time - last_update_time >= 10:  # 5秒が経過したかを確認
#         current_mode = get_mode()

#         new_mode = 1 - current_mode

#         update_mode(env_ids, new_mode[env_ids])  # 新しいモードで更新

#         last_update_time = current_time  # 最後の更新時刻を更新
        
#         # print("==========")
#         # # print(env_ids)
#         # print_mode()        
#         # # print(current_mode)

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

    # 現在のモードと停止状態を取得
    current_mode = get_mode()[env_ids]  # 環境ごとのモード取得
    current_stop = get_stop()[env_ids]  # 環境ごとの停止状態取得

    # 各環境ごとに処理
    for i, env_id in enumerate(env_ids):
        # 停止していない環境で5秒経過した場合
        if current_time - last_update_time[env_id] >= 15 and current_stop[i] == 0:
            update_stop([env_id], torch.tensor([1], device=device))  # 停止
            print(f"環境 {env_id}: 停止状態に入りました。")
            last_update_time[env_id] = current_time  # 更新時刻を更新

        # 停止している環境で3秒経過した場合
        elif current_time - last_update_time[env_id] >= 5 and current_stop[i] == 1:
            update_stop([env_id], torch.tensor([0], device=device))  # 動作再開
            new_mode = 1 - current_mode[i]  # モード切替
            update_mode([env_id], torch.tensor([new_mode], device=device))
            print(f"環境 {env_id}: モードを切り替え、動作を再開しました。")
            last_update_time[env_id] = current_time  # 更新時刻を更新

    # モードと停止状態を表示
    # print_mode()
    # print_stop()

def change_mode_rule(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_name: str,
    rider_asset_cfg: SceneEntityCfg,
    thresholds: list,
    tolerances: list,
    pause_duration: float,
):
    """
    閾値に到達したら停止し、停止中に搭乗者(rider_asset_cfg)の質量を1kgに変更する。
    停止時間が経過しモード切替を行った後、3秒後に元の質量に戻す。
    """
    global next_idx, last_update_time, robot, teslabot, pause_until, paused_flag, mass_restore_time, rider_original_mass, mode_change_mass_tmp

    # 初期化
    if next_idx is None:
        n = env.num_envs
        next_idx = torch.zeros(n, dtype=torch.long, device=device)
        last_update_time = torch.zeros(n, dtype=torch.float64, device=device)
        pause_until = torch.zeros(n, dtype=torch.float64, device=device)
        paused_flag = torch.zeros(n, dtype=torch.bool, device=device)
        mass_restore_time = torch.zeros(n, dtype=torch.float64, device=device)
    if robot is None:
        robot = env.scene[asset_name]
    # if teslabot is None:
    #     teslabot = env.scene[rider_asset_cfg.asset_name]

    current_time = time.time()
    positions = robot.data.root_pos_w[:, 1]

    # print("====")
    # print_mode()
    # print_stop()

    # print(robot.data.root_pos_w)
    for env_id in env_ids.tolist():
        idx = next_idx[env_id].item()
        x = positions[env_id].item()

        # Reset 判定
        if x < thresholds[0] - tolerances[0]:
            next_idx[env_id] = 0
            paused_flag[env_id] = False
            pause_until[env_id] = 0.0
            # もし質量復元用タイマーがセットされていたらクリア
            mass_restore_time[env_id] = 0.0
            continue

        if idx >= len(thresholds):
            continue

        target, tol = thresholds[idx], tolerances[idx]

        # print(mass_utils.get_asset_masses(env=env, env_ids=env_ids, asset_cfg=rider_asset_cfg))

        # まだ停止開始していない & 閾値到達
        if not paused_flag[env_id] and abs(x - target) <= tol:
            update_stop([env_id], torch.tensor([1], device=device, dtype=get_stop().dtype))
            pause_until[env_id] = current_time + pause_duration
            paused_flag[env_id] = True
            # print(f"[Env {env_id}] reached {target:.2f}, pausing until {pause_until[env_id]:.2f}")
            # 変形開始時、搭乗者の質量を1kgに変更する
            # まず、現在の質量を保存しておく（後で元に戻すため）
            # rider_mass = mass_utils.get_asset_masses(env=env, env_ids=torch.tensor([env_id], device=device), asset_cfg=rider_asset_cfg)
            # rider_original_mass[env_id] = rider_mass
            # mass_utils.set_asset_mass(env, rider_asset_cfg, torch.tensor([env_id], device=device), new_mass=mode_change_mass_tmp, recompute_inertia=True)
            # print(f"[Env {env_id}] rider mass set to 1kg for transformation.")
            continue

        # 停止中 → モード切替（ただし、ここでは動作再開は行わず、質量復元まで停止状態を維持）
        if paused_flag[env_id] and current_time >= pause_until[env_id]:
            current_mode = get_mode()[env_id].unsqueeze(0)
            new_mode = torch.tensor([1 - current_mode], device=device)
            update_mode([env_id], new_mode)
            # ※ここでは update_stop は解除せず、停止状態を維持する

            root_states = robot.data.default_root_state[env_id].clone()
            # print(root_states)
            # 現在のロボットの位置を取得（仮に最初の3要素が位置情報とする）
            current_pos = robot.data.root_pos_w[env_id].clone()  # root_pos_w[env_id, :3].detach().cpu()  # CPU上で値を取り出す
            current_pos[2] = 0.4  # z 座標だけを 0.5 に変更
            # 現在の向きを取得。ここでは、直接 root_quat_w から取得
            current_quat = robot.data.root_quat_w[env_id].clone()  # shape (4,)
            # quaternion を Euler 角 (roll, pitch, yaw) に変換する。math_utils.quat_to_euler_xyz() があると仮定
            roll, pitch, yaw = math_utils.euler_xyz_from_quat(current_quat.unsqueeze(0))  # (roll, pitch, yaw)
            roll = torch.tensor([0], device=device)
            pitch = torch.tensor([0], device=device)
            # 変換後、再び quaternion に変換
            new_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw) # shape (1,4)
            # print(new_quat.shape)
            # 新しい pose を作成: [position (3), quaternion (4)]
            new_pose = torch.cat([current_pos, new_quat[0]], dim=-1)

            # current_quat = robot.data.root_quat_w[env_id].clone()
            # new_pose = torch.cat([current_pos, current_quat], dim=-1) 
            # force_move_robot を使って、該当環境のロボットを新しい位置に移動する
            # force_move_robot.force_move_robot(env, torch.tensor([env_id], device=device), new_pose.to(device))
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

            # ★ ここから追加：移動直後の状態補正処理 ★
            # 1. ルート速度をゼロにリセットする
            zero_root_vel = torch.zeros_like(robot.data.root_vel_w[env_id])
            # ※ write_root_velocity_to_sim は、通常は batch 形式 ([1, ...]) を要求するので unsqueeze(0) して渡す
            robot.write_root_velocity_to_sim(zero_root_vel.unsqueeze(0), env_ids=torch.tensor([env_id], device=device))
            
            # 2. 関節状態のうち、速度部分もゼロにする
            # ここではデフォルトの関節位置をそのまま使いつつ、関節速度はゼロベクトルに設定
            joint_pos = robot.data.default_joint_pos[env_id].clone()
            # print(joint_pos)
            joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_id])
            robot.write_joint_state_to_sim(joint_pos.unsqueeze(0), joint_vel.unsqueeze(0), env_ids=torch.tensor([env_id], device=device))
            

            # 3. シーン全体のリセット処理も呼び出して、内部状態をすべて更新
            env.scene.reset(torch.tensor([env_id], device=device))
            env.observation_manager.reset(torch.tensor([env_id], device=device))
            env.action_manager.reset(torch.tensor([env_id], device=device))
            env.event_manager.reset(torch.tensor([env_id], device=device))
            env.recorder_manager.reset(torch.tensor([env_id], device=device))
            env.scene.write_data_to_sim()
            env.sim.forward()
            # ★ ここまで追加 ★

            # env.scene.reset(torch.tensor([env_id], device=device))
            # # if "reset" in env.event_manager.available_modes:
            # #     env_step_count = env._sim_step_counter // env.cfg.decimation
            # #     env.event_manager.apply(mode="reset", env_ids=torch.tensor([env_id], device=device), global_env_step_count=env_step_count)
            # env.observation_manager.reset(torch.tensor([env_id], device=device))
            # env.action_manager.reset(torch.tensor([env_id], device=device))
            # env.event_manager.reset(torch.tensor([env_id], device=device))
            # env.recorder_manager.reset(torch.tensor([env_id], device=device))
            # env.scene.write_data_to_sim()
            # env.sim.forward()


            # update_stop([env_id], torch.tensor([0], device=device, dtype=get_stop().dtype))

            next_idx[env_id] += 1
            paused_flag[env_id] = False
            last_update_time[env_id] = current_time
            # print(f"[Env {env_id}] mode switched to {new_mode.item()} at x={x:.2f}")
            # 質量復元のタイミングをセット（3秒後）
            mass_restore_time[env_id] = current_time + 2
            continue

        # 質量復元タイミングが来たら、元の質量に戻し、停止解除して動作再開
        if mass_restore_time[env_id] > 0 and current_time >= mass_restore_time[env_id]:
            update_stop([env_id], torch.tensor([0], device=device, dtype=get_stop().dtype))
            if env_id in rider_original_mass:
                #mass_utils.set_asset_mass(env, rider_asset_cfg, torch.tensor([env_id], device=device), new_mass=rider_original_mass[env_id], recompute_inertia=True)
                print(f"[Env {env_id}] rider mass restored to original.")
                # 動作再開
                mass_restore_time[env_id] = 0.0
                del rider_original_mass[env_id]
                
        # # 停止中 → 停止解除＆モード切替（pause_duration 経過後）
        # if paused_flag[env_id] and current_time >= pause_until[env_id]:
        #     current_mode = get_mode()[env_id].unsqueeze(0)
        #     new_mode = torch.tensor([1 - current_mode], device=device)
        #     update_mode([env_id], new_mode)
        #     update_stop([env_id], torch.tensor([0], device=device, dtype=get_stop().dtype))
        #     next_idx[env_id] += 1
        #     paused_flag[env_id] = False
        #     last_update_time[env_id] = current_time
        #     print(f"[Env {env_id}] switching mode → {new_mode.item()} at x={x:.2f}")
        #     # モード切替後、3秒後に元の質量へ戻すタイミングをセット
        #     mass_restore_time[env_id] = current_time + 3.0
        #     continue

        # # モード切替後、3秒経過していたら元の質量に戻す
        # if mass_restore_time[env_id] > 0 and current_time >= mass_restore_time[env_id]:
        #     if env_id in rider_original_mass:
        #         mass_utils.set_asset_mass(env, rider_asset_cfg, torch.tensor([env_id], device=device), new_mass=rider_original_mass[env_id], recompute_inertia=True)
        #         print(f"[Env {env_id}] rider mass restored to original.")
        #         mass_restore_time[env_id] = 0.0
        #         del rider_original_mass[env_id]