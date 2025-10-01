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

models_dir = "models"
walking_mode_model = "walking_mode.pt"
wheel_mode_model = "wheel_mode.pt"
change_mode_model = "change_mode.pt"
walking_mode_model_path = os.path.join(os.path.dirname(__file__), models_dir, walking_mode_model)
wheel_mode_model_path = os.path.join(os.path.dirname(__file__), models_dir, wheel_mode_model)
change_mode_model_path = os.path.join(os.path.dirname(__file__), models_dir, change_mode_model)

# CUDAが利用可能か確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(walking_mode_model_path):
    print(f"[Error] Model file not found at {walking_mode_model_path}")
elif not os.path.exists(wheel_mode_model_path):
    print(f"[Error] Model file not found at {wheel_mode_model_path}")
elif not os.path.exists(change_mode_model_path):
    print(f"[Error] Model file not found at {change_mode_model_path}")
else:
    try:
        # JITモデルをロード
        walking_mode_policy = torch.jit.load(walking_mode_model_path, map_location=device)
        walking_mode_policy.to(device)  # モデルをCUDAデバイスに移動
        walking_mode_policy.eval()
        print("[INFO] Walking Mode Policy loaded successfully as JIT model.")

        wheel_mode_policy = torch.jit.load(wheel_mode_model_path, map_location=device)
        wheel_mode_policy.to(device)  # モデルをCUDAデバイスに移動
        wheel_mode_policy.eval()
        print("[INFO] Wheel Mode Policy loaded successfully as JIT model.")

        change_mode_policy = torch.jit.load(change_mode_model_path, map_location=device)
        change_mode_policy.to(device)  # モデルをCUDAデバイスに移動
        change_mode_policy.eval()
        print("[INFO] Change Mode Policy loaded successfully as JIT model.")

    except Exception as e:
        print(f"[Error] Failed to load JIT model: {e}")   

# 必要な変数の初期化
len_env_ids = 64
walking_mode_policy_action_scale = 0.1
previous_actions_walking_mode = torch.zeros((len_env_ids, 12), dtype=torch.float32, device=device)
wheel_mode_policy_action_scale = 0.1
previous_actions_wheel_mode = torch.zeros((len_env_ids, 12), dtype=torch.float32, device=device)
actions_change_mode = torch.ones((len_env_ids, 1), dtype=torch.float32, device=device)

count = 0

def validate_env_and_joint_ids(env_ids: torch.Tensor, joint_ids: torch.Tensor):
    """env_ids と joint_ids の検証"""
    if env_ids.numel() == 0:
        return False
        # raise ValueError("[Error] env_ids is empty. Check your environment setup.")
    if joint_ids.numel() == 0:
        # raise ValueError("[Error] joint_ids is empty. Check your joint configuration.")
        return False
    
    return True

# 保存ディレクトリの設定
output_dir = "saved_data/output_dir"  # 出力パスを指定
os.makedirs(output_dir, exist_ok=True)

# バッチ書き込み用のバッファ
write_buffer = defaultdict(list)
BATCH_SIZE = 100  # 一度に書き込むデータ数

# 揺れデータを収集しCSVに保存
def collect_vibration_data(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    TESLABOTの揺れデータを収集し、ISO 2631評価に向けて時系列データを保存。
    """
    global write_buffer

    # IMUデータ取得 (X, Y, Z軸の加速度)
    observations = env.observation_manager.compute()
    base_acc = observations['policy'][:, :3].cpu().numpy()  # 最初の3要素が加速度 (X, Y, Z)

    # シミュレーション時刻を取得
    sim_context = SimulationContext.instance()
    sim_time = sim_context.current_time

    for idx, env_id in enumerate(env_ids):
        env_id_int = int(env_id.item())
        acc_data = base_acc[idx]  # 各環境IDの加速度データ (X, Y, Z)

        # 加速度データを文字列化して保存フォーマットに変換
        acc_str = ";".join(f"{val:.4f}" for val in acc_data)
        write_buffer[env_id_int].append(f"{sim_time:.4f},{acc_str}\n")

        # バッファが一定数を超えたらファイルに書き込み
        if len(write_buffer[env_id_int]) >= BATCH_SIZE:
            file_path = os.path.join(output_dir, f"env_{env_id_int}_vibration.csv")
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write("time,acc_x,acc_y,acc_z\n")
            with open(file_path, "a") as f:
                f.writelines(write_buffer[env_id_int])
            write_buffer[env_id_int] = []

    print(f"[INFO] Vibration data collected and saved for {len(env_ids)} environments.")

def illegal_contact_with_collect_data(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    global write_buffer

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    state = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
    # if state == True:
    sim_context = SimulationContext.instance()
    sim_time = sim_context.current_time  
    # print(f"{sim_time:.4f} + l:{len(state)}")

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
    """
    観測データ内の指定されたカラムを新しいデータで置き換える関数。

    Args:
        observation (torch.Tensor): 観測データのテンソル。
        info (dict): 各観測データのカラム名とそのサイズを保持する辞書。
        column_name (str): 置き換えるカラムの名前。
        new_data (torch.Tensor): 置き換える新しいデータ。

    Returns:
        torch.Tensor: 置き換え後の観測データ。
    """
    # カラムの開始インデックスを計算
    start_idx = 0
    for key, size in info.items():
        if key == column_name:
            break
        start_idx += size
    
    end_idx = start_idx + info[column_name]

    # print(f"start_idx: {start_idx}, end_idx: {end_idx}")

    # 観測データの置き換え
    updated_observation = torch.cat([
        observation[:, :start_idx],  # 置き換え前部分
        new_data,                    # 新しいデータ
        observation[:, end_idx:]     # 置き換え後部分
    ], dim=1)
    
    return updated_observation

def set_joint_angles(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_angles: Dict[str, float] = None,
):
    """
    階段をのぼるときの椅子をロボットの角度に応じて動的に変更するためのプログラム
    
    Parameters:
        env (ManagerBasedEnv): 環境オブジェクト。
        env_ids (torch.Tensor): 環境IDのテンソル。
        asset_cfg (SceneEntityCfg): アセット設定。
        joint_pos_to_fix (Dict[str, float]): 固定するジョイント名と角度の辞書。
    """
    # extract the used quantities (to enable type-hinting)
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

def four_wheel_independent_steering(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_pos_to_fix: Dict[str, float] = None,
    front_left_steer: str = None,
    front_right_steer: str = None,
    rear_left_steer: str = None,
    rear_right_steer: str = None,
    front_left_wheel: str = None,
    front_right_wheel: str = None,
    rear_left_wheel: str = None,
    rear_right_wheel: str = None,
    debug_mode: bool = False,
):
    """
    四輪独立操舵の制御情報を取得し、各ステアとホイールの角度をバッチ処理で管理する関数。

    Args:
        env (ManagerBasedEnv): 環境インスタンス。
        env_ids (torch.Tensor): 環境IDのリスト。
        *_steer (str): 各ステアジョイント名。
        *_wheel (str): 各ホイールジョイント名。
    """

    global previous_actions_wheel_mode
    global actions_change_mode

    # タイマー開始
    # start_time = time.time()

    # try:
    #     last_action_value = mdp.last_action(env)
    # except:
    #     pass

    # 条件に一致する環境IDを取得
    valid_env_mask = actions_change_mode.squeeze(-1) < 0.5  # 値が < 0 の環境を抽出
    if debug_mode:
        valid_env_ids = env_ids # debug_modeがTrueだったらすべての環境に対して実行
    else:
        valid_env_ids = env_ids[valid_env_mask.nonzero(as_tuple=True)[0]]

    if valid_env_ids.numel() == 0:
        print("[INFO] No environments met the condition for four_wheel_independent_steering.")
        return

    # print(f"len valid_env_ids: {len(valid_env_ids)}")

    # 観測データを取得
    try:
        observations = env.observation_manager.compute()
        current_observations = torch.tensor(
            observations['policy'], dtype=torch.float32, device=device
        ).clone().detach()
        # print(f"[INFO] Observations shape: {current_observations.shape}")
        # 観測値の絶対値が100を超える場合の処理

    except Exception as e:
        print(f"[Error] Failed to get observations: {e}")
        return


    # print(f"[INFO] Observations: {current_observations}")
    # threshold = 100  # 許容範囲の絶対値の閾値

    # # 異常値が含まれているかチェック
    # if torch.any(torch.abs(current_observations) > threshold):
    #     print("[Error] current_observations contains values exceeding the threshold!")
    #     print("Abnormal observations detected. Stopping the simulation for debugging.")
    #     return  # 処理を終了
 
    # num_nan = torch.sum(torch.isnan(current_observations))
    # if num_nan > 0:
    #     print(f"Detected NaN ! ({num_nan})")
    #     return

    # 観測データを調整
    try:
        if current_observations.shape[1] == 244:
            # actions (Index 6) を置き換え
            action_start_idx = sum([3, 3, 3, 3, 16, 28])  # actions の開始インデックス (上記表から計算)
            action_end_idx = action_start_idx + 1  # 現在の actions は (1,)

            # rint(current_observations[:,action_start_idx:action_end_idx])

            # actions 部分を上書きせず、新しいカラムを挿入
            current_observations = torch.cat([
                current_observations[:, :action_start_idx],  # actions手前まで
                previous_actions_wheel_mode,  # (8,) の actions
                current_observations[:, action_end_idx:]  # actions 以降
            ], dim=1)

            # print(f"current_observations[b]: {current_observations[0]}")
            # print(f"[INFO] Actions replaced without affecting height_scan. New shape: {current_observations.shape}")
        else:
            print(f"[Error] Unexpected observation shape: {current_observations.shape}")
            return
    except Exception as e:
        print(f"[Error] Failed to adjust observations: {e}")
        return

    # 方策からアクションを取得
    try:
        with torch.no_grad():
            actions = wheel_mode_policy(current_observations).to(device) 
            actions_steering = actions[:, :8]
            actions_wheel = actions[:, 8:]
            # print(f"[INFO] Actions from policy: {actions.shape}")
    except Exception as e:
        print(f"[Error] Failed to infer actions from policy: {e}")
        return

    # アクションを記録（次回の入力に使用）
    # previous_actions_wheel_mode = actions.clone().detach() # ここをコメントアウトしたら上手く動くようになった。本来は正しくないけど、実際は基本0に近い値だから問題ないだろうと思われる

    # 各ジョイントのオフセットを直接指定（車輪移動モードの時に使用したもの）
    joint_offsets = {
        'FL_hip_joint': 0.1,
        'FR_hip_joint': -0.1,
        'RL_hip_joint': 0.1,
        'RR_hip_joint': -0.1, 
        'FL_thigh_joint': 0.0,
        'FR_thigh_joint': 0.0,
        'RL_thigh_joint': 0.0,
        'RR_thigh_joint': 0.0,
    }

    # アセット取得
    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = asset.joint_names
  
    # ジョイントへのマッピング
    target_joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"
    ]

    # ジョイントインデックスのリストを取得
    joint_indices = [joint_names.index(joint_name) for joint_name in target_joint_names]

    # 各ジョイントのオフセットを取得
    offsets = torch.tensor(
        [joint_offsets.get(joint_name, 0.0) for joint_name in target_joint_names],
        device=device
    )

    # アクションの調整をベクトル化
    adjusted_actions = actions_steering[valid_env_ids] * wheel_mode_policy_action_scale + offsets.unsqueeze(0)
    # adjusted_actions = actions[valid_env_ids] * wheel_mode_policy_action_scale + offsets.unsqueeze(0)

    # 有効なジョイントインデックスとアクションを一括で適用
    # asset.set_joint_position_target(
    #     target=adjusted_actions,
    #     joint_ids=torch.tensor(joint_indices, device=device),
    #     env_ids=valid_env_ids
    # )

    # for i, joint_name in enumerate(target_joint_names):
    #     joint_index = joint_names.index(joint_name)
        
    #     # ✅ 事前定義されたオフセットを適用
    #     offset = joint_offsets.get(joint_name, 0.0)

    #     if validate_env_and_joint_ids(valid_env_ids, torch.tensor([joint_index])):  # 起動初期は数が合わないことがあるのでチェック（もっとちゃんとしたやり方があるはずだが.tmpなやり方です）
    #         # ✅ オフセットを考慮したアクションを適用
    #         adjusted_action = actions[valid_env_ids, i].unsqueeze(-1) * wheel_mode_policy_action_scale + offset
    #         asset.set_joint_position_target(
    #             target=adjusted_action,
    #             joint_ids=joint_index,
    #             env_ids=valid_env_ids
    #         )

    # 必要なジョイント名をリスト化
    steer_joints = [front_left_steer, front_right_steer, rear_left_steer, rear_right_steer]
    wheel_joints = [front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel]

    # 各ジョイントのインデックスをバッチで取得
    steer_joint_indices = torch.tensor([joint_names.index(joint) for joint in steer_joints], device=env_ids.device)
    wheel_joint_indices = torch.tensor([joint_names.index(joint) for joint in wheel_joints], device=env_ids.device)

    # 観測データ取得 (ロボット基準座標系)
    velocity_commands = mdp.generated_commands(env=env, command_name="base_velocity")
    velocity_commands = velocity_commands[valid_env_ids]

    # ロボット基準座標系
    linear_x = velocity_commands[:, 0]  # 前後移動速度
    linear_y = velocity_commands[:, 1]  # 左右移動速度
    angular_z = velocity_commands[:, 2]  # 旋回速度

    # 車両パラメータ
    wheel_base = 0.64  # 前後輪間の距離
    track_width = 0.6  # 左右輪間の距離

    # 各ホイールのオフセット（X: 前後, Y: 左右）
    offsets = torch.tensor([
        [wheel_base / 2, track_width / 2],  # front_left
        [wheel_base / 2, -track_width / 2],  # front_right
        [-wheel_base / 2, track_width / 2],  # rear_left
        [-wheel_base / 2, -track_width / 2]  # rear_right
    ], device=linear_x.device)

    # 🚗 **座標変換**  
    # - 前後移動 (x方向) → 正しいホイール速度  
    # - 左右移動 (y方向) → ステア角度  
    # - 旋回 (angular_z) → 各ホイールへ反映  

    velocity = linear_x.unsqueeze(1) - angular_z.unsqueeze(1) * offsets[:, 1]
    lateral_velocity = linear_y.unsqueeze(1) + angular_z.unsqueeze(1) * offsets[:, 0]

    # 座標軸の回転修正（90度ずれを補正）
    angle = torch.atan2(
        lateral_velocity,
        velocity + 1e-6
    )

    wheel_speeds = torch.sqrt(
        velocity ** 2 + lateral_velocity ** 2
    )

    # 方向の調整（逆転している可能性を補正）
    angle = -angle  # 必要に応じて逆転

    # ステアとホイールのジョイントインデックスを統合
    all_joint_indices = torch.cat([steer_joint_indices, torch.tensor(joint_indices, device=device)])

    # ステア角度とホイール速度を統合
    all_targets = torch.cat([angle, adjusted_actions], dim=1)

    # 全ジョイントを一括で設定
    asset.set_joint_position_target(
        target=all_targets,
        joint_ids=all_joint_indices,
        env_ids=valid_env_ids
    )

    # ステア角度とホイール速度を設定
    # asset.set_joint_position_target(
    #     target=angle, 
    #     joint_ids=steer_joint_indices,
    #     env_ids=valid_env_ids
    # )

    wheel_speeds_adj = wheel_speeds + actions_wheel[valid_env_ids] * 0.25

    asset.set_joint_velocity_target(
        target=wheel_speeds * 55, 
        joint_ids=wheel_joint_indices,
        env_ids=valid_env_ids
    )

    # 固定しなければならないジョイントがあるなら固定
    if joint_pos_to_fix:
        set_joint_angles(env, valid_env_ids, asset_cfg, joint_pos_to_fix)

    # end_time = time.time()
    # print(f"[INFO] Processing time: {end_time - start_time:.6f} seconds")

def apply_learned_policy(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_pos_to_fix: Dict[str, float] = None,
    observation_info: Dict[str, float] = None,
    debug_mode: bool = False,
):
    """
    学習済み方策を使用してロボットを制御する関数。

    Args:
        env (ManagerBasedEnv): 環境インスタンス。
        env_ids (torch.Tensor): 環境IDのリスト。
        asset_cfg (SceneEntityCfg): アセット設定。
        policy_path (str): 学習済み方策ファイルへのパス。

    Returns:
        None
    """
    global previous_actions_walking_mode
    global actions_change_mode

    # try:
    #     last_action_value = mdp.last_action(env)
    # except:
    #     pass

    # 条件に一致する環境IDを取得
    valid_env_mask = actions_change_mode.squeeze(-1) > 0.5  # 値が > 0 の環境を抽出
    if debug_mode:
        valid_env_ids = env_ids # debug_modeがTrueだったらすべての環境に対して実行
    else:
        valid_env_ids = env_ids[valid_env_mask.nonzero(as_tuple=True)[0]]

    if valid_env_ids.numel() == 0:
        print("[INFO] No environments met the condition for apply_learned_policy.")
        return

    # ✅ 観測データを取得
    try:
        observations = env.observation_manager.compute()
        current_observations = torch.tensor(
            observations['policy'], dtype=torch.float32, device=device
        ).clone().detach()
        # print(f"[INFO] Observations shape: {current_observations.shape}")
    except Exception as e:
        print(f"[Error] Failed to get observations: {e}")
        return

    # info = {
    #     "base_lin_vel": 3,
    #     "base_ang_vel": 3,
    #     "projected_gravity": 3,
    #     "velocity_commands": 3,
    #     "joint_pos": 16,
    #     "joint_vel": 28,
    #     "actions": 1,
    #     "height_scan": 187,
    # }

    # ✅ 観測データを調整 (actions 部分を置き換え)
    # current_observations = replace_observation_column(
    #     current_observations, 
    #     observation_info, 
    #     'actions', 
    #     previous_actions_walking_mode
    # )

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
            # print(f"[INFO] Actions replaced without affecting height_scan. New shape: {current_observations.shape}")
        else:
            print(f"[Error] Unexpected observation shape: {current_observations.shape}")
            return
    except Exception as e:
        print(f"[Error] Failed to adjust observations: {e}")
        return

    # ✅ 方策からアクションを取得
    try:
        with torch.no_grad():
            actions = walking_mode_policy(current_observations).to(device) 
            # print(f"[INFO] Actions from policy: {actions.shape}")
    except Exception as e:
        print(f"[Error] Failed to infer actions from policy: {e}")
        return

    # ✅ アクションを記録（次回の入力に使用）
    previous_actions_walking_mode = actions.clone().detach()
    
    # ✅ 各ジョイントのオフセットを直接指定（歩行を学習したときに使用したものを設定する）
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
        "slider_joint": 0.325
    }

    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = asset.joint_names

    # ✅ ジョイントへのアクションマッピング
    target_joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
    ]

    for i, joint_name in enumerate(target_joint_names):
        if joint_name in joint_names:
            joint_index = joint_names.index(joint_name)
            
            # ✅ 事前定義されたオフセットを適用
            offset = joint_offsets.get(joint_name, 0.0)

            if validate_env_and_joint_ids(valid_env_ids, torch.tensor([joint_index])):  # 起動初期は数が合わないことがあるのでチェック（もっとちゃんとしたやり方があるはずだが.tmpなやり方です）
                # ✅ オフセットを考慮したアクションを適用
                adjusted_action = actions[valid_env_ids, i].unsqueeze(-1) * walking_mode_policy_action_scale + offset
                asset.set_joint_position_target(
                    target=adjusted_action,
                    joint_ids=joint_index,
                    env_ids=valid_env_ids
                )
        
    # 固定しなければならないジョイントがあるなら固定
    if joint_pos_to_fix:
        set_joint_angles(env, valid_env_ids, asset_cfg, joint_pos_to_fix)

def change_mode_prediction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    debug_mode: bool = False,
):
    global actions_change_mode

    # print(actions_change_mode)

    if len(env_ids) == 0:
        return

    # ① ActionManager からバイナリリスト用の ActionTerm を取得する
    # ここでは "binary_action" というキーで登録されていると仮定します。
    try:
        # print("[START]")
        continuous_action_term = env.action_manager.get_term("continuous_action")
        continuous_list = continuous_action_term.get_list()
        # actions_change_mode = (continuous_list > 0).float()
        actions_change_mode = torch.ones((len_env_ids, 1), dtype=torch.float32, device=device)

        # 平均と標準偏差を計算
        mean_val = continuous_list.mean()
        std_val = continuous_list.std()
        # print("[END]")
        # print(continuous_list)
        print(f"mean = {mean_val:.4f}, std = {std_val:.4f}")

        return 
    except Exception as e:
        print(f"[Error] Failed to get binary_action term: {e}")
        return

    # print(actions_change_mode)

    # # ✅ 観測データを取得
    # try:
    #     observations = env.observation_manager.compute()
    #     current_observations = torch.tensor(
    #         observations['policy'], dtype=torch.float32, device=device
    #     ).clone().detach()[env_ids]
    #     # print(f"[INFO] Observations shape: {current_observations.shape}")
    # except Exception as e:
    #     print(f"[Error] Failed to get observations: {e}")
    #     return

    # try:
    #     if current_observations.shape[1] == 244:
    #         # actions (Index 6) を置き換え
    #         action_start_idx = sum([3, 3, 3, 3, 16, 28])  # actions の開始インデックス (上記表から計算)
    #         action_end_idx = action_start_idx + 1  # 現在の actions は (1,)

    #         # actions 部分を削除
    #         current_observations = torch.cat([
    #             current_observations[:, :action_start_idx],  # actions手前まで
    #             current_observations[:, action_end_idx:]  # actions 以降
    #         ], dim=1)
    #         # print(f"[INFO] Actions replaced without affecting height_scan. New shape: {current_observations.shape}")
    #     else:
    #         print(f"[Error] Unexpected observation shape: {current_observations.shape}")
    #         return
    # except Exception as e:
    #     print(f"[Error] Failed to adjust observations: {e}")
    #     return

    # # ✅ 方策からアクションを取得
    # try:
    #     with torch.no_grad():
    #         actions = change_mode_policy(current_observations).to(device) 
    #         # print(f"[INFO] Actions from policy: {actions.shape}")
    # except Exception as e:
    #     print(f"[Error] Failed to infer actions from policy: {e}")
    #     return

    # # ✅ アクションを記録（次回の入力に使用）
    # actions_change_mode[env_ids] = (actions.clone().detach() > 0.5).to(torch.float32)
   
    # # sim_context = SimulationContext.instance()
    # # sim_time = sim_context.current_time 
    # # print(sim_time)
    # # print(len(env_ids))

def change_mode_reward(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    terrain_y: Dict[str, float] = None,
    desirable_mode: Dict[str, float] = None,
    debug_mode: bool = False,
):
    """
    - env_idsごとにロボットのY座標を取得し、
    - 事前に定義された terrain_y を使って「最も近いサブテレイン」を特定、
    - さらに desirable_mode から、そのサブテレインに望ましいモード (1 or -1) を取得する。
    - actions_change_mode と比較して簡易的な報酬を付与するサンプル。
    """
    global actions_change_mode

    # もし地形情報やモードラベルが設定されていなければ何もしない
    if terrain_y is None or desirable_mode is None:
        return

    # env_ids が空なら何もしない
    if len(env_ids) == 0:
        return

    # ----------------------------------------------------------------------------
    # 1. ロボットのY座標を取得（Isaac Labでは get_world_poses() ではなく root_pos_w を参照）
    # ----------------------------------------------------------------------------
    asset: Articulation = env.scene["robot"]
    # shape: (num_envs, 3) → (pos_x, pos_y, pos_z)
    pos_w = asset.data.root_pos_w[env_ids]  # env_idsで抽出
    y_positions = pos_w[:, 1]               # Y座標のみ

    # ----------------------------------------------------------------------------
    # 2. terrain_yから「最も近いサブテレイン」を検索
    # ----------------------------------------------------------------------------
    #   terrain_y: {サブテレイン名: y座標}
    terrain_names = list(terrain_y.keys())                        
    terrain_vals = torch.tensor(list(terrain_y.values()),
                                device=y_positions.device,
                                dtype=torch.float32)               

    # 距離行列 dist: shape = (len(env_ids), num_terrains)
    dist = torch.abs(y_positions.unsqueeze(1) - terrain_vals.unsqueeze(0))
    # 各envについて最小距離の列インデックスを取得
    min_idx = torch.argmin(dist, dim=1)

    # サブテレインに対応するモードラベルをまとめて取得
    terrain_labels_list = []
    for idx in min_idx:
        terrain_name = terrain_names[idx]
        terrain_labels_list.append(desirable_mode[terrain_name])  # -1 or 1
    terrain_labels = torch.tensor(terrain_labels_list, device=y_positions.device, dtype=torch.float32)
    
    # print(terrain_labels)

    # # ----------------------------------------------------------------------------
    # # 3. actions_change_mode（実際に選択しているモード）と比較し、報酬を加算
    # # ----------------------------------------------------------------------------
    # #   - actions_change_mode[env_id] が 0.5以上 → 1 (歩行モード)
    # #   - それ未満 → -1 (車輪モード)
    # mode_raw = actions_change_mode[env_ids].squeeze(-1)
    # chosen_mode = torch.where(mode_raw > 0.5,
    #                           torch.tensor(1.0, device=y_positions.device),
    #                           torch.tensor(-1.0, device=y_positions.device))

    # # 一致すれば +1、違えば -1 の簡易報酬例
    # reward = torch.where(chosen_mode == terrain_labels,
    #                      torch.tensor(1.0, device=y_positions.device),
    #                      torch.tensor(-1.0, device=y_positions.device))

    # # 環境の reward_buf に加算
    # env.reward_buf[env_ids] += reward

    # # ----------------------------------------------------------------------------
    # # 4. デバッグ出力 (任意)
    # # ----------------------------------------------------------------------------
    # if debug_mode:
    #     for i, e_id in enumerate(env_ids):
    #         print(f"[DEBUG] Env {int(e_id.item())}: y={y_positions[i].item():.2f}, "
    #               f"Nearest Terrain={terrain_names[min_idx[i]]}, Label={terrain_labels[i].item()}, "
    #               f"ChosenMode={chosen_mode[i].item()}, Reward={reward[i].item()}")
