# import torch
# from isaaclab.managers import SceneEntityCfg
# # 必要に応じて他のモジュールもimportしてください

# def force_move_robot(
#     env,  # ManagerBasedEnv型の環境インスタンス
#     env_ids: torch.Tensor,
#     desired_position: torch.Tensor,  # shape: (3,) or (len(env_ids), 3)
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ):
#     """
#     ロボット本体のroot state の位置を強制的に desired_position に移動させます。
#     姿勢（orientation）は現状の値をそのまま利用しますが、必要に応じて変更できます。

#     Args:
#       env: ManagerBasedEnv のインスタンス
#       env_ids: 位置を更新する環境IDのテンソル
#       desired_position: 各環境に対して適用する絶対位置。
#                         形状は (3,) あるいは (len(env_ids), 3) となっている必要があります。
#       asset_cfg: 対象資産の設定（デフォルトは "robot"）
#     """
#     # 資産を取得
#     asset = env.scene[asset_cfg.name]
#     # 現在の root state（位置・姿勢・その他の情報）のコピーを取得
#     # ここでは default_root_state は [position(3), orientation(4), velocity(6)] の配列と仮定
#     root_states = asset.data.default_root_state[env_ids].clone()
    
#     # 現在の位置情報を上書きする（desired_position が (3,) なら全環境に同じ値を適用）
#     if desired_position.ndim == 1:
#         # (3,) -> (N, 3) に拡張
#         desired_position = desired_position.unsqueeze(0).repeat(len(env_ids), 1)
#     elif desired_position.shape[0] != len(env_ids):
#         raise ValueError("desired_position の先頭次元は env_ids の数と一致する必要があります。")
    
#     # 現在の姿勢はそのまま利用する
#     orientations = root_states[:, 3:7]
    
#     # 新しい root state 用のテンソルを連結して作成
#     # ここでは、位置とorientation のみ書き換えていますが、必要に応じて速度等も調整できます。
#     new_root_state = torch.cat([desired_position, orientations], dim=-1)
    
#     # 物理シミュレーションに新しい root pose を反映する
#     asset.write_root_pose_to_sim(new_root_state, env_ids=env_ids)


# force_move_robot.py
# force_move_robot.py
# force_move_robot.py

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv  # 型定義は実際のプロジェクトに合わせる

def force_move_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    desired_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:

    asset = env.scene[asset_cfg.name]
    
    asset.write_root_pose_to_sim(desired_pose, env_ids=env_ids)

def force_move_robot_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    desired_pose: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    if len(desired_pose.shape) == 1:
        desired_pose = desired_pose.unsqueeze(0)

    root_states = asset.data.default_root_state[env_ids].clone()
    # root_states[:,:7] = desired_pose
        
    # print(f"desired_pose: {desired_pose.shape}")
    # print(f"env.scene.env_origins[env_ids]: {env.scene.env_origins[env_ids].shape}")
    # positions = root_states[:, 0:3]
    # orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    # orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(desired_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)