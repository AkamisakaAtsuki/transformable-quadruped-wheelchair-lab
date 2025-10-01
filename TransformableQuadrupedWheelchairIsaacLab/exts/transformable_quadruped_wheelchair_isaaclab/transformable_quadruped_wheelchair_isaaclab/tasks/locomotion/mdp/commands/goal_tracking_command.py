from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
import math
import isaaclab.utils.math as math_utils
import torch

from dataclasses import MISSING
from isaaclab.managers import CommandTerm, CommandTermCfg

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.manage_mode import get_mode, get_stop

if TYPE_CHECKING:
    from .goal_tracking_command_cfg import GoalTrackingCommandCfg

def get_yaw_from_quat(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """
    クォータニオン (x, y, z, w) からヨー角 ([-pi, pi]) を取り出す関数。
    入力:
      quat_xyzw: shape = (N, 4), 各行が (qx, qy, qz, qw)
    出力:
      yaw: shape = (N,), 各環境のヨー角 [rad]
    """
    # x = quat_xyzw[:, 0]
    # y = quat_xyzw[:, 1]
    # z = quat_xyzw[:, 2]
    # w = quat_xyzw[:, 3]

    w = quat_xyzw[:, 0]
    x = quat_xyzw[:, 1]
    y = quat_xyzw[:, 2]
    z = quat_xyzw[:, 3]

    # 2D回転のヨー角は以下の式で算出 (標準的なatan2公式)
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    yaw = torch.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))
    return yaw

class GoalTrackingCommand(CommandTerm):
    """
    ロボットの現在位置と設定されたゴール位置との差分から、目標方向に向かう速度コマンドを生成するコマンド生成器。
    
    ① 環境のカリキュラム（各環境ごとのゴール位置）は設定パラメータとして渡す。
    ② _resample_command 内で、各環境ごとに現在位置とゴール位置との差分を計算し、
       ゲインをかけた上で最大速度でクリッピングした速度コマンドを生成する。
    """
    cfg: GoalTrackingCommandCfg

    # 矢印可視化用
    cmd_vel_visualizer: VisualizationMarkers | None = None
    actual_vel_visualizer: VisualizationMarkers | None = None

    def __init__(self, cfg: GoalTrackingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]

        # コマンド出力用バッファ
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """(num_envs, 3) のベースフレーム速度コマンド"""
        """stopモードに対応"""
        _stop = get_stop()
        if _stop != None:
            return (1 - _stop)*self.vel_command_b
        else:
            return self.vel_command_b
    
    # @property
    # def command(self) -> torch.Tensor:
    #     """(num_envs, 3) のベースフレーム速度コマンド"""
    #     return self.vel_command_b

#    def _resample_command(self, env_ids: Sequence[int]):
#         # ---------------------------------------------------
#         # 1. ワールド座標系での差分ベクトルを求める
#         # ---------------------------------------------------
#         current_positions = self.robot.data.root_pos_w[env_ids]      # shape: (n,3)
#         terrain_levels     = self._env.scene.terrain.terrain_levels[env_ids]
#         goal_positions     = self.cfg.goal_positions[terrain_levels] # shape: (n,3)

#         diff_w = goal_positions - curren t_positions                  # shape: (n,3)
#         diff_norm = torch.norm(diff_w, dim=1, keepdim=True)          # shape: (n,1)

#         # 方向ベクトル (ワールド座標系)
#         direction_w = diff_w / (diff_norm + 1e-8)
        
#         # 速度大きさを決定 (例: 常に max_speed)
#         desired_speed = self.cfg.max_speed
#         # ゲインを掛けるなら: desired_speed *= self.cfg.gain

#         # ワールド座標系での目標速度
#         desired_vel_w = direction_w * desired_speed

#         # Z成分を無視する場合
#         desired_vel_w[:, 2] = 0.0

#         # ---------------------------------------------------
#         # 2. ワールド → ベース座標系への変換
#         # ---------------------------------------------------
#         # ロボットrootの姿勢(クォータニオン)
#         quat_w = self.robot.data.root_quat_w[env_ids]  # shape: (n,4)

#         # quat_rotate_inverse でワールド座標のベクトルをベース座標系へ変換
#         desired_vel_b = math_utils.quat_rotate_inverse(quat_w, desired_vel_w)

#         # ---------------------------------------------------
#         # 3. 結果を velocity command buffer に格納
#         # ---------------------------------------------------
#         self.vel_command_b[env_ids] = desired_vel_b  

    def _resample_command(self, env_ids: Sequence[int]):
        # ---------------------------------------------------
        # 1. ロボットの現在位置/ゴール位置の差分を計算
        # ---------------------------------------------------
        current_positions = self.robot.data.root_pos_w[env_ids]         # shape: (n,3)
        terrain_levels = self._env.scene.terrain.terrain_levels[env_ids]
        positions_tensor = torch.tensor(self.cfg.goal_positions, device="cuda:0")
        goal_positions = positions_tensor[terrain_levels]  # shape = (len(terrain_levels), 3)

        # goal_positions     = torch.tensor(self.cfg.goal_positions[terrain_levels], device='cuda:0')   # shape: (n,3)

        diff_w = goal_positions - current_positions                     # (n,3)
        diff_norm = torch.norm(diff_w, dim=1, keepdim=True)             # (n,1)

        # 方向ベクトル (ワールド座標系)
        direction_w = diff_w / (diff_norm + 1e-8)

        # ---------------------------------------------------
        # 2. ロボットを前進させるための速度大きさを決定
        # ---------------------------------------------------
        desired_speed = self.cfg.max_speed
        # 例: 距離に応じてスケーリングするなら:
        # desired_speed = torch.clamp(self.cfg.gain * diff_norm.squeeze(-1),
        #                             max=self.cfg.max_speed)

        # ワールド座標系での並進速度ベクトル (z成分は無視)
        desired_vel_w = direction_w * desired_speed
        desired_vel_w[:, 2] = 0.0

        # ---------------------------------------------------
        # 3. ゴール方向のヨー角を計算
        # ---------------------------------------------------
        heading_w = torch.atan2(direction_w[:, 1], direction_w[:, 0])  # (n,)

        # ロボットの現在ヨー角をクォータニオンから取得
        quat_w = self.robot.data.root_quat_w[env_ids]  # (n,4)
        current_yaw = get_yaw_from_quat(quat_w)         # (n,)
        
        # print(f"heading_w: {heading_w}, quat_w: {quat_w}, current_yaw: {current_yaw}")

        # ---------------------------------------------------
        # 4. ヨー角誤差を [-pi, pi] に折り返す
        # ---------------------------------------------------
        yaw_error = heading_w - current_yaw
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))

        # ---------------------------------------------------
        # 5. ヨー角速度(heading_rate)を計算
        # ---------------------------------------------------
        kp_yaw = 1.5
        heading_rate = kp_yaw * yaw_error
        heading_rate = torch.clamp(heading_rate, min=-2.0, max=2.0)

        # ---------------------------------------------------
        # 6. 並進速度をベース座標系に変換 + yaw軸速度をセット
        # ---------------------------------------------------
        desired_vel_b = math_utils.quat_apply_inverse(quat_w, desired_vel_w)

        self.vel_command_b[env_ids, 0] = desired_vel_b[:, 0]  # 前進速度
        self.vel_command_b[env_ids, 1] = desired_vel_b[:, 1]  # 横移動 (不要なら 0.0)
        self.vel_command_b[env_ids, 2] = heading_rate         # yaw角速度

    def _update_command(self):
        """ここでは特に処理せず、_resample_commandで決めたコマンドをそのまま使う。"""
        pass

    def _update_metrics(self):
        """コマンドと実際の速度の誤差を計算してメトリクスに格納。"""
        actual_vel_b = self.robot.data.root_lin_vel_b
        error = torch.norm(self.vel_command_b - actual_vel_b, dim=1)
        self.metrics["velocity_error"] = error

    def _set_debug_vis_impl(self, debug_vis: bool):
        """
        デバッグ可視化（矢印）を有効/無効にする。
        """
        if debug_vis:
            if not self.cmd_vel_visualizer:
                # 命令方向（緑）
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_velocity"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.cmd_vel_visualizer = VisualizationMarkers(marker_cfg)

            if not self.actual_vel_visualizer:
                # 実際の進行方向（青）
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/actual_velocity"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.actual_vel_visualizer = VisualizationMarkers(marker_cfg)

            self.cmd_vel_visualizer.set_visibility(True)
            self.actual_vel_visualizer.set_visibility(True)
        else:
            if self.cmd_vel_visualizer:
                self.cmd_vel_visualizer.set_visibility(False)
            if self.actual_vel_visualizer:
                self.actual_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """
        命令方向(緑)と実際の進行方向(青)を可視化するコールバック。
        """
        if not self.robot.is_initialized:
            return
        if not (self.cmd_vel_visualizer and self.actual_vel_visualizer):
            return

        # ロボットの位置を少し上にオフセット
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # 命令方向（緑）
        _stop = get_stop()
        if _stop != None:
            cmd_xy = (1 - _stop)*self.vel_command_b[:, :2]
        else:
            cmd_xy = self.vel_command_b[:, :2]
        cmd_scale, cmd_quat = self._resolve_xy_velocity_to_arrow(cmd_xy)

        self.cmd_vel_visualizer.visualize(
            translations=base_pos_w,
            orientations=cmd_quat,
            scales=cmd_scale,
        )

        # 実際の進行方向（青）
        actual_xy = self.robot.data.root_lin_vel_b[:, :2]
        actual_scale, actual_quat = self._resolve_xy_velocity_to_arrow(actual_xy)

        self.actual_vel_visualizer.visualize(
            translations=base_pos_w,
            orientations=actual_quat,
            scales=actual_scale,
        )

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        XYベクトルから矢印マーカー用のスケール・クォータニオンを生成。
        """
        num_envs = xy_velocity.shape[0]
        # デフォルトスケール
        default_scale = torch.tensor([0.5, 0.5, 0.5], device=self.device)

        # ベクトルの大きさ
        norm_xy = torch.norm(xy_velocity, dim=1)
        # 矢印の X軸方向スケールに速度の大きさを乗じる
        arrow_scale = default_scale.unsqueeze(0).repeat(num_envs, 1)
        arrow_scale[:, 0] *= norm_xy * 3.0  # 係数3.0は見やすさのため

        # 進行方向
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)

        # Euler -> Quaternion (roll=0, pitch=0, yaw=heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        # ロボット姿勢に合わせて回転させる（ワールド座標で可視化する場合）
        arrow_quat = math_utils.quat_mul(self.robot.data.root_quat_w, arrow_quat)

        return arrow_scale, arrow_quat