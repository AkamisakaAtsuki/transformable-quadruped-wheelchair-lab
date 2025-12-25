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
    w = quat_xyzw[:, 0]
    x = quat_xyzw[:, 1]
    y = quat_xyzw[:, 2]
    z = quat_xyzw[:, 3]

    yaw = torch.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))
    return yaw

class GoalTrackingCommand(CommandTerm):
    cfg: GoalTrackingCommandCfg

    cmd_vel_visualizer: VisualizationMarkers | None = None
    actual_vel_visualizer: VisualizationMarkers | None = None

    def __init__(self, cfg: GoalTrackingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]

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

    def _resample_command(self, env_ids: Sequence[int]):
        current_positions = self.robot.data.root_pos_w[env_ids]        
        terrain_levels = self._env.scene.terrain.terrain_levels[env_ids]
        positions_tensor = torch.tensor(self.cfg.goal_positions, device="cuda:0")
        goal_positions = positions_tensor[terrain_levels] 

        diff_w = goal_positions - current_positions                   
        diff_norm = torch.norm(diff_w, dim=1, keepdim=True)            

        direction_w = diff_w / (diff_norm + 1e-8)

        desired_speed = self.cfg.max_speed
      
        desired_vel_w = direction_w * desired_speed
        desired_vel_w[:, 2] = 0.0

        heading_w = torch.atan2(direction_w[:, 1], direction_w[:, 0])  # (n,)

        quat_w = self.robot.data.root_quat_w[env_ids]  # (n,4)
        current_yaw = get_yaw_from_quat(quat_w)         # (n,)
       
        yaw_error = heading_w - current_yaw
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))

        kp_yaw = 1.5
        heading_rate = kp_yaw * yaw_error
        heading_rate = torch.clamp(heading_rate, min=-2.0, max=2.0)

        desired_vel_b = math_utils.quat_apply_inverse(quat_w, desired_vel_w)

        self.vel_command_b[env_ids, 0] = desired_vel_b[:, 0] 
        self.vel_command_b[env_ids, 1] = desired_vel_b[:, 1]  
        self.vel_command_b[env_ids, 2] = heading_rate        

    def _update_command(self):
       
        pass

    def _update_metrics(self):
        
        actual_vel_b = self.robot.data.root_lin_vel_b
        error = torch.norm(self.vel_command_b - actual_vel_b, dim=1)
        self.metrics["velocity_error"] = error

    def _set_debug_vis_impl(self, debug_vis: bool):
        
        if debug_vis:
            if not self.cmd_vel_visualizer:
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_velocity"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.cmd_vel_visualizer = VisualizationMarkers(marker_cfg)

            if not self.actual_vel_visualizer:
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
        if not self.robot.is_initialized:
            return
        if not (self.cmd_vel_visualizer and self.actual_vel_visualizer):
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

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

        actual_xy = self.robot.data.root_lin_vel_b[:, :2]
        actual_scale, actual_quat = self._resolve_xy_velocity_to_arrow(actual_xy)

        self.actual_vel_visualizer.visualize(
            translations=base_pos_w,
            orientations=actual_quat,
            scales=actual_scale,
        )

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = xy_velocity.shape[0]
        default_scale = torch.tensor([0.5, 0.5, 0.5], device=self.device)
        norm_xy = torch.norm(xy_velocity, dim=1)
        arrow_scale = default_scale.unsqueeze(0).repeat(num_envs, 1)
        arrow_scale[:, 0] *= norm_xy * 3.0  

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)

        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        arrow_quat = math_utils.quat_mul(self.robot.data.root_quat_w, arrow_quat)

        return arrow_scale, arrow_quat