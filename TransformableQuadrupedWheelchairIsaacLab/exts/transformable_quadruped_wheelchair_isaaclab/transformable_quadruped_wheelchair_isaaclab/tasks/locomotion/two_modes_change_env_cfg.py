# # Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import math
from .base_quadruped_wheelchair_env_cfg import *
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp as quadruped_wheelchair_mdp
import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.walking_mode as walking_mode_events
import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.wheel_mode as wheel_mode_events
import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.manage_mode as manage_mode_events
import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.change_mode as change_mode_events

from transformable_quadruped_wheelchair_isaaclab.utils.terrains.config.custom_terrains import MODE_CHANGE_TERRAINS_CFG

from transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.two_modes_env_cfg import (
    QuadrupedWheelchairTwoModesWithModeVectorEnv_Normal_Cfg,
    QuadrupedWheelchairTwoModesEventCfg
)

@configclass
class QuadrupedWheelchairTwoModesChangeEventCfg(QuadrupedWheelchairTwoModesEventCfg):
    """Configuration for events."""

    # # reset
    reset_mode = EventTerm(
        func=manage_mode_events.reset_mode,
        mode="reset",
    )

    # # interval
    # apply_walking_policy = EventTerm(
    #     func=walking_mode_events.apply_walking_policy,
    #     mode="interval",
    #     interval_range_s=(0.02, 0.02),  # environment step size: 0.02
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot")
    #     },
    # )

    # four_wheel_independent_steering = EventTerm(
    #     func=wheel_mode_events.four_wheel_independent_steering,
    #     mode="interval",
    #     interval_range_s=(0.02, 0.02),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "front_left_steer": 'LFUpper2_joint',
    #         "front_right_steer": 'RFUpper2_joint',
    #         "rear_left_steer": 'LRUpper2_joint',
    #         "rear_right_steer": 'RRUpper2_joint',
    #         "front_left_wheel": 'LFTire1_joint',
    #         "front_right_wheel": 'RFTire1_joint',
    #         "rear_left_wheel": 'LRTire1_joint',
    #         "rear_right_wheel": 'RRTire1_joint',
    #     },
    # )

    change_mode_events = EventTerm(
        func=change_mode_events.change_mode_rule,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "asset_name": "robot",
            "rider_asset_cfg": SceneEntityCfg("robot", body_names="teslabot"),
            "thresholds": [-24, -14, -8, 2, 8, 18, 24, 34],
            "tolerances": [1, 1, 1, 1, 1, 1, 1, 1],
            "pause_duration": 2
        }
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    
    # 新たにゴール追従用のコマンドを追加
    base_velocity = quadruped_wheelchair_mdp.GoalTrackingCommandCfg(
        asset_name="robot",
        debug_vis=True,
        # 各カリキュラムにおけるゴール位置（例として下記のテンソルを与える）
        goal_positions=[
            [-36.0,  36.0,  0.1225], # もともとはy=28
            [-28.0,  36.0,  0.15817],
            [-20.0,  36.0,  0.24731],
            [-12.0,  36.0,  0.30337],
            [ -4.0,  36.0,  0.42151],
            [  4.0,  36.0,  0.53068],
            [ 12.0,  36.0,  0.54651],
            [ 20.0,  36.0,  0.65704],
            [ 28.0,  36.0,  0.76092],
            [ 36.0,  36.0,  0.76487]
        ],
        gain=1.0,        # 差分に対する比例ゲイン（必要に応じて調整）
        max_speed=1.0,   # 各軸の最大速度（m/s）
        resampling_time_range=(0.1, 0.1)
    )

@configclass
class ChangeModeCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=quadruped_wheelchair_mdp.terrain_levels_vel)    

@configclass
class QuadrupedWheelchairTwoModesChangeEnv_Normal_Cfg(QuadrupedWheelchairTwoModesWithModeVectorEnv_Normal_Cfg):
    events: QuadrupedWheelchairTwoModesChangeEventCfg = QuadrupedWheelchairTwoModesChangeEventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: ChangeModeCurriculumCfg = ChangeModeCurriculumCfg()    

    def __post_init__(self):

        class NoOpRanges:
            def __setattr__(self, name, value):
                # この２つだけ無視
                if name in ("lin_vel_x", "lin_vel_y"):
                    return
                super().__setattr__(name, value)

        # GoalTrackingCommandCfg に元々無い ranges を一時的に差し替え
        self.commands.base_velocity.ranges = NoOpRanges()

        # ─── ② あとはそのまま親の __post_init__ を呼ぶ ───────────────────
        super().__post_init__()

        # ─── ③ （必要なら）もう dummy ranges は不要なので消しておく ─────────
        delattr(self.commands.base_velocity, "ranges")

        self.episode_length_s = 75
        print(f"self.episode_length_s: {self.episode_length_s}")

        self.scene.terrain.terrain_generator = MODE_CHANGE_TERRAINS_CFG
        # slopeは平坦にする
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range = (0, 0)
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0, 0)
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_2"].slope_range = (0, 0)
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv_2"].slope_range = (0, 0)
       
        self.events.reset_base.params["pose_range"]["yaw"] = (math.pi/2, math.pi/2)

        self.scene.terrain.max_init_terrain_level = 9

        DEBUG_WALKING_MODE = None

        # self.scene.robot.init_state.joint_pos['slider_joint'] = 0.325 # 歩行移動を学習したときのオフセットに設定する！（こうしないとうまくいかない）
        # print(f"[DEBUG] scene.robot.init_state.joint_pos: {self.scene.robot.init_state.joint_pos}")

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

@configclass
class QuadrupedWheelchairTwoModesChangeEnv_Normal_Cfg_PLAY(QuadrupedWheelchairTwoModesChangeEnv_Normal_Cfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        class NoOpRanges:
            def __setattr__(self, name, value):
                # この２つだけ無視
                if name in ("lin_vel_x", "lin_vel_y"):
                    return
                super().__setattr__(name, value)

        # GoalTrackingCommandCfg に元々無い ranges を一時的に差し替え
        self.commands.base_velocity.ranges = NoOpRanges()

        # ─── ② あとはそのまま親の __post_init__ を呼ぶ ───────────────────
        super().__post_init__()

        # ─── ③ （必要なら）もう dummy ranges は不要なので消しておく ─────────
        delattr(self.commands.base_velocity, "ranges")

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None