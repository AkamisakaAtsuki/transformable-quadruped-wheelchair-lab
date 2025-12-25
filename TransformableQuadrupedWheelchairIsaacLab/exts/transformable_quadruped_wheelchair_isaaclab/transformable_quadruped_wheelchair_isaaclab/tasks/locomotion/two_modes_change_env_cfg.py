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
    
    base_velocity = quadruped_wheelchair_mdp.GoalTrackingCommandCfg(
        asset_name="robot",
        debug_vis=True,
        goal_positions=[
            [-36.0,  36.0,  0.1225],
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
        gain=1.0,     
        max_speed=1.0, 
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

       
        self.commands.base_velocity.ranges = NoOpRanges()

      
        super().__post_init__()

        delattr(self.commands.base_velocity, "ranges")

        self.episode_length_s = 75
        print(f"self.episode_length_s: {self.episode_length_s}")

        self.scene.terrain.terrain_generator = MODE_CHANGE_TERRAINS_CFG
       
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range = (0, 0)
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0, 0)
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_2"].slope_range = (0, 0)
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv_2"].slope_range = (0, 0)
       
        self.events.reset_base.params["pose_range"]["yaw"] = (math.pi/2, math.pi/2)

        self.scene.terrain.max_init_terrain_level = 9

        DEBUG_WALKING_MODE = None

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

      
        self.commands.base_velocity.ranges = NoOpRanges()

        super().__post_init__()

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