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

@configclass
class QuadrupedWheelchairChangeModeRuleEventCfg(BaseQuadrupedWheelchairEventCfg):
    """Configuration for events."""

    # reset
    reset_mode = EventTerm(
        func=manage_mode_events.reset_mode,
        mode="reset",
    )

    # interval
    apply_walking_policy = EventTerm(
        func=walking_mode_events.apply_walking_policy,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # environment step size: 0.02
        params={
            "asset_cfg": SceneEntityCfg("robot")
        },
    )

    four_wheel_independent_steering = EventTerm(
        func=wheel_mode_events.four_wheel_independent_steering,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "front_left_steer": 'LFUpper2_joint',
            "front_right_steer": 'RFUpper2_joint',
            "rear_left_steer": 'LRUpper2_joint',
            "rear_right_steer": 'RRUpper2_joint',
            "front_left_wheel": 'LFTire1_joint',
            "front_right_wheel": 'RFTire1_joint',
            "rear_left_wheel": 'LRTire1_joint',
            "rear_right_wheel": 'RRTire1_joint',
        },
    )

    change_mode_events = EventTerm(
        func=change_mode_events.change_mode_rule,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "asset_name": "robot",
            "rider_asset_cfg": SceneEntityCfg("robot", body_names="teslabot"),
            "thresholds": [-24, -14, -8, 2, 8, 18, 24],
            "tolerances": [3, 1, 1, 1, 1, 1, 1],
            "pause_duration": 2
        }
    )

@configclass
class ChangeModeActionsCfg:
    """Action specifications for the MDP."""

    change_mode = quadruped_wheelchair_mdp.ListActionCfg(
        list_length=1,
        default_value=0
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
  
    base_velocity = quadruped_wheelchair_mdp.GoalTrackingCommandCfg(
        asset_name="robot",
        debug_vis=True,
        goal_positions=[
            [-36.0,  28.0,  0.1225],
            [-28.0,  28.0,  0.15817],
            [-20.0,  28.0,  0.24731],
            [-12.0,  28.0,  0.30337],
            [ -4.0,  28.0,  0.42151],
            [  4.0,  28.0,  0.53068],
            [ 12.0,  28.0,  0.54651],
            [ 20.0,  28.0,  0.65704],
            [ 28.0,  28.0,  0.76092],
            [ 36.0,  28.0,  0.76487]
        ],
        gain=1.0,
        max_speed=1.0, 
        resampling_time_range=(0.1, 0.1)
    )

@configclass
class ChangeModeCurriculumCfg:
    terrain_levels = CurrTerm(func=quadruped_wheelchair_mdp.terrain_levels_vel)    

@configclass
class QuadrupedWheelchairChangeModeRuleEnvCfg(BaseQuadrupedWheelchairEnvCfg):
    actions: ChangeModeActionsCfg = ChangeModeActionsCfg()
    events: QuadrupedWheelchairChangeModeRuleEventCfg = QuadrupedWheelchairChangeModeRuleEventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: ChangeModeCurriculumCfg = ChangeModeCurriculumCfg()    

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.episode_length_s = 5
        print(f"self.episode_length_s: {self.episode_length_s}")

        self.scene.terrain.terrain_generator = MODE_CHANGE_TERRAINS_CFG
      
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range = (0, 0)
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0, 0)
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_2"].slope_range = (0, 0)
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv_2"].slope_range = (0, 0)

        self.events.four_wheel_independent_steering.params["use_mode_flag"] = True
        self.events.four_wheel_independent_steering.params["mode_num"] = 0
        self.events.apply_walking_policy.params["use_mode_flag"] = True
        self.events.apply_walking_policy.params["mode_num"] = 1
        
        self.events.reset_base.params["pose_range"]["yaw"] = (math.pi/2, math.pi/2)

        DEBUG_WALKING_MODE = None
        
        self.events.apply_walking_policy.params["joint_pos_to_fix"] = {
            'ChairArm_joint': 0.0,  
            'SittingChairAngle_joint': 0.0, 
            'BottomSeat_joint': 0.0, 
            'BackSeat_joint': 0.0, 
            'FootLegSupport_joint': 0.0, 
            'LeftArmSupport_joint': 0.0, 
            'RightArmSupport_joint': 0.0,
            'slider_joint': 0.325,

            'LFTire1_joint': 0.0,
            'RFTire1_joint': 0.0,
            'LRTire1_joint': 0.0,
            'RRTire1_joint': 0.0,

            'LFUpper2_joint': 0.0, 
            'RFUpper2_joint': 0.0, 
            'LRUpper2_joint': 0.0,
            'RRUpper2_joint': 0.0, 
        }          
        
        self.events.four_wheel_independent_steering.params["joint_pos_to_fix"] = {
            'ChairArm_joint': 0.0,  
            'SittingChairAngle_joint': 0.0, 
            'BottomSeat_joint': 0.0, 
            'BackSeat_joint': 0.0, 
            'FootLegSupport_joint': 0.0, 
            'LeftArmSupport_joint': 0.0, 
            'RightArmSupport_joint': 0.0,
            'slider_joint': 0.325,

            'FL_calf_joint': -2.0,
            'FR_calf_joint': -2.0,
            'RL_calf_joint': -2.0,
            'RR_calf_joint': -2.0,
        }

        self.events.apply_walking_policy.params["observation_info"] = {
            "base_lin_vel": 3,
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "velocity_commands": 3,
            "joint_pos": 16,
            "joint_vel": 28,
            "actions": 1,
            "height_scan": 187,
        }

        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            # "base", 
            ".*_thigh", 
            ".*_hip",
            # ".*_calf",
            # ".*Upper2_1",
            'ChairArm_1', 
            'Slider_1', 
            'SittingChairAngle_1', 
            'BottomSeat_1', 
            'BackSeat_1', 
            'LeftArmSupport_1', 
            'RightArmSupport_1', 
            # 'FootLegSupport_1', 
            'RightLiDAR_base_1', 
            'RightLiDAR_sphere_1', 
            'LeftLiDAR_base_1', 
            'LeftLiDAR_sphere_1',
            # 'teslabot',
        ]

        self.events.four_wheel_independent_steering.params["joint_offsets"] = {
            'FL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RL_hip_joint': 0.1,
            'RR_hip_joint': -0.1, 
            'FL_thigh_joint': 0.0,
            'FR_thigh_joint': 0.0,
            'RL_thigh_joint': 0.0,
            'RR_thigh_joint': 0.0,
        }
        
        self.events.four_wheel_independent_steering.params["use_learned_model"] = True

        self.scene.robot.init_state.joint_pos['slider_joint'] = 0.325 
        print(f"[DEBUG] scene.robot.init_state.joint_pos: {self.scene.robot.init_state.joint_pos}")

        if DEBUG_WALKING_MODE == True: # True
            self.events.four_wheel_independent_steering = None
            self.events.apply_walking_policy.params["debug_mode"] = True

            self.events.apply_walking_policy.params["debug_mode"] = True

        elif DEBUG_WALKING_MODE == False: # False
          
            self.scene.robot.init_state.joint_pos = {
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.0,
                "R[L,R]_thigh_joint": 0.0,
                ".*_calf_joint": -2.0,
                'slider_joint': 0.325,
            }
            self.events.four_wheel_independent_steering.params["debug_mode"] = True

            self.events.apply_walking_policy = None # 歩行モードを無効化

            self.scene.terrain.terrain_generator.sub_terrains = {
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
                )
            }

        else:
            pass

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

@configclass
class QuadrupedWheelchairChangeModeRuleEnvCfg_PLAY(QuadrupedWheelchairChangeModeRuleEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

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