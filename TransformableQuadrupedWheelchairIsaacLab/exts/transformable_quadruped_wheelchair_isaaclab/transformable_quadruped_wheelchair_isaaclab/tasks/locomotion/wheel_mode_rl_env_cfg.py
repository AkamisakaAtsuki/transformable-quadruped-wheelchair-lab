# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .base_quadruped_wheelchair_env_cfg import *
import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp as quadruped_wheelchair_mdp
import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.wheel_mode as wheel_mode_events

@configclass
class QuadrupedWheelchairWheelModeEventCfg(BaseQuadrupedWheelchairEventCfg):
    """Configuration for events."""

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
    

@configclass
class WheelModeActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=None, 
        scale=0.5, 
        use_default_offset=True
    )

    wheel_adjustment = quadruped_wheelchair_mdp.ListActionCfg(
        list_length=1,
        default_value=0
    )

@configclass
class QuadrupedWheelchairWheelModeRlEnvCfg(BaseQuadrupedWheelchairEnvCfg):
    actions: WheelModeActionsCfg = WheelModeActionsCfg()
    events: QuadrupedWheelchairWheelModeEventCfg = QuadrupedWheelchairWheelModeEventCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot.init_state.joint_pos = { # 車輪移動や変形学習をメインにするバージョン
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.0,
            "R[L,R]_thigh_joint": 0.0,
            ".*_calf_joint": -2.0,
            'slider_joint': 0.325,
        }

        self.events.set_joint_angles.params["joint_angles"] = {
            'ChairArm_joint': 0.0,  
            'SittingChairAngle_joint': 0.0, 
            'BottomSeat_joint': 0.0, 
            'BackSeat_joint': 0.0, 
            'FootLegSupport_joint': 0.0, 
            'LeftArmSupport_joint': 0.0, 
            'RightArmSupport_joint': 0.0,
            'FL_calf_joint': -2.0,
            'FR_calf_joint': -2.0,
            'RL_calf_joint': -2.0,
            'RR_calf_joint': -2.0,
            'slider_joint': 0.325
        }

        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base", 
            ".*_thigh", 
            ".*_hip",
            ".*_calf",
            ".*Upper2_1",
            'ChairArm_1', 
            'Slider_1', 
            'SittingChairAngle_1', 
            'BottomSeat_1', 
            'BackSeat_1', 
            'LeftArmSupport_1', 
            'RightArmSupport_1', 
            'FootLegSupport_1', 
            'RightLiDAR_base_1', 
            'RightLiDAR_sphere_1', 
            'LeftLiDAR_base_1', 
            'LeftLiDAR_sphere_1',
            'teslabot',
            '.*_foot',
        ]

        self.actions.joint_pos.joint_names = [
            'FL_hip_joint',
            'FR_hip_joint',
            'RL_hip_joint',
            'RR_hip_joint',
            'FL_thigh_joint',
            'FR_thigh_joint',
            'RL_thigh_joint',
            'RR_thigh_joint'
        ]

        self.actions.wheel_adjustment.list_length=4
        self.actions.joint_pos.scale = 0.1
    
        self.scene.terrain.max_init_terrain_level = 5
        self.scene.terrain.terrain_generator.sub_terrains = {
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            )
        }

        self.events.collect_vibration_data = None
        self.events.collect_observation = None
        self.events.change_mode_prediction = None
        self.events.four_wheel_independent_steering.params["learning_model"] = True

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"

@configclass
class QuadrupedWheelchairWheelModeRlEnvCfg_PLAY(QuadrupedWheelchairWheelModeRlEnvCfg):
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

        self.events.four_wheel_independent_steering.params["use_learned_model"] = True

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
