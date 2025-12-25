# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .base_quadruped_wheelchair_env_cfg import *
from transformable_quadruped_wheelchair_isaaclab.utils.terrains.config.custom_terrains import MODE_CHANGE_TERRAINS_CFG
  
@configclass
class QuadrupedWheelchairWalkingModeRlEnvCfg(BaseQuadrupedWheelchairEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

                # TRY
        TRY = True
        if TRY:
            self.scene.terrain.terrain_generator = MODE_CHANGE_TERRAINS_CFG

            self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range = (0, 0)
            self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0, 0)
            self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_2"].slope_range = (0, 0)
            self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv_2"].slope_range = (0, 0)
            
            if getattr(self.curriculum, "terrain_levels", None) is not None:
                if self.scene.terrain.terrain_generator is not None:
                    self.scene.terrain.terrain_generator.curriculum = True
            else:
                if self.scene.terrain.terrain_generator is not None:
                    self.scene.terrain.terrain_generator.curriculum = False

        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            'slider_joint': 0.325,
        }
        self.events.set_joint_angles.params["joint_angles"] = {
            'ChairArm_joint': 0.0,  
            'SittingChairAngle_joint': 0.0, 
            'LFUpper2_joint': 0.0, 
            'RFUpper2_joint': 0.0, 
            'LRUpper2_joint': 0.0,
            'RRUpper2_joint': 0.0, 
            'BottomSeat_joint': 0.0, 
            'LFTire1_joint': 0.0, 
            'RFTire1_joint': 0.0, 
            'LRTire1_joint': 0.0, 
            'RRTire1_joint': 0.0,
            'BackSeat_joint': 0.0, 
            'FootLegSupport_joint': 0.0, 
            'LeftArmSupport_joint': 0.0, 
            'RightArmSupport_joint': 0.0,
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
            'LFTire2_1',
            'RFTire2_1',
            'LRTire2_1',
            'RRTire2_1',
        ]
        
        self.actions.joint_pos.joint_names = [
            ".*_hip_joint",
            ".*_thigh_joint", 
            ".*_calf_joint"
        ]
        self.actions.joint_pos.scale = 0.1

@configclass
class QuadrupedWheelchairWalkingModeRlEnvCfg_PLAY(QuadrupedWheelchairWalkingModeRlEnvCfg):
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
