# # Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import math
from .base_quadruped_wheelchair_env_cfg import *
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg
import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp as quadruped_wheelchair_mdp
import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.manage_mode as manage_mode_events
import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.change_mode as change_mode_events
import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp.events.apply_action as apply_action_events

from transformable_quadruped_wheelchair_isaaclab.utils.terrains.config.custom_terrains import MODE_CHANGE_TERRAINS_CFG

@configclass
class QuadrupedWheelchairTwoModesObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-5.0, 5.0),
        )
       
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()



@configclass
class QuadrupedWheelchairTwoModesWithModeVectorObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-5.0, 5.0),
        )
        mode_vec = ObsTerm(
            func=quadruped_wheelchair_mdp.mode_vector,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class QuadrupedWheelchairTwoModesEventCfg(BaseQuadrupedWheelchairEventCfg):
    """Configuration for events."""

    apply_action = EventTerm(
        func=apply_action_events.apply_action,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # environment step size: 0.02
        params={
            "joint_pos_control": None,
            "joint_vel_control": None,
            "asset_cfg": SceneEntityCfg("robot")
        },
    )

    apply_predefined_joint_angle = EventTerm(
        func=apply_action_events.apply_predefined_joint_angle,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # environment step size: 0.02
        params={
            "t_joint_names": [
                'slider_joint', 
                'ChairArm_joint', 
                'SittingChairAngle_joint', 
                'BottomSeat_joint', 
                'BackSeat_joint', 
                'FootLegSupport_joint', 
                'LeftArmSupport_joint', 
                'RightArmSupport_joint'],
            "t_joint_angles": [0.325, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )

    apply_predefined_joint_angle_sub = EventTerm(
        func=apply_action_events.apply_predefined_joint_angle,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # environment step size: 0.02
        params={
            "t_joint_names": [
                'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
                'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint',
            ],
            "t_joint_angles": [
                -0.1, 0.1, -0.1, 0.1,
                -2.0, -2.0, -2.0, -2.0, 
            ],
        },
    )

@configclass
class QuadrupedWheelchairTwoModesRewardCfg(RewardsCfg):
    walking_mode_joint_preferred_l2 = RewTerm(
        func=quadruped_wheelchair_mdp.joint_preferred_l2,
        weight=-1.0,
        params={
            "preferred_joint_angles": None,
            "alpha": 1,
        },
    )

    wheeled_mode_joint_preferred_l2 = RewTerm(
        func=quadruped_wheelchair_mdp.joint_preferred_l2,
        weight=-1.0,
        params={
            "preferred_joint_angles": None,
            "alpha": 1,
        },
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    actions_raw = quadruped_wheelchair_mdp.ListActionCfg(
        list_length=None,  # 28個すべての関節を操作対象とする
        default_value=0
    )  

# 関節類の設定
DEFAULT_ACTION_JOINTS_POS = [
    'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
    'LFUpper2_joint', 'RFUpper2_joint', 'LRUpper2_joint', 'RRUpper2_joint', 
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
    'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint', 
]
DEFAULT_ACTION_JOINTS_VEL = ['LFTire1_joint', 'RFTire1_joint', 'LRTire1_joint', 'RRTire1_joint']
DEFAULT_OFFSET = {
    'LFUpper2_joint': 0.0, 'RFUpper2_joint': 0.0, 'LRUpper2_joint': 0.0, 'RRUpper2_joint': 0.0, 
    'FL_hip_joint': 0.0, 'FR_hip_joint': 0.0, 'RL_hip_joint': 0.0, 'RR_hip_joint': 0.0,
    'FL_thigh_joint': 0.0, 'FR_thigh_joint': 0.0, 'RL_thigh_joint': 0.0, 'RR_thigh_joint': 0.0, 
    'FL_calf_joint': 0.0, 'FR_calf_joint': 0.0, 'RL_calf_joint': 0.0, 'RR_calf_joint': 0.0, 
}
# DEFAULT_FIXED_POS_JOINT_TARGET = 

WALKING_MODE_PREFERRED_ANGLES = {
    'LFUpper2_joint': 0.0, 'RFUpper2_joint': 0.0, 'LRUpper2_joint': 0.0, 'RRUpper2_joint': 0.0, 
    'LFTire1_joint': 0.0, 'RFTire1_joint': 0.0, 'LRTire1_joint': 0.0, 'RRTire1_joint': 0.0, 
    'FL_hip_joint': 0.2, 'FR_hip_joint': -0.2, 'RL_hip_joint': 0.2, 'RR_hip_joint': -0.2,       # 以下はweakに制約をかけたほうがよさそうな項目
    'FL_thigh_joint': 0.8, 'FR_thigh_joint': 0.8, 'RL_thigh_joint': 1.0, 'RR_thigh_joint': 1.0, 
    'FL_calf_joint': -1.5, 'FR_calf_joint': -1.5, 'RL_calf_joint': -1.5, 'RR_calf_joint': -1.5, 
}

WHEELED_MODE_PREFERRED_ANGLES = {
    'LFUpper2_joint': 0.0, 'RFUpper2_joint': 0.0, 'LRUpper2_joint': 0.0, 'RRUpper2_joint': 0.0, # 操舵
    # 'LFTire1_joint': 0.0, 'RFTire1_joint': 0.0, 'LRTire1_joint': 0.0, 'RRTire1_joint': 0.0,     # 車輪移動
    'FL_hip_joint': 0.2, 'FR_hip_joint': -0.2, 'RL_hip_joint': 0.2, 'RR_hip_joint': -0.2,       # 以下はweakに制約をかけたほうがよさそうな項目
    'FL_thigh_joint': 0.0, 'FR_thigh_joint': 0.0, 'RL_thigh_joint': 0.0, 'RR_thigh_joint': 0.0, 
    'FL_calf_joint': -2.0, 'FR_calf_joint': -2.0, 'RL_calf_joint': -2.0, 'RR_calf_joint': -2.0,
}

WALKING_OFFSET = { # 中間のオフセットに対応したバージョン
    'FL_hip_joint': 0.0, 'FR_hip_joint': 0.0, 'RL_hip_joint': 0.0, 'RR_hip_joint': 0.0,
    'FL_thigh_joint': 0.8, 'FR_thigh_joint': 0.8, 'RL_thigh_joint': 1.0, 'RR_thigh_joint': 1.0, 
    'LFUpper2_joint': 0.0, 'RFUpper2_joint': 0.0, 'LRUpper2_joint': 0.0, 'RRUpper2_joint': 0.0, 
    'FL_calf_joint': -1.5, 'FR_calf_joint': -1.5, 'RL_calf_joint': -1.5, 'RR_calf_joint': -1.5, 
}

WHEEL_OFFSET = {
    'FL_hip_joint': 0.1, 'FR_hip_joint': -0.1, 'RL_hip_joint': 0.1, 'RR_hip_joint': -0.1,       # 以下はweakに制約をかけたほうがよさそうな項目
    'FL_thigh_joint': 0.0, 'FR_thigh_joint': 0.0, 'RL_thigh_joint': 0.0, 'RR_thigh_joint': 0.0, 
    'LFUpper2_joint': 0.0, 'RFUpper2_joint': 0.0, 'LRUpper2_joint': 0.0, 'RRUpper2_joint': 0.0, # 操舵
    'FL_calf_joint': -2.0, 'FR_calf_joint': -2.0, 'RL_calf_joint': -2.0, 'RR_calf_joint': -2.0,
}
 
WHEELED_ACTION_JOINTS_POS = [
    'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
    'LFUpper2_joint', 'RFUpper2_joint', 'LRUpper2_joint', 'RRUpper2_joint', 
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
    # 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint', 
]

WALKING_ACTION_JOINTS_POS = [
    'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
    # 'LFUpper2_joint', 'RFUpper2_joint', 'LRUpper2_joint', 'RRUpper2_joint', 
    'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
    'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint', 
]

@configclass
class QuadrupedWheelchairTwoModesEnv_Normal_Cfg(BaseQuadrupedWheelchairEnvCfg):
    observations: QuadrupedWheelchairTwoModesObservationsCfg = QuadrupedWheelchairTwoModesObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: QuadrupedWheelchairTwoModesRewardCfg = QuadrupedWheelchairTwoModesRewardCfg()

    events: QuadrupedWheelchairTwoModesEventCfg = QuadrupedWheelchairTwoModesEventCfg()  

    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.COLLECT_WHEELED_DYNAMICS = True
   
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0) 
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0) 

        self.events.apply_action = EventTerm(
            func=apply_action_events.apply_action_16_only,
            mode="interval",
            interval_range_s=(0.02, 0.02),  # environment step size: 0.02
            params={
                "joint_pos_control": None,
                "joint_vel_control": None,
                "joint_offset": None,
                "asset_cfg": SceneEntityCfg("robot")
            },
        )

        ACTION_JOINTS_POS = list(DEFAULT_ACTION_JOINTS_POS) 
        self.events.apply_action.params["joint_offset"] = DEFAULT_OFFSET 

        self._configure_action_dims(ACTION_JOINTS_POS, DEFAULT_ACTION_JOINTS_VEL) 

        self.events.apply_predefined_joint_angle_sub.params["t_joint_names"] = []
        self.events.apply_predefined_joint_angle_sub.params["t_joint_angles"] = []

        self._configure_joint_init()
        self._configure_terminations()
        self._toggle_curriculum()

        self.rewards.wheeled_mode_joint_preferred_l2 = None 
        self.rewards.walking_mode_joint_preferred_l2 = None

        if self.COLLECT_WHEELED_DYNAMICS == True:
            self.scene.terrain.terrain_generator.sub_terrains = {
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
                )
            }

    def _configure_action_dims(self, POS_JOINTS, VEL_JOINTS):
       
        self.events.apply_action.params["joint_pos_control"] = POS_JOINTS
        
        self.events.apply_action.params["joint_vel_control"] = VEL_JOINTS

        self.actions.actions_raw.list_length = len(POS_JOINTS) + len(VEL_JOINTS)

        print("Configured action dimensions:")
        print(f"  Position-controlled joints ({len(POS_JOINTS)}): {POS_JOINTS}")
        print(f"  Velocity-controlled joints ({len(VEL_JOINTS)}): {VEL_JOINTS}")

    def _configure_joint_init(self):
       
        self.scene.robot.init_state.joint_pos = { # 車輪モードと歩行モードの中間姿勢を基準とする
            ".*L_hip_joint": 0.15,
            ".*R_hip_joint": -0.15,
            "F[L,R]_thigh_joint": 0.785,
            "R[L,R]_thigh_joint": 0.785,
            ".*_calf_joint": -2.0,
            "slider_joint": 0.325,
        }

    def _configure_joint_targets(self, TARGET_DICT):
        self.events.apply_predefined_joint_angle_sub.params["t_joint_names"] = list(TARGET_DICT.keys())
        self.events.apply_predefined_joint_angle_sub.params["t_joint_angles"] = list(TARGET_DICT.values())

        print("Configured joint targets:")
        print(f"  Preset joints ({len(TARGET_DICT.keys())}): {list(TARGET_DICT.keys())}")
        print(f"  Preset angles:     {list(TARGET_DICT.values())}")


    def _configure_rewards(self, WALKING_MODE_PREFERRED_ANGLES_DICT, WHEELED_MODE_PREFERRED_ANGLES_DICT):
        self.rewards.walking_mode_joint_preferred_l2.params["preferred_joint_angles"] = WALKING_MODE_PREFERRED_ANGLES_DICT
        self.rewards.wheeled_mode_joint_preferred_l2.params["preferred_joint_angles"] = WHEELED_MODE_PREFERRED_ANGLES_DICT

        print("Configured reward preferred angles:")
        print(
            "  Walking mode (%d joints): %s",
            len(WALKING_MODE_PREFERRED_ANGLES_DICT),
            list(WALKING_MODE_PREFERRED_ANGLES_DICT.items()),
        )
        print(
            "  Wheeled mode (%d joints): %s",
            len(WHEELED_MODE_PREFERRED_ANGLES_DICT),
            list(WHEELED_MODE_PREFERRED_ANGLES_DICT.items()),
        )

    def _configure_terminations(self):
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base", 
            ".*_thigh", 
            ".*_hip",
            # ".*_calf",
            # ".*Upper2_1",
            # ".*Tire.*",
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
    
    def _toggle_curriculum(self):
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

@configclass
class QuadrupedWheelchairTwoModesEnvCfg(BaseQuadrupedWheelchairEnvCfg):
    observations: QuadrupedWheelchairTwoModesObservationsCfg = QuadrupedWheelchairTwoModesObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: QuadrupedWheelchairTwoModesRewardCfg = QuadrupedWheelchairTwoModesRewardCfg()

    events: QuadrupedWheelchairTwoModesEventCfg = QuadrupedWheelchairTwoModesEventCfg()  

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0) 
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0) 

        self.WALKING_MODE = False

        if self.WALKING_MODE:
            self.events.apply_action = EventTerm(
                func=apply_action_events.apply_action_wk_mode,
                mode="interval",
                interval_range_s=(0.02, 0.02),  # environment step size: 0.02
                params={
                    "joint_pos_control": None,
                    "joint_offset": None,
                    "asset_cfg": SceneEntityCfg("robot")
                },
            )

            ACTION_JOINTS_POS = list(WALKING_ACTION_JOINTS_POS)
            self.events.apply_action.params["joint_offset"] = WALKING_OFFSET
        else:
            self.events.apply_action = EventTerm(
                func=apply_action_events.apply_action_wh_mode,
                mode="interval",
                interval_range_s=(0.02, 0.02), 
                params={
                    "joint_pos_control": None,
                    "joint_vel_control": None,
                    "joint_offset": None,
                    "asset_cfg": SceneEntityCfg("robot")
                },
            )
            self.events.apply_action.params["scale"] = 0.05

            ACTION_JOINTS_POS = list(WHEELED_ACTION_JOINTS_POS)
            self.events.apply_action.params["joint_offset"] = WHEEL_OFFSET

            self.scene.terrain.terrain_generator.sub_terrains = {
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
                )
            }

        self._configure_action_dims(ACTION_JOINTS_POS, DEFAULT_ACTION_JOINTS_VEL)

        CONCAT_ACTION_JOINTS = ACTION_JOINTS_POS + DEFAULT_ACTION_JOINTS_VEL
        filtered_wa = {k: WALKING_MODE_PREFERRED_ANGLES[k] for k in WALKING_MODE_PREFERRED_ANGLES if k in CONCAT_ACTION_JOINTS}
        filtered_wh = {k: WHEELED_MODE_PREFERRED_ANGLES[k] for k in WHEELED_MODE_PREFERRED_ANGLES if k in CONCAT_ACTION_JOINTS}
        filtered_wa_n = {k: WALKING_MODE_PREFERRED_ANGLES[k] for k in WALKING_MODE_PREFERRED_ANGLES if k not in CONCAT_ACTION_JOINTS}
        filtered_wh_n = {k: WHEELED_MODE_PREFERRED_ANGLES[k] for k in WHEELED_MODE_PREFERRED_ANGLES if k not in CONCAT_ACTION_JOINTS}

        if self.WALKING_MODE:
            self._configure_joint_targets(filtered_wa_n)
        else:
            self._configure_joint_targets(filtered_wh_n)

        self._configure_joint_init()
        self._configure_rewards(filtered_wa, filtered_wh)
        self._configure_terminations()
        self._toggle_curriculum()

        if self.WALKING_MODE:
            self.rewards.wheeled_mode_joint_preferred_l2 = None # 車輪モードの報酬を無効化する 
            self.rewards.walking_mode_joint_preferred_l2 = None # 歩行モードの報酬を無効化する
        else:
            self.rewards.wheeled_mode_joint_preferred_l2 = None 
            self.rewards.walking_mode_joint_preferred_l2 = None # 歩行モードの報酬を無効化する
            
    def _configure_action_dims(self, POS_JOINTS, VEL_JOINTS):
        print("Configured action dimensions:")
       
        self.events.apply_action.params["joint_pos_control"] = POS_JOINTS
        print(f"  Position-controlled joints ({len(POS_JOINTS)}): {POS_JOINTS}")
        
        if self.WALKING_MODE == True:
            self.actions.actions_raw.list_length = len(POS_JOINTS)   
        else:
            self.events.apply_action.params["joint_vel_control"] = VEL_JOINTS
            print(f"  Velocity-controlled joints ({len(VEL_JOINTS)}): {VEL_JOINTS}")

            self.actions.actions_raw.list_length = len(POS_JOINTS) + len(VEL_JOINTS)

    def _configure_joint_init(self):
      
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.15,
            ".*R_hip_joint": -0.15,
            "F[L,R]_thigh_joint": 0.785,
            "R[L,R]_thigh_joint": 0.785,
            ".*_calf_joint": -2.0,
            "slider_joint": 0.325,
        }

    def _configure_joint_targets(self, TARGET_DICT):
        self.events.apply_predefined_joint_angle_sub.params["t_joint_names"] = list(TARGET_DICT.keys())
        self.events.apply_predefined_joint_angle_sub.params["t_joint_angles"] = list(TARGET_DICT.values())

        print("Configured joint targets:")
        print(f"  Preset joints ({len(TARGET_DICT.keys())}): {list(TARGET_DICT.keys())}")
        print(f"  Preset angles:     {list(TARGET_DICT.values())}")


    def _configure_rewards(self, WALKING_MODE_PREFERRED_ANGLES_DICT, WHEELED_MODE_PREFERRED_ANGLES_DICT):
        self.rewards.walking_mode_joint_preferred_l2.params["preferred_joint_angles"] = WALKING_MODE_PREFERRED_ANGLES_DICT
        self.rewards.wheeled_mode_joint_preferred_l2.params["preferred_joint_angles"] = WHEELED_MODE_PREFERRED_ANGLES_DICT

        print("Configured reward preferred angles:")
        print(
            "  Walking mode (%d joints): %s",
            len(WALKING_MODE_PREFERRED_ANGLES_DICT),
            list(WALKING_MODE_PREFERRED_ANGLES_DICT.items()),
        )
        print(
            "  Wheeled mode (%d joints): %s",
            len(WHEELED_MODE_PREFERRED_ANGLES_DICT),
            list(WHEELED_MODE_PREFERRED_ANGLES_DICT.items()),
        )
        
    def _configure_terminations(self):
        if self.WALKING_MODE:
            self.terminations.base_contact.params["sensor_cfg"].body_names = [
                "base", 
                ".*_thigh", 
                ".*_hip",
                # ".*_calf",
                # ".*Upper2_1",
                # ".*Tire.*",
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
        else:
            self.terminations.base_contact.params["sensor_cfg"].body_names = [
                "base", 
                ".*_thigh", 
                ".*_hip",
                # ".*_calf",
                # ".*Upper2_1",
                # ".*Tire.*",
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
            ]
    
    def _toggle_curriculum(self):
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

@configclass
class QuadrupedWheelchairTwoModesWithModeVectorEnv_Normal_Cfg(QuadrupedWheelchairTwoModesEnv_Normal_Cfg):
    observations: QuadrupedWheelchairTwoModesWithModeVectorObservationsCfg = QuadrupedWheelchairTwoModesWithModeVectorObservationsCfg()
 
    def __post_init__(self):
        # post init of parent
        super().__post_init__()


@configclass
class QuadrupedWheelchairTwoModesEnvCfg_PLAY(QuadrupedWheelchairTwoModesEnvCfg):
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