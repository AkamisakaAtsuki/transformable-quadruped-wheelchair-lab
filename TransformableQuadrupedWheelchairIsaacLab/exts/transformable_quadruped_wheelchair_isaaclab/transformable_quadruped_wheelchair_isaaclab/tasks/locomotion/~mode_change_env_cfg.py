# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp as quadruped_wheelchair_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from transformable_quadruped_wheelchair_isaaclab.utils.terrains.config.custom_terrains import MODE_CHANGE_TERRAINS_CFG  # isort: skip
from transformable_quadruped_wheelchair_isaaclab.assets.config.quadruped_wheelchair import Quadruped_Wheelchair_CFG  # isort: skip

@configclass
class ObservationsCfg:
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
        joint_pos = ObsTerm(
            func=quadruped_wheelchair_mdp.my_joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name='robot', 
                    joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint", ".*Upper2_joint"]
                )
            }, 
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-5.0, 5.0),
        )
        # fall_risk = ObsTerm(
        #     func=lambda env, env_ids: torch.tensor(
        #         [shared_data.predicted_fall_risks[int(env_id.item())] for env_id in env_ids],
        #         dtype=torch.float32,
        #         device=device,
        #     )
        # )
        # predicted_fall_risks = ObsTerm(func=unitree_b2_custom_mdp.get_predicted_fall_risks)
       

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class QuadrupedWheelchairEventCfg(EventCfg):
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    add_teslabot_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="teslabot"),
            "mass_distribution_params": (-20.0, 30.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    set_joint_angles = EventTerm(
        func=quadruped_wheelchair_mdp.set_joint_angles,
        mode="interval",
        interval_range_s=(0.1, 0.15),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_angles": None
        },
    )

    teslabot_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(1.0, 1.5),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="teslabot"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )

    four_wheel_independent_steering = EventTerm(
        func=quadruped_wheelchair_mdp.four_wheel_independent_steering,
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

    apply_learned_policy = EventTerm(
        func=quadruped_wheelchair_mdp.apply_learned_policy,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # environment step size: 0.02
        params={
            "asset_cfg": SceneEntityCfg("robot")
        },
    )

    collect_vibration_data = EventTerm(
        func=quadruped_wheelchair_mdp.collect_vibration_data,
        mode="interval",
        interval_range_s=(0.005, 0.005),
        params={
            "asset_cfg": SceneEntityCfg("robot")
        },
    )

    collect_observation = EventTerm(
        func=quadruped_wheelchair_mdp.collect_mdp_data,
        mode="interval",
        interval_range_s=(0.1, 0.1),
    )

    change_mode_prediction = EventTerm(
        func=quadruped_wheelchair_mdp.change_mode_prediction,
        mode="interval",
        interval_range_s=(1.0, 1.1),
    )

    change_mode_reward = EventTerm(
        func=quadruped_wheelchair_mdp.change_mode_reward,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "terrain_y": {
                "hf_pyramid_slope": -28,
                "random_rough": -20,
                "hf_pyramid_slope_inv": -12,
                "boxes": -4,
                "hf_pyramid_slope_2": 4,
                "pyramid_stairs_inv": 12,
                "hf_pyramid_slope_inv_2": 20,
                "pyramid_stairs": 28
            },
            "desirable_mode": {
                "hf_pyramid_slope": -1,
                "random_rough": 1,
                "hf_pyramid_slope_inv": -1,
                "boxes": 1,
                "hf_pyramid_slope_2": -1,
                "pyramid_stairs_inv": 1,
                "hf_pyramid_slope_inv_2": -1,
                "pyramid_stairs": 1
            }
        },
    )

@configclass
class TerminationsWithDataCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=quadruped_wheelchair_mdp.time_out_with_collect_data, time_out=True)
    base_contact = DoneTerm(
        func=quadruped_wheelchair_mdp.illegal_contact_with_collect_data,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )

# @configclass
# class TerminationsCfg:
#     """Termination terms for the MDP."""

#     time_out = DoneTerm(func=mdp.time_out, time_out=True)
#     # base_contact = DoneTerm(
#     #     func=mdp.illegal_contact,
#     #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
#     # )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=None, 
        scale=0.5, 
        use_default_offset=True
    )

    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", 
        joint_names=None, 
        scale=1000, 
        use_default_offset=True
    )

@configclass
class ModeChangeActionsCfg:
    """Action specifications for the MDP."""

    # 新たにバイナリリストアクションを追加（リスト長5、初期値0）
    continuous_action = quadruped_wheelchair_mdp.ListActionCfg(
        list_length=1,
        default_value=0
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=quadruped_wheelchair_mdp.terrain_levels_vel)

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    
    # 新たにゴール追従用のコマンドを追加
    base_velocity = quadruped_wheelchair_mdp.GoalTrackingCommandCfg(
        asset_name="robot",
        debug_vis=True,
        # 各カリキュラムにおけるゴール位置（例として下記のテンソルを与える）
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
        gain=1.0,        # 差分に対する比例ゲイン（必要に応じて調整）
        max_speed=1.0,   # 各軸の最大速度（m/s）
        resampling_time_range=(1.0, 1.0)
    )

    
@configclass
class QuadrupedWheelchairModeChangeEnvCfg(LocomotionVelocityRoughEnvCfg):
    # mode = "WALKING_ONLY"
    mode = "WALKING_AND_WHEEL"
    # mode = "WHEEL_ONLY"

    collect_data = False
    collect_observation = False

    # terminations: TerminationsCfg = TerminationsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    commands: CommandsCfg = CommandsCfg()

    # print(curriculum)
    if collect_data == True:
        terminations: TerminationsWithDataCfg = TerminationsWithDataCfg()


    if mode == "WALKING_AND_WHEEL":
        actions: ActionsCfg = ModeChangeActionsCfg()
    else:
        actions: ActionsCfg = ActionsCfg()
    events: QuadrupedWheelchairEventCfg = QuadrupedWheelchairEventCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        print(f"self.episode_length_s: {self.episode_length_s}")
        self.episode_length_s = 50

        self.scene.robot = Quadruped_Wheelchair_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        self.scene.terrain.terrain_generator = MODE_CHANGE_TERRAINS_CFG
        # self.sim.physics_material = self.scene.terrain.physics_material

        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.01, 0.12)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.01, 0.12)
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.12)
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range=(0.0, 0.2)
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_2"].slope_range=(0.0, 0.2)
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range=(0.0, 0.2)
        # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv_2"].slope_range=(0.0, 0.2)

        if self.collect_data == False:
            self.events.collect_vibration_data = None
            
        if self.collect_observation == False:
            self.events.collect_mdp_data = None
        
        if self.mode == "WALKING_ONLY":
            pass

        elif self.mode == "WHEEL_ONLY":
            pass

        elif self.mode == "WALKING_AND_WHEEL":

            DEBUG_WALKING_MODE = None

            # 【共通】 >>>
            # self.actions.joint_pos.joint_names = ['slider_joint']
            # self.actions.joint_pos.scale = 0.001
            # self.actions.wheel_vel = None # [".*Tire1_joint"]
            # <<<

            # 【歩行モードの設定】 >>>
            
            # 歩行モード実行時に常時、固定しておく関節
            self.events.apply_learned_policy.params["joint_pos_to_fix"] = {
                'LFUpper2_joint': 0.0, 
                'RFUpper2_joint': 0.0, 
                'LRUpper2_joint': 0.0,
                'RRUpper2_joint': 0.0, 
            }          

            self.events.apply_learned_policy.params["observation_info"] = {
                "base_lin_vel": 3,
                "base_ang_vel": 3,
                "projected_gravity": 3,
                "velocity_commands": 3,
                "joint_pos": 16,
                "joint_vel": 28,
                "actions": 1,
                "height_scan": 187,
            }

            # 歩行モード時の終了判定
            self.terminations.base_contact.params["sensor_cfg"].body_names = [
                "base", 
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
                'teslabot',
            ]
            # <<<

            # 車輪モードの設定 >>>
            # 車輪モード時に固定しておくパーツ
            self.events.four_wheel_independent_steering.params["joint_pos_to_fix"] = {
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
            
            # <<<

            if DEBUG_WALKING_MODE == True: # True
                # 歩行モードの場合
                # 関節の初期オフセットを設定
                self.scene.robot.init_state.joint_pos = { # 歩行移動を学習したときのオフセットに設定する！（こうしないとうまくいかない）
                    ".*L_hip_joint": 0.1,
                    ".*R_hip_joint": -0.1,
                    "F[L,R]_thigh_joint": 0.8,
                    "R[L,R]_thigh_joint": 1.0,
                    ".*_calf_joint": -1.5,
                    'slider_joint': 0.325,
                }
                
                self.events.four_wheel_independent_steering = None # 車輪モードを無効化
                self.events.apply_learned_policy.params["debug_mode"] = True

            elif DEBUG_WALKING_MODE == False: # False
                # 車輪モードの場合
                # 関節の初期オフセットを設定
                self.scene.robot.init_state.joint_pos = { # 車輪移動を学習したときのオフセットに設定する！（こうしないとうまくいかない）
                    ".*L_hip_joint": 0.1,
                    ".*R_hip_joint": -0.1,
                    "F[L,R]_thigh_joint": 0.0,
                    "R[L,R]_thigh_joint": 0.0,
                    ".*_calf_joint": -2.0,
                    'slider_joint': 0.325,
                }
                self.events.apply_learned_policy = None # 歩行モードを無効化

                self.events.four_wheel_independent_steering.params["debug_mode"] = True
            else:
                pass

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

        # self.rewards.feet_air_time.weight = 0.01
        # self.rewards.undesired_contacts = None
        # self.rewards.dof_torques_l2.weight = -0.0002
        # self.rewards.track_lin_vel_xy_exp.weight = 1.5
        # self.rewards.track_ang_vel_z_exp.weight = 0.75
        # self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        # terminations

        # 並びがランダム化されないためにも必要
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
        


@configclass
class QuadrupedWheelchairModeChangeEnvCfg_PLAY(QuadrupedWheelchairModeChangeEnvCfg):
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
