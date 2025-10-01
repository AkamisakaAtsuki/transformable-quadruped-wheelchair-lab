# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2025, Atsuki Akamisaka.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp as quadruped_wheelchair_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##

from transformable_quadruped_wheelchair_isaaclab.assets.config.quadruped_wheelchair import Quadruped_Wheelchair_CFG  # isort: skip

@configclass
class BaseQuadrupedWheelchairObservationsCfg:
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

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class BaseQuadrupedWheelchairEventCfg(EventCfg):
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
            "mass_distribution_params": (-20.0, 10.0),
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

    set_joint_angles = EventTerm(
        func=quadruped_wheelchair_mdp.events.base.set_joint_angles,
        mode="interval",
        interval_range_s=(0.1, 0.15),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_angles": None
        },
    )


@configclass
class BaseQuadrupedWheelchairActionCfg:
    # ほとんどの環境で共通するジョイント位置アクション。
    # 各モードで異なる設定が必要な場合は、サブクラスで上書き・拡張してください。
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        scale=0.1,
        use_default_offset=True
    )
    
@configclass
class BaseQuadrupedWheelchairEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: BaseQuadrupedWheelchairObservationsCfg = BaseQuadrupedWheelchairObservationsCfg()
    events: BaseQuadrupedWheelchairEventCfg = BaseQuadrupedWheelchairEventCfg()
    actions: BaseQuadrupedWheelchairActionCfg = BaseQuadrupedWheelchairActionCfg()
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = Quadruped_Wheelchair_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.01, 0.12)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.01, 0.12)
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.12)

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
class BaseQuadrupedWheelchairEnvCfg_PLAY(BaseQuadrupedWheelchairEnvCfg):
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
