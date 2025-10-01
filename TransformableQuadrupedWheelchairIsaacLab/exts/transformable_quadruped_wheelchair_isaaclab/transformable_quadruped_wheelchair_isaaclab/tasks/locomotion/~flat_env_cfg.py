# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from .rough_env_cfg import QuadrupedWheelchairRoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg

import transformable_quadruped_wheelchair_isaaclab.tasks.locomotion.mdp as quadruped_wheelchair_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class QuadrupedWheelchairEventCfg(EventCfg):

    # interval
    set_joint_angles = EventTerm(
        func=quadruped_wheelchair_mdp.set_joint_angles,
        mode="interval",
        interval_range_s=(0.1, 0.15),
        params={
            "asset_cfg": SceneEntityCfg("robot")
        },
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"], 
        scale=0.5, 
        use_default_offset=True
    )

@configclass
class QuadrupedWheelchairFlatEnvCfg(QuadrupedWheelchairRoughEnvCfg):

    actions: ActionsCfg = ActionsCfg()
    events: QuadrupedWheelchairEventCfg = QuadrupedWheelchairEventCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class QuadrupedWheelchairFlatEnvCfg_PLAY(QuadrupedWheelchairFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
