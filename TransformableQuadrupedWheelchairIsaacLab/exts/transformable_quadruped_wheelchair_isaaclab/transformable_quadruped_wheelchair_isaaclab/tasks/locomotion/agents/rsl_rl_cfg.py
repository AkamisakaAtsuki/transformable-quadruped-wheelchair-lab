# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlPpoAlgorithmCfg,
    RslRlDistillationAlgorithmCfg,
)

# walking mode rl env
@configclass
class QuadrupedWheelchairWalkingModeRlPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "quadruped_wheelchair_walking_mode_rl"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

# wheel mode rl env
@configclass
class QuadrupedWheelchairWheelModeRlPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "quadruped_wheelchair_wheel_mode_rl"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

# walking mode collect data env

# wheel mode collect data env

# mode change rule env 

# mode change rl env
# wheel mode rl env
@configclass
class QuadrupedWheelchairChangeModeRlPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "quadruped_wheelchair_change_mode_rl"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )




@configclass
class QuadrupedWheelchairRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "quadruped_wheelchair_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class QuadrupedWheelchairFlatPPORunnerCfg(QuadrupedWheelchairRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "quadruped_wheelchair_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
    
@configclass
class QuadrupedWheelchairModeChangePPORunnerCfg(QuadrupedWheelchairRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "quadruped_wheelchair_mode_change"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]

@configclass
class QuadrupedWheelchairTwoModesPPORunnerCfg(QuadrupedWheelchairRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "tqw_two_modes"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]

@configclass
class QuadrupedWheelchairTwoModesDistillPPORunnerCfg(QuadrupedWheelchairRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1000
        self.experiment_name = "distill_additional_learning"
        self.policy.actor_hidden_dims = [128] # dummy
        self.policy.critic_hidden_dims = [128] # dummy

@configclass
class QuadrupedWheelchairStudentRoughPPORunnerCfg(QuadrupedWheelchairRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = 64
        self.max_iterations = 300
        self.experiment_name = "tqw_student"
        self.teacher_experiment_name = "tqw_student"
        self.policy = RslRlDistillationStudentTeacherCfg(
            init_noise_std= 0.001,
            student_hidden_dims = [512, 256, 128],
            teacher_hidden_dims = [512, 256, 128],
            activation="elu"
        )
        self.algorithm = RslRlDistillationAlgorithmCfg(
            num_learning_epochs=5,
            learning_rate=1e-03,
            gradient_length=2.
        )

@configclass
class QuadrupedWheelchairTeacherRoughPPORunnerCfg(QuadrupedWheelchairRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 30000
        self.experiment_name = "tqw_teacher"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]

@configclass
class QuadrupedWheelchairTwoModesWithModeVectorPPORunnerCfg(QuadrupedWheelchairRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "tqw_two_modes_with_modeVec"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]