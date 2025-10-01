# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
# from . flat_env_cfg, rough_env_cfg, mode_change_env_cfg
from . import walking_mode_rl_env_cfg, walking_mode_collect_data_env_cfg
from . import wheel_mode_rl_env_cfg, wheel_mode_collect_data_env_cfg
from . import two_modes_change_env_cfg
from . import two_modes_env_cfg

##
# Register Gym environments.
##


# walking mode rl env
gym.register(
    id="TQW-Walking-Mode-Rl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walking_mode_rl_env_cfg.QuadrupedWheelchairWalkingModeRlEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:QuadrupedWheelchairWalkingModeRlPPORunnerCfg",
    },
)
gym.register(
    id="TQW-Walking-Mode-Rl-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walking_mode_rl_env_cfg.QuadrupedWheelchairWalkingModeRlEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:QuadrupedWheelchairWalkingModeRlPPORunnerCfg",
    },
)

# # wheel mode rl env
gym.register(
    id="TQW-Wheel-Mode-Rl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheel_mode_rl_env_cfg.QuadrupedWheelchairWheelModeRlEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:QuadrupedWheelchairWheelModeRlPPORunnerCfg",
    },
)
gym.register(
    id="TQW-Wheel-Mode-Rl-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheel_mode_rl_env_cfg.QuadrupedWheelchairWheelModeRlEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:QuadrupedWheelchairWheelModeRlPPORunnerCfg",
    },
)

# walking mode collect data env

# wheel mode collect data env

# mode change rule env 
gym.register(
    id="TQW-Change-Mode-Rule-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:QuadrupedWheelchairChangeModeRlPPORunnerCfg",
        "env_cfg_entry_point": two_modes_change_env_cfg.QuadrupedWheelchairTwoModesChangeEnv_Normal_Cfg,
        "sb3_cfg_entry_point":  f"{agents.__name__}:sb3_mode_change_rule.yaml",
    },
)
gym.register(
    id="TQW-Change-Mode-Rule-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": two_modes_change_env_cfg.QuadrupedWheelchairTwoModesChangeEnv_Normal_Cfg_PLAY,
        "sb3_cfg_entry_point":  f"{agents.__name__}:sb3_mode_change_rule.yaml",
    },
)

# mode change rl env


# two modes rl env
gym.register(
    id="TQW-Two-Modes-Rl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": two_modes_env_cfg.QuadrupedWheelchairTwoModesEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:QuadrupedWheelchairTwoModesPPORunnerCfg",
    },
)

gym.register(
    id="TQW-Two-Modes-Normal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": two_modes_env_cfg.QuadrupedWheelchairTwoModesEnv_Normal_Cfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:QuadrupedWheelchairTwoModesDistillPPORunnerCfg",
    },
)

# 蒸留モデルを作成するための実験環境
gym.register(
    id="TQW-Two-Modes-Distillation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": two_modes_env_cfg.QuadrupedWheelchairTwoModesEnv_Normal_Cfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:QuadrupedWheelchairStudentRoughPPORunnerCfg",
    },
)

# 変形実験をするためにモードベクトルを追加したもの
gym.register(
    id="TQW-Two-Modes-with-ModeVector-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": two_modes_env_cfg.QuadrupedWheelchairTwoModesWithModeVectorEnv_Normal_Cfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:QuadrupedWheelchairTwoModesWithModeVectorPPORunnerCfg",
    },
)