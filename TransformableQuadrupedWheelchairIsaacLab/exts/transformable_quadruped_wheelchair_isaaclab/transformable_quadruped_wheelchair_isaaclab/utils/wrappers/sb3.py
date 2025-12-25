# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to Stable-Baselines3 vectorized environment.

The following example shows how to wrap an environment for Stable-Baselines3:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper

    env = Sb3VecEnvWrapper(env)

"""

# needed to import for allowing type-hinting: torch.Tensor | dict[str, torch.Tensor]
from __future__ import annotations

import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
import torch
import torch.nn as nn  # noqa: F401
from typing import Any

from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class Sb3DiscreteVecEnvWrapper(VecEnv):
    def __init__(self, env: ManagerBasedRLEnv, discrete_map: list[torch.Tensor]):
       
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        self.env = env
        self.discrete_map = discrete_map
        self.sim_device = env.unwrapped.device

        self.n_actions = len(discrete_map)

        self.num_envs = env.unwrapped.num_envs

        obs_space = env.observation_space["policy"]  # ここは4.5.0へバージョンアップさせる際に修正した

        self.action_space = Discrete(self.n_actions)

        VecEnv.__init__(self, self.num_envs, obs_space, self.action_space)

        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.sim_device)

        self._async_actions = None

    def seed(self, seed: int | None = None) -> list[int | None]:
        self.env.unwrapped.seed(seed)
        return [seed] * self.num_envs

    def reset(self) -> VecEnvObs:
        obs_dict, _ = self.env.reset()
        return self._convert_obs(obs_dict)

    def step_async(self, actions: np.ndarray | torch.Tensor) -> None:
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.sim_device)
        self._async_actions = actions  # shape (num_envs,)

    def step_wait(self) -> VecEnvStepReturn:
        if self._async_actions is None:
            raise ValueError("No actions set in step_async.")

        num_envs = self.num_envs
        continuous_actions = []

        if self._async_actions.dim() == 0:
            self._async_actions = self._async_actions.unsqueeze(0)  

        print(f"self._async_actions.dim() : {self._async_actions.dim() }, num_envs: {num_envs}")
        if self._async_actions.dim() == 0 and num_envs == 1:
            discrete_idx = int(self._async_actions.item())
            if not (0 <= discrete_idx < self.n_actions):
                raise ValueError(f"Discrete action {discrete_idx} out of range.")

            cont_vec = self.discrete_map[discrete_idx].to(self.sim_device)
            continuous_actions = cont_vec.unsqueeze(0)  # shape (1, action_dim)
        else:
            continuous_actions_list = []
            for i in range(self.num_envs):
                discrete_idx_tensor = self._async_actions[i] 
                discrete_idx = int(discrete_idx_tensor.item())
                if not (0 <= discrete_idx < self.n_actions):
                    raise ValueError(f"Discrete action {discrete_idx} out of range.")

                cont_vec = self.discrete_map[discrete_idx].to(self.sim_device)
                continuous_actions_list.append(cont_vec.unsqueeze(0))

            continuous_actions = torch.cat(continuous_actions_list, dim=0) 

        obs_dict, rew, terminated, truncated, extras = self.env.step(continuous_actions)

        done = (terminated | truncated).detach()
        done_np = done.cpu().numpy()

        rew_np = rew.detach().cpu().numpy()

        self._ep_rew_buf += rew
        self._ep_len_buf += 1

        reset_ids = (done > 0).nonzero(as_tuple=False)

        obs_np = self._convert_obs(obs_dict)

        infos = [dict() for _ in range(num_envs)]
        for i in range(num_envs):
            if done_np[i]:
                infos[i]["episode"] = {
                    "r": float(self._ep_rew_buf[i].item()),
                    "l": int(self._ep_len_buf[i].item()),
                }
        self._ep_rew_buf[reset_ids] = 0
        self._ep_len_buf[reset_ids] = 0

        return obs_np, rew_np, done_np, infos

    def close(self) -> None:
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        """Optional: SB3 VecEnv API"""
        if indices is None:
            indices = range(self.num_envs)
        results = []
        for _ in indices:
            results.append(getattr(self.env, attr_name, None))
        return results

    def set_attr(self, attr_name, value, indices=None):
        """Optional: SB3 VecEnv API (not implemented)"""
        raise NotImplementedError("set_attr not implemented")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Optional: SB3 VecEnv API (not implemented)"""
        if indices is None:
            indices = range(self.num_envs)
        results = []
        for _ in indices:
            fn = getattr(self.env, method_name)
            results.append(fn(*method_args, **method_kwargs))
        return results

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Optional: SB3 VecEnv API (not implemented)"""
        return [False] * self.num_envs

    def _convert_obs(self, obs_dict) -> np.ndarray:
        obs_torch = obs_dict["policy"]
        obs_np = obs_torch.detach().cpu().numpy()
        return obs_np
