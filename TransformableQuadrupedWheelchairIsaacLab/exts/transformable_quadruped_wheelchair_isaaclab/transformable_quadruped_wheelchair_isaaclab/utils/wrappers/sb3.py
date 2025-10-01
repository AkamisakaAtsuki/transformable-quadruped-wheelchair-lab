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
    """
    Isaac Lab (ManagerBasedRLEnv) + 離散アクション(DQN)向けの統合ラッパクラス

    - SB3 から見ると:
        * 離散アクション空間: Discrete(n_discrete)
        * 観測空間        : もとの連続観測 (Box)
        * num_envs        : ManagerBasedRLEnv の並列数

    - 内部 (self.env) は:
        * 連続アクション (Box)
        * 観測 shape: (num_envs, obs_dim)
        * 報酬 shape: (num_envs,)

    - 本クラスで離散→連続変換を行う (discrete_map)。
      step() では離散アクションを受け取り、連続ベクトルに変換して self.env.step() へ。
    """

    def __init__(self, env: ManagerBasedRLEnv, discrete_map: list[torch.Tensor]):
        """
        Args:
            env:
                ManagerBasedRLEnv (num_envs>1 でも OK)
            discrete_map:
                離散→連続アクションへのマッピング表。例えば2値なら
                [
                  torch.tensor([-1.0]),
                  torch.tensor([+1.0])
                ]
                のように並べる。
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        self.env = env
        self.discrete_map = discrete_map
        self.sim_device = env.unwrapped.device

        # 離散アクション数
        self.n_actions = len(discrete_map)

        # === SB3 の VecEnv に必要な情報 ===
        # 1) num_envs
        self.num_envs = env.unwrapped.num_envs

        # 2) observation_space
        #    ManagerBasedRLEnv の single_observation_space["policy"] を想定
        obs_space = env.observation_space["policy"]  # ここは4.5.0へバージョンアップさせる際に修正した

        # 3) action_space: SB3(DQN) に見せるのは Discrete(n)
        self.action_space = Discrete(self.n_actions)

        # 4) 以上を使って VecEnv を初期化
        VecEnv.__init__(self, self.num_envs, obs_space, self.action_space)

        # エピソードリターン / ステップ数 をトラッキング
        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.sim_device)

        # 内部で step_async -> step_wait の流れを再現するためのバッファ
        self._async_actions = None

    def seed(self, seed: int | None = None) -> list[int | None]:
        """環境の seed を設定 (SB3の仕様上、listを返す)"""
        self.env.unwrapped.seed(seed)
        return [seed] * self.num_envs

    def reset(self) -> VecEnvObs:
        """環境をリセットし、観測を返す (SB3は old-gym API で obsのみ返す想定)."""
        # ManagerBasedRLEnv.reset() は (obs_dict, extras) のような返り値になるはず
        obs_dict, _ = self.env.reset()
        return self._convert_obs(obs_dict)

    def step_async(self, actions: np.ndarray | torch.Tensor) -> None:
        """
        SB3から与えられた離散アクションをバッファに保持。
        shapeは (num_envs,) の想定。
        """
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.sim_device)
        self._async_actions = actions  # shape (num_envs,)

    def step_wait(self) -> VecEnvStepReturn:
        """
        保持した離散アクションを連続に変換して内部環境で step し、結果を返す。
        戻り値は (obs, reward, done, infos) の old-gym API 形式。
        """
        if self._async_actions is None:
            raise ValueError("No actions set in step_async.")

        # 離散→連続変換
        #   self._async_actions: shape (num_envs,)
        #   discrete_map[a_i]: shape (action_dim,) か (1,)
        #   => 結果 continuous_actions: shape (num_envs, action_dim)
        num_envs = self.num_envs
        continuous_actions = []

        if self._async_actions.dim() == 0:
            self._async_actions = self._async_actions.unsqueeze(0)  # 0次元なら1次元に変換

        print(f"self._async_actions.dim() : {self._async_actions.dim() }, num_envs: {num_envs}")
        # 場合によっては shape が (num_envs,) ではなく 0次元になることがあるので
        # 必要に応じて flatten しておく (例: shape=(1,)->(1,), shape=()->(1,) など)
        if self._async_actions.dim() == 0 and num_envs == 1:
            discrete_idx = int(self._async_actions.item())
            if not (0 <= discrete_idx < self.n_actions):
                raise ValueError(f"Discrete action {discrete_idx} out of range.")

            cont_vec = self.discrete_map[discrete_idx].to(self.sim_device)
            continuous_actions = cont_vec.unsqueeze(0)  # shape (1, action_dim)
        else:
            # 複数環境の場合
            continuous_actions_list = []
            for i in range(self.num_envs):
                discrete_idx_tensor = self._async_actions[i]  # <= ここで0次元だとエラー
                discrete_idx = int(discrete_idx_tensor.item())
                if not (0 <= discrete_idx < self.n_actions):
                    raise ValueError(f"Discrete action {discrete_idx} out of range.")

                cont_vec = self.discrete_map[discrete_idx].to(self.sim_device)
                continuous_actions_list.append(cont_vec.unsqueeze(0))

            continuous_actions = torch.cat(continuous_actions_list, dim=0)  # shape (num_envs, action_dim)

        # for i in range(num_envs):
        #     discrete_idx = int(self._async_actions[i].item())
        #     # 安全チェック
        #     if not (0 <= discrete_idx < self.n_actions):
        #         raise ValueError(f"Discrete action {discrete_idx} out of range.")
        #     # map
           
        # continuous_actions = torch.cat(continuous_actions, dim=0)  # shape (num_envs, action_dim)

        # 内部環境 step
        obs_dict, rew, terminated, truncated, extras = self.env.step(continuous_actions)

        #  episode 終了チェック
        done = (terminated | truncated).detach()
        done_np = done.cpu().numpy()

        # 報酬を numpy化
        rew_np = rew.detach().cpu().numpy()

        # episode リターン / 長さ 更新
        self._ep_rew_buf += rew
        self._ep_len_buf += 1

        # resetしたenvのインデックス
        reset_ids = (done > 0).nonzero(as_tuple=False)

        # 次の観測 (resetした env は既に reset 後の観測になっている想定)
        obs_np = self._convert_obs(obs_dict)

        # info を env 個ぶん作成
        infos = [dict() for _ in range(num_envs)]
        for i in range(num_envs):
            if done_np[i]:
                infos[i]["episode"] = {
                    "r": float(self._ep_rew_buf[i].item()),
                    "l": int(self._ep_len_buf[i].item()),
                }
        # reset
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
        """
        ManagerBasedRLEnvは通常:
          obs_dict["policy"]: shape (num_envs, obs_dim)  (torch.Tensor)
        を返す想定。
        これを SB3 が想定する (num_envs, obs_dim) の np.ndarray に変換。
        """
        # 1) まず "policy" キーを抜き出し
        obs_torch = obs_dict["policy"]  # shape: (num_envs, obs_dim)
        # 2) to numpy
        obs_np = obs_torch.detach().cpu().numpy()
        return obs_np
