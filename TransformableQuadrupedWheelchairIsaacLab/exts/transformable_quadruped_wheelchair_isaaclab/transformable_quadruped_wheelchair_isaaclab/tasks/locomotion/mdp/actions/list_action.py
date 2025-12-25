from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
import torch
from typing import Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .list_action_cfg import ListActionCfg, BinaryListActionCfg

class ListAction(ActionTerm):
    cfg: ListActionCfg

    def __init__(self, cfg: ListActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        self._list_actions = torch.full(
            (self.num_envs, self.cfg.list_length),
            self.cfg.default_value,
            dtype=torch.float32,
            device=getattr(env, "device", "cpu")
        )
        self._raw_actions = torch.zeros(
            (self.num_envs, self.cfg.list_length),
            dtype=torch.float32,
            device=getattr(env, "device", "cpu")
        )

    @property
    def action_dim(self) -> int:
        return self.cfg.list_length

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._list_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._list_actions[:] = actions

    def apply_actions(self):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._list_actions.fill_(self.cfg.default_value)
        else:
            self._raw_actions[env_ids] = 0.0
            self._list_actions[env_ids].fill_(self.cfg.default_value)

    def get_list(self) -> torch.Tensor:
        return self._list_actions.clone()

class BinaryListAction(ActionTerm):
    
    cfg: BinaryListActionCfg

    def __init__(self, cfg: BinaryListActionCfg, env):
        super().__init__(cfg, env)
        self.cfg = cfg
        self._binary_actions = torch.full(
            (self.num_envs, self.cfg.list_length),
            self.cfg.default_value,
            dtype=torch.int32,
            device=getattr(env, "device", "cpu")
        )
        self._raw_actions = torch.zeros(
            self.num_envs, self.cfg.list_length,
            dtype=torch.float32,
            device=getattr(env, "device", "cpu")
        )

    @property
    def action_dim(self) -> int:
        return self.cfg.list_length

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._binary_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        binary_actions = (actions > 0.5).int()
        self._binary_actions[:] = binary_actions

    def apply_actions(self):
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._binary_actions.fill_(self.cfg.default_value)
        else:
            self._raw_actions[env_ids] = 0.0
            self._binary_actions[env_ids].fill_(self.cfg.default_value)

    def get_binary_list(self) -> torch.Tensor:
        return self._binary_actions.clone()
