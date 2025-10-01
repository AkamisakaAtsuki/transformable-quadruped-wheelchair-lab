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
    """
    ジョイントに依存せず、外部からアクセス可能な「リスト（実数値）」の行動クラス。

    - このクラスは、入力された実数値のテンソルをそのまま内部に保持し、
      他の関数やクラスから get_list() によって取得できるようにしています。
    - ※ このアクションはロボットの関節には影響を与えず、
      単に内部で「リスト（実数値）」として保持するだけのものです。
    """

    cfg: ListActionCfg

    def __init__(self, cfg: ListActionCfg, env: ManagerBasedEnv):
        """
        Args:
            cfg: ListActionCfg のインスタンス。
            env: ManagerBasedEnv のインスタンス（環境）。
        """
        super().__init__(cfg, env)
        self.cfg = cfg
        # env.num_envs がなければ 1 とみなす
        # self.num_envs = getattr(env, "num_envs", 1)

        # リスト（実数値）の初期テンソルを用意
        self._list_actions = torch.full(
            (self.num_envs, self.cfg.list_length),
            self.cfg.default_value,
            dtype=torch.float32,
            device=getattr(env, "device", "cpu")
        )
        # 生の入力値を記録するためのテンソル（processed_actionsとは別にしてもOK）
        # 今回は同じでも問題なければ省略可能
        self._raw_actions = torch.zeros(
            (self.num_envs, self.cfg.list_length),
            dtype=torch.float32,
            device=getattr(env, "device", "cpu")
        )

    @property
    def action_dim(self) -> int:
        """このアクションの次元数（list_length）。"""
        return self.cfg.list_length

    @property
    def raw_actions(self) -> torch.Tensor:
        """外部から取得可能な「生のアクション値」のバッファ。"""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """外部から取得可能な「処理後アクション値」のバッファ。ここでは同じ。"""
        return self._list_actions

    def process_actions(self, actions: torch.Tensor):
        """
        入力されたアクションを処理して内部に保持する。

        このクラスでは単純に actions をそのまま格納する。
        """
        # 入力値を保存（生の値）
        self._raw_actions[:] = actions
        # 何らかの処理を入れたければここで行うが、今回はそのまま格納
        self._list_actions[:] = actions

    def apply_actions(self):
        """このアクションは実際にジョイント等を動かすものではないので、ここでは何もしません。"""
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """
        リセット時の処理。指定された環境ID、または全環境でアクション値を初期値に戻す。
        """
        if env_ids is None:
            self._raw_actions.zero_()
            self._list_actions.fill_(self.cfg.default_value)
        else:
            self._raw_actions[env_ids] = 0.0
            self._list_actions[env_ids].fill_(self.cfg.default_value)

    def get_list(self) -> torch.Tensor:
        """
        外部からアクセス可能なリスト（実数値）を返す。

        戻り値はクローン（コピー）されるため、外部での変更は内部状態に影響しません。
        """
        return self._list_actions.clone()

class BinaryListAction(ActionTerm):
    """ジョイントに依存せず、外部からアクセス可能なバイナリリストの行動クラス。

    このクラスは、入力された実数値のテンソルを受け取り、
    閾値（例として0.5）を用いて各要素を0または1に変換します。
    また、他の関数やクラスから get_binary_list() によってバイナリリストを取得できるようにしています。
    
    ※このアクションはロボットの関節には影響を与えず、
    単に内部でバイナリのリストとして保持するだけのものです。
    """
    
    cfg: BinaryListActionCfg

    def __init__(self, cfg: BinaryListActionCfg, env):
        # ActionTerm の初期化（env には、num_envs などの属性があることを前提とします）
        super().__init__(cfg, env)
        self.cfg = cfg
        # 環境ごとの数。env に num_envs 属性がない場合は1と仮定します。
        # self.num_envs = getattr(env, "num_envs", 1)
        # バイナリリストの初期テンソル（整数型）を用意
        self._binary_actions = torch.full(
            (self.num_envs, self.cfg.list_length),
            self.cfg.default_value,
            dtype=torch.int32,
            device=getattr(env, "device", "cpu")
        )
        # 生の入力値を記録するためのテンソル
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
        """入力されたアクションを処理してバイナリリストに変換する。
        
        閾値を 0.5 として、各要素が 0.5 を超えるなら 1、そうでなければ 0 としています。
        """
        # 入力値を保存
        self._raw_actions[:] = actions
        # 0.5を閾値にしてバイナリに変換（torch.int() にキャスト）
        binary_actions = (actions > 0.5).int()
        self._binary_actions[:] = binary_actions

    def apply_actions(self):
        """このアクションは実際にジョイント等を動かすものではないので、ここでは何もしません。"""
        pass

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """リセット時の処理。指定された環境ID、または全環境で入力値とバイナリリストを初期値に戻す。"""
        if env_ids is None:
            self._raw_actions.zero_()
            self._binary_actions.fill_(self.cfg.default_value)
        else:
            self._raw_actions[env_ids] = 0.0
            self._binary_actions[env_ids].fill_(self.cfg.default_value)

    def get_binary_list(self) -> torch.Tensor:
        """外部からアクセス可能なバイナリリストを返す。
        
        戻り値はクローン（コピー）されるため、外部での変更は内部状態に影響しません。
        """
        return self._binary_actions.clone()
