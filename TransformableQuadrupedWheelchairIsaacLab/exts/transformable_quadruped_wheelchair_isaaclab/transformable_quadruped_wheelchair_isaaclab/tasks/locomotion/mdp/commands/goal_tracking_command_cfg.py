import math
from dataclasses import dataclass, field
from typing import Sequence
from dataclasses import MISSING

from typing import List
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from .goal_tracking_command import GoalTrackingCommand

@configclass
class GoalTrackingCommandCfg(CommandTermCfg):
    """
    ロボットの現在位置と各環境のゴール位置との差分から、移動方向および向きを決定するコマンド生成の設定クラス。
    
    Attributes:
        asset_name: ロボットアセットの名前。
        goal_positions: 各環境（カリキュラム）のゴール位置を格納したテンソル。形状は (num_envs, 3) 。
        gain: 差分に対する比例ゲイン。デフォルトは 1.0。
        max_speed: 各軸での最大速度（絶対値）。デフォルトは 1.0 m/s。
        resampling_time_range: コマンドの再サンプリングの時間範囲。
    """
    # ここで class_type を指定する
    class_type: type = GoalTrackingCommand
    
    asset_name: str = MISSING
    goal_positions: List[List[float]] = MISSING  # shape: (num_envs, 3)
    gain: float = 1.0
    max_speed: float = 1.0
    resampling_time_range: tuple[float, float] = (5.0, 5.0)