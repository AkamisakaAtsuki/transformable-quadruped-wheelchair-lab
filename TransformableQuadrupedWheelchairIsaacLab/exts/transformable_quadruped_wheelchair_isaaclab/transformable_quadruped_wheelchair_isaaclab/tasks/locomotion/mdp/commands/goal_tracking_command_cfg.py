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

    class_type: type = GoalTrackingCommand
    
    asset_name: str = MISSING
    goal_positions: List[List[float]] = MISSING  # shape: (num_envs, 3)
    gain: float = 1.0
    max_speed: float = 1.0
    resampling_time_range: tuple[float, float] = (5.0, 5.0)