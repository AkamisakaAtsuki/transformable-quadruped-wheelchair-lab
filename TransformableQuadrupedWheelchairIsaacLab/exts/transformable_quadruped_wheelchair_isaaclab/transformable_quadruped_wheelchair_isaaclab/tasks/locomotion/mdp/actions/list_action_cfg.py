from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .list_action import ListAction, BinaryListAction

@configclass
class ListActionCfg(ActionTermCfg):
    asset_name: str = "robot" 
    class_type: type[ActionTerm] = ListAction

    list_length: int = 1
    default_value: int = 0

@configclass
class BinaryListActionCfg(ActionTermCfg):
    asset_name: str = "robot" 
    class_type: type[ActionTerm] = BinaryListAction

    list_length: int = 1
    default_value: int = 0