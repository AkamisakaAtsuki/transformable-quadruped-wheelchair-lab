from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .list_action import ListAction, BinaryListAction

@configclass
class ListActionCfg(ActionTermCfg):
    """リストの行動設定用コンフィグクラス。
    
    Attributes:
        list_length (int): バイナリリストの長さ（要素数）。
        default_value (int): リセット時の初期値。通常は0。
    """
    asset_name: str = "robot" # ダミーで設定（この行動クラスではassetは使用しない）
    class_type: type[ActionTerm] = ListAction

    list_length: int = 1
    default_value: int = 0

@configclass
class BinaryListActionCfg(ActionTermCfg):
    """バイナリリストの行動設定用コンフィグクラス。
    
    Attributes:
        list_length (int): バイナリリストの長さ（要素数）。
        default_value (int): リセット時の初期値。通常は0。
    """
    asset_name: str = "robot" # ダミーで設定（この行動クラスではassetは使用しない）
    class_type: type[ActionTerm] = BinaryListAction

    list_length: int = 1
    default_value: int = 0