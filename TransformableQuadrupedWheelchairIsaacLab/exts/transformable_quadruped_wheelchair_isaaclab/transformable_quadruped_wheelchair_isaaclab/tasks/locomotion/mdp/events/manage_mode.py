from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

mode = None
stop = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reset_mode(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
):
    global mode, stop, device

    if mode == None:  # 最初のリセット時は全環境に対して実施
        mode = torch.zeros(
            (len(env_ids), 1), 
            dtype=torch.float32, 
            device=device
        )
        stop = torch.zeros(
            (len(env_ids), 1), 
            dtype=torch.float32, 
            device=device
        )
    else: # エピソードが終了しリセットがかかった環境に限定して実施
        mode[env_ids] = 0
        stop[env_ids] = 0

def get_mode():
    global mode

    if mode != None:
        return mode

def get_stop():
    global stop

    if stop != None:
        return stop

def update_mode(ids, new_mode):
    global mode

    mode[ids] = new_mode

def update_stop(ids, new_stop):
    global stop

    stop[ids] = new_stop

def print_mode():
    global mode
    print(mode)

def print_stop():
    global stop
    print(stop)
