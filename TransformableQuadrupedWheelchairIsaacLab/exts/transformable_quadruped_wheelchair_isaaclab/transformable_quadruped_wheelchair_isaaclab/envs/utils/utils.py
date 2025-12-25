import torch
from isaaclab.assets import Articulation

def load_policy_torch(model_path, device="cuda"):
    policy = torch.jit.load(str(model_path), map_location=device)
    policy.eval()
    return policy

def run_policy_torch(policy, obs_tensor, device="cuda"):
    with torch.no_grad():
        return policy(obs_tensor.to(device))

def post_process_walking_mode():
    pass

def unwrap_env(env):
    raw = env
    while hasattr(raw, "env"):
        raw = raw.env
    return raw

def get_joint_names(env):
    base_env = unwrap_env(env)
    asset: Articulation = base_env.scene["robot"]
    joint_names = asset.joint_names
    return joint_names

def get_joint_idxs_offsets(env, joints, OFFSETS, joint_names, device):
    idxs = []

    base_env = unwrap_env(env)
    offsets = torch.zeros(size=(1, base_env.action_manager.total_action_dim), device=device)
    for _joint in joints:
        _idx = joint_names.index(_joint)
        idxs.append(_idx)

        offsets[:, _idx] = torch.tensor(OFFSETS[_joint], dtype=torch.float)
    
    return idxs, offsets

def get_joint_idxs_velues(env, joints, OFFSETS, joint_names, device):
    idxs = []

    base_env = unwrap_env(env)
    values = torch.zeros(size=(1, base_env.action_manager.total_action_dim), device=device)
    for _joint in joints:
        _idx = joint_names.index(_joint)
        idxs.append(_idx)

        values[:, _idx] = torch.tensor(OFFSETS[_joint], dtype=torch.float)
    
    return idxs, values

def apply_offset(actions, idxs, offsets, scale):
    actions[:, idxs] = actions[:, idxs] * scale + offsets[:, idxs]
    return actions

def apply_fixed_value(actions, idxs, values):
    actions[:, idxs] = values[:, idxs]
    return actions

def action_idx(env): # actionのインデックスを取得する
    base_env = unwrap_env(env)
    
