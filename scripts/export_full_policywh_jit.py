import os
import json
import torch
import torch.nn as nn

WALKING_MODE_PREFERRED_ANGLES = {
    'LFUpper2_joint': 0.0, 'RFUpper2_joint': 0.0, 'LRUpper2_joint': 0.0, 'RRUpper2_joint': 0.0, 
    'LFTire1_joint': 0.0, 'RFTire1_joint': 0.0, 'LRTire1_joint': 0.0, 'RRTire1_joint': 0.0, 
    'FL_hip_joint': 0.2, 'FR_hip_joint': -0.2, 'RL_hip_joint': 0.2, 'RR_hip_joint': -0.2,       
    'FL_thigh_joint': 0.8, 'FR_thigh_joint': 0.8, 'RL_thigh_joint': 1.0, 'RR_thigh_joint': 1.0, 
    'FL_calf_joint': -1.5, 'FR_calf_joint': -1.5, 'RL_calf_joint': -1.5, 'RR_calf_joint': -1.5, 
}

WHEELED_MODE_PREFERRED_ANGLES = {
    'LFUpper2_joint': 0.0, 'RFUpper2_joint': 0.0, 'LRUpper2_joint': 0.0, 'RRUpper2_joint': 0.0, # 操舵
    # 'LFTire1_joint': 0.0, 'RFTire1_joint': 0.0, 'LRTire1_joint': 0.0, 'RRTire1_joint': 0.0,     # 車輪移動
    'FL_hip_joint': 0.2, 'FR_hip_joint': -0.2, 'RL_hip_joint': 0.2, 'RR_hip_joint': -0.2,       
    'FL_thigh_joint': 0.0, 'FR_thigh_joint': 0.0, 'RL_thigh_joint': 0.0, 'RR_thigh_joint': 0.0, 
    'FL_calf_joint': -2.0, 'FR_calf_joint': -2.0, 'RL_calf_joint': -2.0, 'RR_calf_joint': -2.0,
}

WALKING_OFFSET = { # 中間のオフセットに対応したバージョン
    'FL_hip_joint': 0.0, 'FR_hip_joint': 0.0, 'RL_hip_joint': 0.0, 'RR_hip_joint': 0.0,
    'FL_thigh_joint': 0.8, 'FR_thigh_joint': 0.8, 'RL_thigh_joint': 1.0, 'RR_thigh_joint': 1.0, 
    'LFUpper2_joint': 0.0, 'RFUpper2_joint': 0.0, 'LRUpper2_joint': 0.0, 'RRUpper2_joint': 0.0, 
    'FL_calf_joint': -1.5, 'FR_calf_joint': -1.5, 'RL_calf_joint': -1.5, 'RR_calf_joint': -1.5, 
}

WHEEL_OFFSET = {
    'FL_hip_joint': 0.1, 'FR_hip_joint': -0.1, 'RL_hip_joint': 0.1, 'RR_hip_joint': -0.1,      
    'FL_thigh_joint': 0.0, 'FR_thigh_joint': 0.0, 'RL_thigh_joint': 0.0, 'RR_thigh_joint': 0.0, 
    'LFUpper2_joint': 0.0, 'RFUpper2_joint': 0.0, 'LRUpper2_joint': 0.0, 'RRUpper2_joint': 0.0, # 操舵
    'FL_calf_joint': -2.0, 'FR_calf_joint': -2.0, 'RL_calf_joint': -2.0, 'RR_calf_joint': -2.0,
}

action_s_idx = 68
action_e_idx = 88
trained_jit_policy_path = "wheeled_mode_restricted_obs.pt"

def load_joint_meta():
    current_file = os.path.abspath(__file__)
    base_dir = os.path.dirname(current_file)
    parent_dir = os.path.dirname(base_dir)
    
    joint_index_map_path = os.path.join(parent_dir, "meta", "joint_index_map.json")
    walking_joints_path   = os.path.join(parent_dir, "meta", "walking_joints.json")
    wheeled_joints_path   = os.path.join(parent_dir, "meta", "wheeled_joints.json")
    
    with open(joint_index_map_path, "r", encoding="utf-8") as f:
        joint_index_map_data = json.load(f)
    with open(walking_joints_path, "r", encoding="utf-8") as f:
        walking_joints_data = json.load(f)
    with open(wheeled_joints_path, "r", encoding="utf-8") as f:
        wheeled_joints_data = json.load(f)
    
    return joint_index_map_data, walking_joints_data, wheeled_joints_data

class PreprocessPolicyWrapper(nn.Module):
    def __init__(
        self,
        policy_wh_path,
        action_s_idx,
        action_e_idx,
        restricted_action_dim,
        full_action_dim,
        wheeled_action_map,   # Tensor: [full_action_dim] index=-1ならdefault
        wheeled_action_out_indices,
        wheeled_offsets,      # Tensor: [full_action_dim]
        wheeled_offsets_indices,
        wheeled_defaults,     # Tensor: [full_action_dim]
        keep_mask,
        device="cpu"
    ):
        super().__init__()
        self.policy_wh = torch.jit.load(policy_wh_path, map_location=device)
        self.action_s_idx = action_s_idx
        self.action_e_idx = action_e_idx
        self.full_action_dim = full_action_dim

        self.register_buffer("wheeled_action_out_indices", wheeled_action_out_indices)
        self.register_buffer("wheeled_action_map", wheeled_action_map)
        self.register_buffer("wheeled_offsets", wheeled_offsets)
        self.register_buffer("wheeled_offsets_indices", wheeled_offsets_indices)
        self.register_buffer("wheeled_defaults", wheeled_defaults)
        self.register_buffer("keep_mask", keep_mask)

        self.prev_full_action_wh = torch.zeros(
            (1, restricted_action_dim + 4),  # +4をしているのはタイヤの回転を考慮するため 
            dtype=torch.float32, 
            device=device
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        _obs_before = obs[:, :self.action_s_idx]
        _obs_after  = obs[:, self.action_e_idx:]
        obs_wh = torch.cat([_obs_before, self.prev_full_action_wh.expand(batch_size, -1), _obs_after], dim=1)
        _actions_wh = self.policy_wh(obs_wh)
        self.prev_full_action_wh = _actions_wh

        # 出力
        full_actions_wh = torch.zeros(
            batch_size,
            self.full_action_dim,
            dtype=_actions_wh.dtype,
            device=_actions_wh.device
        )

        full_actions_wh[:, self.wheeled_offsets_indices] = self.wheeled_defaults
        full_actions_wh[:, self.wheeled_action_out_indices] = _actions_wh[:,:-4] * 0.05 + self.wheeled_offsets

        keep_idx = self.keep_mask.unsqueeze(0).expand(batch_size, -1)  # [B, full_dim]
        reduced_actions_wh = full_actions_wh.gather(1, keep_idx) 

        # reduced_actions_wk = full_actions_wk.index_select(1, self.keep_mask)  # [B, len(keep_mask)]
        combined_actions_wh = torch.cat([reduced_actions_wh, _actions_wh[:,-4:]], dim=1)

        return combined_actions_wh

    
joint_index_map_data, walking_joints_data, wheeled_joints_data = load_joint_meta()

walking_joints_data = sorted(
    walking_joints_data,
    key=lambda name: joint_index_map_data[name]
)
wheeled_joints_data = sorted(
    wheeled_joints_data,
    key=lambda name: joint_index_map_data[name]
)
joint_to_walking_pos = { name: i for i, name in enumerate(walking_joints_data) }
joint_to_wheeled_pos = { name: i for i, name in enumerate(wheeled_joints_data) }

full_action_dim = 28
walking_action_map = torch.full((full_action_dim,), -1, dtype=torch.long)
wheeled_action_map = torch.full((full_action_dim,), -1, dtype=torch.long)
# for j, joint_name in enumerate(walking_joints_data):
#     if joint_name in joint_index_map_data:
#         i = joint_index_map_data[joint_name]
#         walking_action_map[i] = j

# walking_offsets = torch.tensor([WALKING_OFFSET[joint] for joint in joint_index_map_data.keys()])
# walking_offsets_indices = torch.tensor([joint_index_map_data[joint] for joint in WALKING_OFFSET.keys()])
# walking_defaults = torch.tensor([WALKING_MODE_PREFERRED_ANGLES[joint] for joint in joint_index_map_data.keys()])

# wheeled_offsets = torch.tensor([WHEELED_OFFSET[joint] for joint in joint_index_map_data.keys()])
wheeled_offsets_indices = torch.tensor([joint_index_map_data[joint] for joint in WHEEL_OFFSET.keys()])
# wheeled_defaults = torch.tensor([WHEELED_MODE_PREFERRED_ANGLES[joint] for joint in joint_index_map_data.keys()])

current_file = os.path.abspath(__file__)
base_dir = os.path.dirname(current_file)
trained_jit_policy_full_path = base_dir + "/" + trained_jit_policy_path

restricted_action_dim = len(wheeled_joints_data)

wheeled_action_out_indices = torch.tensor(
    [joint_index_map_data[joint_name] for joint_name in wheeled_joints_data],
    dtype=torch.long
)
wheeled_offsets = torch.tensor(
    [WHEEL_OFFSET[joint_name] for joint_name in wheeled_joints_data],
    dtype=torch.float
)
wheeled_defaults = torch.tensor(
    [WHEELED_MODE_PREFERRED_ANGLES[joint_name] for joint_name in joint_index_map_data.keys()],
    dtype=torch.float
)

keep_mask = torch.tensor(sorted(joint_index_map_data.values()), dtype=torch.long)

print(keep_mask)

wrapper = PreprocessPolicyWrapper(
    policy_wh_path=trained_jit_policy_full_path,
    action_s_idx=68,
    action_e_idx=88,
    restricted_action_dim=restricted_action_dim,
    full_action_dim=full_action_dim,
    wheeled_action_map=wheeled_action_map,
    wheeled_action_out_indices=wheeled_action_out_indices,
    wheeled_offsets=wheeled_offsets,
    wheeled_offsets_indices=wheeled_offsets_indices,
    wheeled_defaults=wheeled_defaults,
    keep_mask = keep_mask,
    device="cpu"
)
# Script化
scripted_policy = torch.jit.script(wrapper)  # できればscript。traceでもOK
save_name = trained_jit_policy_full_path[:-3] + "_full.pt"
torch.jit.save(scripted_policy, save_name)

