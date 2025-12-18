import os
import json
import torch
import torch.nn as nn

WALKING_MODE_PREFERRED_ANGLES = {
    'LFUpper2_joint': 0.0, 'RFUpper2_joint': 0.0, 'LRUpper2_joint': 0.0, 'RRUpper2_joint': 0.0, 
    'LFTire1_joint': 0.0, 'RFTire1_joint': 0.0, 'LRTire1_joint': 0.0, 'RRTire1_joint': 0.0, 
    'FL_hip_joint': 0.2, 'FR_hip_joint': -0.2, 'RL_hip_joint': 0.2, 'RR_hip_joint': -0.2,       # 以下はweakに制約をかけたほうがよさそうな項目
    'FL_thigh_joint': 0.8, 'FR_thigh_joint': 0.8, 'RL_thigh_joint': 1.0, 'RR_thigh_joint': 1.0, 
    'FL_calf_joint': -1.5, 'FR_calf_joint': -1.5, 'RL_calf_joint': -1.5, 'RR_calf_joint': -1.5, 
}

WHEELED_MODE_PREFERRED_ANGLES = {
    'LFUpper2_joint': 0.0, 'RFUpper2_joint': 0.0, 'LRUpper2_joint': 0.0, 'RRUpper2_joint': 0.0, # 操舵
    # 'LFTire1_joint': 0.0, 'RFTire1_joint': 0.0, 'LRTire1_joint': 0.0, 'RRTire1_joint': 0.0,     # 車輪移動
    'FL_hip_joint': 0.2, 'FR_hip_joint': -0.2, 'RL_hip_joint': 0.2, 'RR_hip_joint': -0.2,       # 以下はweakに制約をかけたほうがよさそうな項目
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
    'FL_hip_joint': 0.1, 'FR_hip_joint': -0.1, 'RL_hip_joint': 0.1, 'RR_hip_joint': -0.1,       # 以下はweakに制約をかけたほうがよさそうな項目
    'FL_thigh_joint': 0.0, 'FR_thigh_joint': 0.0, 'RL_thigh_joint': 0.0, 'RR_thigh_joint': 0.0, 
    'LFUpper2_joint': 0.0, 'RFUpper2_joint': 0.0, 'LRUpper2_joint': 0.0, 'RRUpper2_joint': 0.0, # 操舵
    'FL_calf_joint': -2.0, 'FR_calf_joint': -2.0, 'RL_calf_joint': -2.0, 'RR_calf_joint': -2.0,
}

action_s_idx = 68
action_e_idx = 88
trained_jit_policy_path = "walking_mode_restricted_obs.pt"

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

# class PreprocessPolicyWrapper(nn.Module):
#     def __init__(self, policy_wk_path, action_s_idx, action_e_idx, restricted_action_dim, full_action_dim, joint_index_map_data, walking_joints_data, WALKING_MODE_PREFERRED_ANGLES, device="cpu"):
#         super().__init__()
#         # JITモデルをロード
#         self.policy_wk = torch.jit.load(policy_wk_path, map_location=device)
#         self.action_s_idx = action_s_idx
#         self.action_e_idx = action_e_idx
#         self.full_action_dim = full_action_dim
#         self.joint_index_map_data = joint_index_map_data
#         self.walking_joints_data = walking_joints_data
#         self.WALKING_MODE_PREFERRED_ANGLES = WALKING_MODE_PREFERRED_ANGLES

#         self.prev_full_action_wk = torch.zeros(
#             (1, restricted_action_dim), 
#             dtype=torch.float32, 
#             device=device
#         )

#     def forward(self, obs):
#         # obs: (batch_size, obs_dim)のtensorを想定
#         # 必要な部分だけ抜き出し
#         batch_size = obs.shape[0]
#         _obs = obs.clone()
#         _obs_before = _obs[:, :self.action_s_idx]
#         _obs_after  = _obs[:, self.action_e_idx:]
#         obs_wk = torch.cat([_obs_before, self.prev_full_action_wk, _obs_after], dim=1)

#         # JIT policyへ
#         _actions_wk = self.policy_wk(obs_wk)
#         self.prev_full_action_wk = _actions_wk

#         full_actions_wk = torch.zeros(
#             batch_size, 
#             self.full_action_dim, 
#             device=_actions_wk.device
#         )
            
#         for joint_name, target_idx in joint_index_map_data.items():
#             if joint_name in joint_to_walking_pos:
#                 pos = joint_to_walking_pos[joint_name]
#                 full_actions_wk[:, target_idx] = _actions_wk[:, pos] * 0.1 + self.WALKING_OFFSET[joint_name]
#             else:
#                 full_actions_wk[:, target_idx] = self.WALKING_MODE_PREFERRED_ANGLES[joint_name]
        
        # keep_mask = sorted(joint_index_map_data.values())
        # reduced_actions_wk = full_actions_wk[:, keep_mask]

        # zeros4 = torch.zeros(batch_size, 4, device=_actions_wk.device)
        # combined_actions_wk = torch.cat([reduced_actions_wk, zeros4], dim=1)

#         return combined_actions_wk

import torch
import torch.nn as nn

class PreprocessPolicyWrapper(nn.Module):
    def __init__(
        self,
        policy_wk_path,
        action_s_idx,
        action_e_idx,
        restricted_action_dim,
        full_action_dim,
        walking_action_map,   # Tensor: [full_action_dim] index=-1ならdefault
        walking_action_out_indices,
        walking_offsets,      # Tensor: [full_action_dim]
        walking_offsets_indices,
        walking_defaults,     # Tensor: [full_action_dim]
        keep_mask,
        device="cpu"
    ):
        super().__init__()
        self.policy_wk = torch.jit.load(policy_wk_path, map_location=device)
        self.action_s_idx = action_s_idx
        self.action_e_idx = action_e_idx
        self.full_action_dim = full_action_dim

        self.register_buffer("walking_action_out_indices", walking_action_out_indices)
        self.register_buffer("walking_action_map", walking_action_map)
        self.register_buffer("walking_offsets", walking_offsets)
        self.register_buffer("walking_offsets_indices", walking_offsets_indices)
        self.register_buffer("walking_defaults", walking_defaults)
        self.register_buffer("keep_mask", keep_mask)

        self.prev_full_action_wk = torch.zeros(
            (1, restricted_action_dim), 
            dtype=torch.float32, 
            device=device
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        _obs_before = obs[:, :self.action_s_idx]
        _obs_after  = obs[:, self.action_e_idx:]
        obs_wk = torch.cat([_obs_before, self.prev_full_action_wk.expand(batch_size, -1), _obs_after], dim=1)
        _actions_wk = self.policy_wk(obs_wk)
        self.prev_full_action_wk = _actions_wk

        # 出力
        full_actions_wk = torch.zeros(
            batch_size,
            self.full_action_dim,
            dtype=_actions_wk.dtype,
            device=_actions_wk.device
        )
        # --- Tensor index-based mapping (JIT-compatible)
        # walking_action_map: 各full_actionのindexに対応するrestricted_actionのindex, -1ならdefault
        # for i in range(self.full_action_dim):
        #     idx = int(self.walking_action_map[i].item())
        #     if idx >= 0:
        #         # 学習済みpolicyの出力＋オフセット
        #         full_actions_wk[:, i] = _actions_wk[:, idx] * 0.1 + self.walking_offsets[i]
        #     else:
        #         # preferred angle（デフォルト）
        #         full_actions_wk[:, i] = self.walking_defaults[i]

        full_actions_wk[:, self.walking_offsets_indices] = self.walking_defaults
        full_actions_wk[:, self.walking_action_out_indices] = _actions_wk * 0.1 + self.walking_offsets

        keep_idx = self.keep_mask.unsqueeze(0).expand(batch_size, -1)  # [B, full_dim]
        reduced_actions_wk = full_actions_wk.gather(1, keep_idx) 

        # reduced_actions_wk = full_actions_wk.index_select(1, self.keep_mask)  # [B, len(keep_mask)]
        zeros4 = torch.zeros(batch_size, 4, device=_actions_wk.device)
        combined_actions_wk = torch.cat([reduced_actions_wk, zeros4], dim=1)

        return combined_actions_wk

    
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
# for j, joint_name in enumerate(walking_joints_data):
#     if joint_name in joint_index_map_data:
#         i = joint_index_map_data[joint_name]
#         walking_action_map[i] = j

walking_offsets = torch.tensor([WALKING_OFFSET[joint] for joint in joint_index_map_data.keys()])
walking_offsets_indices = torch.tensor([joint_index_map_data[joint] for joint in WALKING_OFFSET.keys()])
walking_defaults = torch.tensor([WALKING_MODE_PREFERRED_ANGLES[joint] for joint in joint_index_map_data.keys()])

current_file = os.path.abspath(__file__)
base_dir = os.path.dirname(current_file)
trained_jit_policy_full_path = base_dir + "/" + trained_jit_policy_path

restricted_action_dim = len(walking_joints_data)

walking_action_out_indices = torch.tensor(
    [joint_index_map_data[joint_name] for joint_name in walking_joints_data],
    dtype=torch.long
)
walking_offsets = torch.tensor(
    [WALKING_OFFSET[joint_name] for joint_name in walking_joints_data],
    dtype=torch.float
)
walking_defaults = torch.tensor(
    [WALKING_MODE_PREFERRED_ANGLES[joint_name] for joint_name in joint_index_map_data.keys()],
    dtype=torch.float
)

#         for joint_name, target_idx in joint_index_map_data.items():
#             if joint_name in joint_to_walking_pos:
#                 pos = joint_to_walking_pos[joint_name]
#                 full_actions_wk[:, target_idx] = _actions_wk[:, pos] * 0.1 + self.WALKING_OFFSET[joint_name]
#             else:
#                 full_actions_wk[:, target_idx] = self.WALKING_MODE_PREFERRED_ANGLES[joint_name]

keep_mask = torch.tensor(sorted(joint_index_map_data.values()), dtype=torch.long)

print(keep_mask)

wrapper = PreprocessPolicyWrapper(
    policy_wk_path=trained_jit_policy_full_path,
    action_s_idx=68,
    action_e_idx=88,
    restricted_action_dim=restricted_action_dim,
    full_action_dim=full_action_dim,
    # joint_target_idxs,    # Tensor: [full_action_dim] 各joint名が何番目か
    # walking_pos_idxs,
    walking_action_map=walking_action_map,
    walking_action_out_indices=walking_action_out_indices,
    walking_offsets=walking_offsets,
    walking_offsets_indices=walking_offsets_indices,
    walking_defaults=walking_defaults,
    keep_mask = keep_mask,
    device="cpu"
)
# Script化
scripted_policy = torch.jit.script(wrapper)  # できればscript。traceでもOK
save_name = trained_jit_policy_full_path[:-3] + "_full.pt"
torch.jit.save(scripted_policy, save_name)
