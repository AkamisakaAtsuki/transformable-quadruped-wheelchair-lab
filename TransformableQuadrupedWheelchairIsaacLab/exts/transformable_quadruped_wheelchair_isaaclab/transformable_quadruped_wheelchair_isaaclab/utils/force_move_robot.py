import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv  

def force_move_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    desired_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:

    asset = env.scene[asset_cfg.name]
    
    asset.write_root_pose_to_sim(desired_pose, env_ids=env_ids)

def force_move_robot_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    desired_pose: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    if len(desired_pose.shape) == 1:
        desired_pose = desired_pose.unsqueeze(0)

    root_states = asset.data.default_root_state[env_ids].clone()
   
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    asset.write_root_pose_to_sim(desired_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)