# mass_utils.py
import torch
from typing import Optional, Union
from isaaclab.assets import Articulation

def get_asset_masses(
    env,
    asset_cfg, 
    env_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
   
    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    masses = asset.root_physx_view.get_masses()
    current_masses = masses[env_ids[:, None], body_ids].clone()
    return current_masses

def set_asset_mass(
    env,
    asset_cfg,
    env_ids: Optional[torch.Tensor],
    new_mass: Union[float, torch.Tensor],
    recompute_inertia: bool = True,
) -> None:
   
    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    masses = asset.root_physx_view.get_masses().clone()
    
    default_mass = asset.data.default_mass[env_ids[:, None], body_ids].clone()
    if isinstance(new_mass, float):
        new_masses_subset = torch.full_like(default_mass, fill_value=new_mass)
    else:
        new_masses_subset = new_mass
   
    masses[env_ids[:, None], body_ids] = new_masses_subset

    all_env_ids = torch.arange(masses.shape[0], device=masses.device)
    asset.root_physx_view.set_masses(masses, all_env_ids)

    if recompute_inertia:
        ratios = new_masses_subset / default_mass
        inertias = asset.root_physx_view.get_inertias().clone()
        if hasattr(asset.data, "default_inertia"):
            
            if isinstance(asset, Articulation):
                inertias[env_ids[:, None], body_ids] = asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
            else:
                inertias[env_ids[:, None], body_ids] = asset.data.default_inertia[env_ids[:, None], body_ids] * ratios
            asset.root_physx_view.set_inertias(inertias, all_env_ids)