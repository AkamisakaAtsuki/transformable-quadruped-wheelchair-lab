# mass_utils.py
import torch
from typing import Optional, Union
from isaaclab.assets import Articulation

# ここで ManagerBasedEnv, SceneEntityCfg, Articulation など必要な型をインポートしてください
# 例:
# from your_project.env_manager import ManagerBasedEnv
# from your_project.scene_config import SceneEntityCfg
# from your_project.articulation import Articulation

def get_asset_masses(
    env,
    asset_cfg, 
    env_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    指定された asset_cfg に対応する資産の、対象環境（env_ids）の現在の質量を取得する。

    Args:
        env: マネージャーベースの環境インスタンス
        asset_cfg: 質量を取得する対象の asset 設定（例: SceneEntityCfg("teslabot")）
        env_ids: 対象とする環境IDの torch.Tensor（None の場合は全環境）

    Returns:
        対象 body の質量を含む torch.Tensor（shape: [num_envs, num_bodies]）
    """
    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # 対象の body インデックスを決定
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # 現在の質量を取得
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
    """
    対象の asset_cfg に対応する資産の、指定された環境(env_ids)での質量を new_mass に変更する。
    recompute_inertia=True の場合、慣性テンソルも新しい質量に合わせて再計算する。

    この実装例では、全環境の質量テンソルを取得し、該当する環境・ボディ部分だけを上書きします。
    """
    asset = env.scene[asset_cfg.name]

    # env_ids の解決（全環境対象なら torch.arange を使う）
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # 対象の body インデックスを決定（全ボディなら slice(None) など）
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # 物理エンジン側から全体の質量テンソルを取得（形状例：(num_envs, num_bodies)）
    masses = asset.root_physx_view.get_masses().clone()
    
    # 対象環境の部分だけを変更する
    default_mass = asset.data.default_mass[env_ids[:, None], body_ids].clone()
    if isinstance(new_mass, float):
        new_masses_subset = torch.full_like(default_mass, fill_value=new_mass)
    else:
        new_masses_subset = new_mass
    # masses の対象部分だけ上書き
    masses[env_ids[:, None], body_ids] = new_masses_subset

    # 全環境のインデックスを作成して渡す
    all_env_ids = torch.arange(masses.shape[0], device=masses.device)
    asset.root_physx_view.set_masses(masses, all_env_ids)

    if recompute_inertia:
        ratios = new_masses_subset / default_mass
        inertias = asset.root_physx_view.get_inertias().clone()
        if hasattr(asset.data, "default_inertia"):
            # すでに isaaclab.assets から Articulation をインポートしているのでそれを利用
            if isinstance(asset, Articulation):
                inertias[env_ids[:, None], body_ids] = asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
            else:
                inertias[env_ids[:, None], body_ids] = asset.data.default_inertia[env_ids[:, None], body_ids] * ratios
            asset.root_physx_view.set_inertias(inertias, all_env_ids)

# def set_asset_mass(
#     env,
#     asset_cfg,
#     env_ids: Optional[torch.Tensor],
#     new_mass: Union[float, torch.Tensor],
#     recompute_inertia: bool = True,
# ) -> None:
#     """
#     対象の asset_cfg に対応する資産の、指定された環境(env_ids)での質量を new_mass に変更する。
#     recompute_inertia=True の場合、慣性テンソルも新しい質量に合わせて再計算する。

#     Args:
#         env: マネージャーベースの環境インスタンス
#         asset_cfg: 質量を変更する対象の asset 設定（例: SceneEntityCfg("teslabot")）
#         env_ids: 対象とする環境IDの torch.Tensor（None の場合は全環境）
#         new_mass: 新しい質量。float を指定した場合は全ての body に対して同じ値を設定する。
#                    すでに tensor を渡す場合は、shape が default_mass と同じであることが必要。
#         recompute_inertia: True の場合、慣性テンソルを再計算して設定する。
#     """
#     asset = env.scene[asset_cfg.name]

#     if env_ids is None:
#         env_ids = torch.arange(env.scene.num_envs, device="cpu")
#     else:
#         env_ids = env_ids.cpu()

#     # 対象の body インデックスを決定
#     if asset_cfg.body_ids == slice(None):
#         body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
#     else:
#         body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

#     # デフォルトの質量を元に新しい質量テンソルを作成（変更前の値として利用）
#     default_mass = asset.data.default_mass[env_ids[:, None], body_ids].clone()
#     if isinstance(new_mass, float):
#         new_masses = torch.full_like(default_mass, fill_value=new_mass)
#     else:
#         new_masses = new_mass

#     # 物理シミュレーションに新しい質量を反映
#     asset.root_physx_view.set_masses(new_masses, env_ids)

#     if recompute_inertia:
#         # 新旧質量比を計算
#         ratios = new_masses / default_mass
#         inertias = asset.root_physx_view.get_inertias()
#         if hasattr(asset.data, "default_inertia"):
#             # asset が Articulation か否かで形状が異なる
#             from your_project.articulation import Articulation  # 必要に応じて調整
#             if isinstance(asset, Articulation):
#                 inertias[env_ids[:, None], body_ids] = asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
#             else:
#                 inertias[env_ids] = asset.data.default_inertia[env_ids] * ratios
#             asset.root_physx_view.set_inertias(inertias, env_ids)
