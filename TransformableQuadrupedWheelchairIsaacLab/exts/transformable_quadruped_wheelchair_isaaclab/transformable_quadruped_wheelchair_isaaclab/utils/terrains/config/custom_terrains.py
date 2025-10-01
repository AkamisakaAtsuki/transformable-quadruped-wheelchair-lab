# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

MODE_CHANGE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.00, 0.04), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.01, 0.1), platform_width=2.0
        ),
        "hf_pyramid_slope_2": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2, 
            step_height_range=(0.01, 0.15),  # Max 0.23 (Default)
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope_inv_2": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.01, 0.15),  # Max 0.23 (Default)
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

# ROUGH_TERRAINS_IMPROVED_CFG = TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=20.0,
#     num_rows=10,
#     num_cols=20,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     use_cache=False,
#     sub_terrains={
#         "pyramid_stairs_1": terrain_gen.MeshPyramidStairsTerrainCfg(
#             proportion=0.2,
#             step_height_range=(0.01, 0.23),
#             step_width=0.3,
#             platform_width=3.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "pyramid_stairs_inv_1": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.2,
#             step_height_range=(0.01, 0.23),
#             step_width=0.3,
#             platform_width=3.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "boxes_1": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=0.2, grid_width=0.45, grid_height_range=(0.01, 0.2), platform_width=2.0
#         ),
#         "random_rough_1": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
#         ),
#         "hf_pyramid_slope_1": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
#         ),
#         "hf_pyramid_slope_inv_1": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
#             proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
#         ),
#         "pyramid_stairs_2": terrain_gen.MeshPyramidStairsTerrainCfg(
#             proportion=0.2,
#             step_height_range=(0.01, 0.23),
#             step_width=0.3,
#             platform_width=3.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "pyramid_stairs_inv_2": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.2,
#             step_height_range=(0.01, 0.23),
#             step_width=0.3,
#             platform_width=3.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "boxes_2": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=0.2, grid_width=0.45, grid_height_range=(0.01, 0.2), platform_width=2.0
#         ),
#         "random_rough_2": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
#         ),
#         "hf_pyramid_slope_2": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
#         ),
#         "hf_pyramid_slope_inv_2": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
#             proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
#         ),
#     },
# )
# """Rough terrains configuration."""

# PYRAMID_STAIRS_CFG = terrain_gen.MeshPyramidStairsTerrainCfg(
#     proportion=0.2,
#     step_height_range=(0.01, 0.23),
#     step_width=0.3,
#     platform_width=3.0,
#     border_width=1.0,
#     holes=False,
# )

# PYRAMID_STAIRS_INV_CFG = terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#     proportion=0.2,
#     step_height_range=(0.01, 0.23),
#     step_width=0.3,
#     platform_width=3.0,
#     border_width=1.0,
#     holes=False,
# )

# # 階段のみの地形環境を用意
# PYRAMIS_STAIR_ONLY_CFG = TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=20.0,
#     num_rows=10,
#     num_cols=20,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     use_cache=False,
#     sub_terrains={
#         "pyramid_stairs": PYRAMID_STAIRS_CFG,
#         "pyramid_stairs_inv": PYRAMID_STAIRS_INV_CFG,
#     },
# )

# PYRAMIS_STAIR_ONLY_IMPROVED_CFG = TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=20.0,
#     num_rows=10,
#     num_cols=20,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     use_cache=False,
#     sub_terrains={
#         "pyramid_stairs_1": PYRAMID_STAIRS_CFG,
#         "pyramid_stairs_inv_1": PYRAMID_STAIRS_INV_CFG,
#         "pyramid_stairs_2": PYRAMID_STAIRS_CFG,
#         "pyramid_stairs_inv_2": PYRAMID_STAIRS_INV_CFG,
#         "pyramid_stairs_3": PYRAMID_STAIRS_CFG,
#         "pyramid_stairs_inv_3": PYRAMID_STAIRS_INV_CFG,
#         "pyramid_stairs_4": PYRAMID_STAIRS_CFG,
#         "pyramid_stairs_inv_4": PYRAMID_STAIRS_INV_CFG,
#         "pyramid_stairs_5": PYRAMID_STAIRS_CFG,
#         "pyramid_stairs_inv_5": PYRAMID_STAIRS_INV_CFG,
#         "pyramid_stairs_6": PYRAMID_STAIRS_CFG,
#         "pyramid_stairs_inv_6": PYRAMID_STAIRS_INV_CFG,
#         "pyramid_stairs_7": PYRAMID_STAIRS_CFG,
#         "pyramid_stairs_inv_7": PYRAMID_STAIRS_INV_CFG,
#         "pyramid_stairs_8": PYRAMID_STAIRS_CFG,
#         "pyramid_stairs_inv_8": PYRAMID_STAIRS_INV_CFG,
#         "pyramid_stairs_9": PYRAMID_STAIRS_CFG,
#         "pyramid_stairs_inv_9": PYRAMID_STAIRS_INV_CFG,
#         "pyramid_stairs_10": PYRAMID_STAIRS_CFG,
#         "pyramid_stairs_inv_10": PYRAMID_STAIRS_INV_CFG,
#     },
# )