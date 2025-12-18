# transformable-quadruped-wheelchair-lab

This repository provides research-oriented code to reproduce and evaluate a transformable quadruped wheelchair capable of switching between walking mode and wheeled mode, using large-scale parallel simulations on NVIDIA Isaac Lab.

Specifically, this repository covers:

- Acquisition of locomotion policies for each mode using reinforcement learning (PPO)
- Frequency-domain analysis of passenger acceleration to compare vibration characteristics
- Integration of walking and wheeled policies into a single unified policy via mode-conditioned policy distillation
- Evaluation on long-distance navigation tasks with explicit mode switching

## Installation
```bash
git clone https://github.com/AkamisakaAtsuki/transformable-quadruped-wheelchair-lab.git
cd transformable-quadruped-wheelchair-lab/TransformableQuadrupedWheelchairIsaacLab
python -m pip install -e exts/transformable_quadruped_wheelchair_isaaclab
```
Note: NVIDIA Isaac Lab must be installed in advance.

## Reinforcement Learning for Walking and Wheeled Modes

In this study, walking mode and wheeled mode are trained independently, as their reference postures and support structures differ significantly.

### Walking Mode Training
```bash
python IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task TQW-Walking-Mode-Rl-v0 --num_envs 2048 --max_iteration 20000
```

### Wheeled Mode Training
```bash
python IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task TQW-Wheel-Mode-Rl-v0 --num_envs 2048 --max_iteration 20000
```

## Vibration (Sway) Analysis
### Data Collection
```bash
python run_wheeled_and_walking_policy_collect_teslabot_positions.py
```
This script executes both walking and wheeled policies while collecting passenger motion data.
Internally, JIT-compiled policies with preprocessing steps are used to align observation representations across modes.

The script should be executed separately for walking mode and wheeled mode.

### PSD Analysis
The collected Tesla Bot acceleration data are analyzed using the following notebook:
```bash
sway_analysis.ipynb
```
This performs power spectral density (PSD) analysis to compare vibration characteristics between the two modes.

## Unifying Input/Output Dimensions of Both Policies
The walking and wheeled policies differ in their input/output dimensions due to different controlled joint sets.
However, policy distillation requires a unified representation.

The following scripts export JIT-compiled policies with aligned input/output dimensions:
```bash
scripts/export_full_policywk_jit.py 
scripts/export_full_policywh_jit.py
```

## Dataset Collection for Policy Distillation

Each environment class used during walking and wheeled training includes an optional data collection mode.

By default, this option is set to None.
By uncommenting the relevant section, observation–action pairs will be automatically collected during environment execution and stored as a dataset for distillation.

## Policy Distillation

Run the following notebook to perform mode-conditioned policy distillation:
```bash
base_student_policy_distillation.ipynb
```
This integrates walking and wheeled teacher policies into a single student policy conditioned on a mode vector.

## Evaluation

A dedicated long-distance evaluation environment with explicit mode switching is implemented in Isaac Lab.

Run the following command to evaluate the distilled policy:

```bash
python IsaacLab\scripts\reinforcement_learning\rsl_rl\play.py --task TQW-Two-Modes-with-ModeVector-v0 --num_envs 100
```

This evaluation collects metrics such as:

- Travel distance per episode
- Time to reach the goal
- Goal completion success