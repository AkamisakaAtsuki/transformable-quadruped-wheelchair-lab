# transformable-quadruped-wheelchair-lab

## インストール
git clone https://github.com/AkamisakaAtsuki/transformable-quadruped-wheelchair-lab.git
cd transformable-quadruped-wheelchair-lab/TransformableQuadrupedWheelchairIsaacLab
python -m pip install -e exts/transformable_quadruped_wheelchair_isaaclab

車輪モードと歩行モードの強化学習の実施
・isaaclabのパッケージ形式
歩行モードの学習
python IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task TQW-Walking-Mode-Rl-v0 --num_envs 2048 --max_iteration 20000

車輪モードの学習
python IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task TQW-Wheel-Mode-Rl-v0 --num_envs 2048 --max_iteration 20000


揺れの分析
・揺れデータの収集
python run_wheeled_and_walking_policy_collect_teslabot_positions.py
を実行する。この際、内部では、観測情報をそろえるための前処理などを加えてjit化したもでるが使用されている。
歩行モードと車輪モードに対して実行する

・jupyter形式
sway_analysis.ipynb
を実行

蒸留
・収集：python

学習時に使用した環境において、データ収集をするところが、Noneになっていると思うので、そこをコメントアウトしてplayを実行

・学習：jupyter
base_student_policy_distillation.ipynbを実行

評価実験
・isaaclab

python IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task TQW-Two-Modes-with-ModeVector-v0 --num_envs 100
