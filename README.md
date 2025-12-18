# transformable-quadruped-wheelchair-lab

<video controls width="640">
  <source src="media/journal-2025-video.mp4" type="video/mp4">
</video>

## インストール
```bash
git clone https://github.com/AkamisakaAtsuki/transformable-quadruped-wheelchair-lab.git
cd transformable-quadruped-wheelchair-lab/TransformableQuadrupedWheelchairIsaacLab
python -m pip install -e exts/transformable_quadruped_wheelchair_isaaclab
```

## 車輪モードと歩行モードの強化学習の実施
### 歩行モードの学習
```bash
python IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task TQW-Walking-Mode-Rl-v0 --num_envs 2048 --max_iteration 20000
```

### 車輪モードの学習
```bash
python IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task TQW-Wheel-Mode-Rl-v0 --num_envs 2048 --max_iteration 20000
```

## 揺れの分析
### データの収集
```bash
python run_wheeled_and_walking_policy_collect_teslabot_positions.py
```
を実行する。この際、内部では、観測情報をそろえるための前処理などを加えてjit化したもでるが使用されている。
歩行モードと車輪モードに対して実行する

### PSD分析
収集したTeslabotの揺れ情報を以下のスクリプトで分析
```bash
sway_analysis.ipynb
```

## 両モデルの入出力次元の統一化
上で学習した歩行モードと車輪モードでは入出力次元数が異なる。しかし、方策蒸留をする上では、入出力次元をそろえたうえでデータ取集をすることが望まれる。そこで、以下のコードを実行して、入出力次元を統一する。
```bash
scripts/export_full_policywk_jit.py 
scripts/export_full_policywh_jit.py
```

## 蒸留向けデータセットの収集
各モードの学習時に使用した環境クラス内に、データ収集オプションがあり、デフォルトではNoneになっているが、そこをコメントアウトすることで、その環境を実行時にそのデータが収集されるようになる。

## 方策蒸留の実行
base_student_policy_distillation.ipynb
を実行

## 評価
isaaclabで評価用の長距離環境を構築している。以下を実行する。各エピソードの移動距離や、かかった時間、ゴールしたかどうか、といった情報を取得できる。
```bash
python IsaacLab\scripts\reinforcement_learning\rsl_rl\play.py --task TQW-Two-Modes-with-ModeVector-v0 --num_envs 100
```