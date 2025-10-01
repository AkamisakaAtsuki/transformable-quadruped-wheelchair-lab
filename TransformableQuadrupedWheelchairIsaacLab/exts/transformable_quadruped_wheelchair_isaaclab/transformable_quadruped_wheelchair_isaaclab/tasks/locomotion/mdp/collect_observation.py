import os
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sim import SimulationContext

# 保存先ディレクトリの設定（必要に応じてパスを変更）
output_dir = "path/to/output_dir"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# バッチ書き込み用のバッファ（全環境分のデータを1つにまとめる）
write_buffer = []
BATCH_SIZE = 100  # 一度にファイルに書き込む行数

# 出力ファイルのパス（1つのファイルに全データを記録）
output_file = os.path.join(output_dir, "observations.csv")

def collect_mdp_data(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """
    環境から観測データを取得し、シミュレーション時刻、環境ID、観測データの3要素を1行として
    1つのCSVファイルにバッチ書き込みする関数。
    """
    # 観測データの収集
    observations = env.observation_manager.compute()
    # シミュレーションの現在時刻を取得
    sim_time = SimulationContext.instance().current_time

    global write_buffer
    # 各環境についてデータをバッファに追加
    for idx, env_id in enumerate(env_ids):
        env_id_int = int(env_id.item())
        observation = observations["policy"][idx].cpu().numpy()
        observation_str = "[" + "; ".join(f"{x:.4f}" for x in observation) + "]"
        # 各行はシミュレーション時刻, 環境ID, 観測データとなる
        line = f"{sim_time:.4f},{env_id_int},{observation_str}\n"
        write_buffer.append(line)

    # バッファに溜まった行数が閾値を超えたら、ファイルに書き込み
    if len(write_buffer) >= BATCH_SIZE:
        # ファイルが存在しない場合はヘッダーを追加
        if not os.path.exists(output_file):
            with open(output_file, "w") as f:
                f.write("time,env_id,observation\n")
        # ファイルに追記
        with open(output_file, "a") as f:
            f.writelines(write_buffer)
        write_buffer = []  # バッファをクリア

    return f"Data saved in {output_file}"
