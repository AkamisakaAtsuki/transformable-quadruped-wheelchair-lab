import os
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sim import SimulationContext

output_dir = "path/to/output_dir"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

write_buffer = []
BATCH_SIZE = 100 

output_file = os.path.join(output_dir, "observations.csv")

def collect_mdp_data(env: ManagerBasedEnv, env_ids: torch.Tensor):
    observations = env.observation_manager.compute()
    sim_time = SimulationContext.instance().current_time

    global write_buffer
    for idx, env_id in enumerate(env_ids):
        env_id_int = int(env_id.item())
        observation = observations["policy"][idx].cpu().numpy()
        observation_str = "[" + "; ".join(f"{x:.4f}" for x in observation) + "]"
        line = f"{sim_time:.4f},{env_id_int},{observation_str}\n"
        write_buffer.append(line)

    if len(write_buffer) >= BATCH_SIZE:
        if not os.path.exists(output_file):
            with open(output_file, "w") as f:
                f.write("time,env_id,observation\n")
        with open(output_file, "a") as f:
            f.writelines(write_buffer)
        write_buffer = []  # バッファをクリア

    return f"Data saved in {output_file}"
