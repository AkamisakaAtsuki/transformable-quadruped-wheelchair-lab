import os
import torch

def load_policy(model_path: str, device: torch.device) -> torch.jit.ScriptModule:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    try:
        policy = torch.jit.load(model_path, map_location=device)
        policy.to(device)
        policy.eval()
        print(f"[INFO] Policy loaded successfully from {model_path}")
        return policy
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def load_all_policies(base_dir: str, device: torch.device):
    walking_model = os.path.join(base_dir, "walking_mode.pt")
    wheel_model = os.path.join(base_dir, "wheel_mode.pt")
    change_model = os.path.join(base_dir, "change_mode.pt")

    walking_policy = load_policy(walking_model, device)
    wheel_policy = load_policy(wheel_model, device)
    change_policy = load_policy(change_model, device)
    
    return walking_policy, wheel_policy, change_policy
