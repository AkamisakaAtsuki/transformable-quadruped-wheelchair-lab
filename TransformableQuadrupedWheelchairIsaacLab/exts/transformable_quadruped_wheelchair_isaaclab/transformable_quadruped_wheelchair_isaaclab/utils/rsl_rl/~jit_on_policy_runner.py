from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.algorithms import PPO

class JITOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", jit_policy=None):
        super().__init__(env, train_cfg, log_dir, device)
        if jit_policy is not None:
            # 元の ActorCritic 部分を無効化
            del self.alg
            # PPO にもたせなおす
            self.alg = PPO(jit_policy, device=device, **self.alg_cfg)
