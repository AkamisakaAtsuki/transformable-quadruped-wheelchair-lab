import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

class CustomNetworkOnPolicyRunner(OnPolicyRunner):
    """
    OnPolicyRunner variant that uses CustomNetworkActorCritic and CustomNetworkPPO.
    Allows injecting custom actor/critic networks and std.
    """
    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        custom_actor: nn.Module = None,
        custom_critic: nn.Module = None,
        noise_std_param: nn.Parameter = None,
        noise_std_module: nn.Module = None,
        **ppo_kwargs,
    ):

        # Base initialization up to policy class resolution
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self._configure_multi_gpu()

        # Build custom ActorCritic
        obs, extras = env.get_observations()
        num_obs = obs.shape[1]
        num_priv = extras.get("observations", {}).get(self.privileged_obs_type, obs).shape[1]
        num_actions = env.num_actions
        # Instantiate custom actor and critic if provided
        assert custom_actor is not None and custom_critic is not None, \
            "Must provide custom_actor and custom_critic modules"
        policy = CustomNetworkActorCritic(
            num_obs,
            num_priv,
            num_actions,
            actor_network=custom_actor,
            critic_network=custom_critic,
            noise_std_param=noise_std_param,
            noise_std_module=noise_std_module,
            **self.policy_cfg,
        ).to(self.device)

        # Initialize PPO algorithm with custom policy
        self.alg = CustomNetworkPPO(policy, device=device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # Initialize storage and normalizers
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.alg.init_storage(
            self.training_type,
            env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_priv],
            [env.num_actions],
        )
        # Log setup
        self.log_dir = log_dir
        self.current_learning_iteration = 0

    # learn(), save(), load(), etc. are inherited from OnPolicyRunner
