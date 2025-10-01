import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from transformable_quadruped_wheelchair_isaaclab.utils.rsl_rl.modules.custom_network_actor_critic import CustomNetworkActorCritic

class CustomNetworkPPO(PPO):
    """
    PPO variant that accepts a pre-instantiated CustomNetworkActorCritic.
    """
    def __init__(self, custom_policy: CustomNetworkActorCritic, **ppo_kwargs):
        super().__init__(policy=custom_policy, **ppo_kwargs)