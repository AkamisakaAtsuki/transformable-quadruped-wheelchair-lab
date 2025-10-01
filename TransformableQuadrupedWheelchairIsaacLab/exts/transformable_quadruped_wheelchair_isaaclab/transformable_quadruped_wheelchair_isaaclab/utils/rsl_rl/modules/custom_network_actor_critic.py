import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic

class CustomNetworkActorCritic(ActorCritic):
    """
    ActorCritic subclass allowing custom actor, critic networks,
    and external definition of noise std (either as a parameter or module).
    """
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_network: nn.Module,
        critic_network: nn.Module,
        noise_std_param: nn.Parameter = None,
        noise_std_module: nn.Module = None,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        # Initialize base with dummy nets (they will be replaced)
        super().__init__(
            num_actor_obs,
            num_critic_obs,
            num_actions,
            actor_hidden_dims=[1], # Dummy
            critic_hidden_dims=[1], # Dummy
            activation="elu",
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )
        # Replace networks
        self.actor = actor_network
        self.critic = critic_network

        # Setup std: either use provided param/module or default
        self.noise_std_type = noise_std_type
        if noise_std_param is not None:
            # direct parameter usage
            self.std = noise_std_param
        elif noise_std_module is not None:
            # module that produces std from obs
            self.noise_std_module = noise_std_module
        else:
            # default scalar parameter
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")

        # Disable distribution validation
        Normal.set_default_validate_args(False)
        self.distribution = None

    def update_distribution(self, observations: torch.Tensor):
        # compute mean
        mean = self.actor(observations)
        # compute std either via module or parameter
        if hasattr(self, 'noise_std_module'):
            std = self.noise_std_module(observations)
        else:
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            else:
                std = torch.exp(self.log_std).expand_as(mean)
        # create distribution
        self.distribution = Normal(mean, std)

    def forward(self, observations, **kwargs): # ActorCriticクラスではNotImplementedErrorとなるように定義されているが、ここでこのように定義してよいのか？
        self.update_distribution(observations)
        value = self.critic(observations)
        return self.distribution, value

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations, **kwargs):
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def reset(self, dones=None, hidden_states=None):
        pass

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
