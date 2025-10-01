import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic
from rsl_rl.utils import resolve_nn_activation

class JITActorCritic(ActorCritic):
    # 再帰型でないことを明示
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=[1],          # ダミー
        critic_hidden_dims=[1],         # ダミー
        activation="relu",              # ダミー
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        # 親の __init__ は呼ばず、必要な属性だけ初期化
        super().__init__(num_actor_obs, num_critic_obs, num_actions,
                         actor_hidden_dims=[1], critic_hidden_dims=[1],
                         activation="relu", init_noise_std=init_noise_std,
                         noise_std_type=noise_std_type)
        # JIT モデルをロード
        # （device は kwargs または CPU を想定）
        device = kwargs.get("device", "cpu")
        self.script: torch.jit.ScriptModule = torch.jit.load(
            "/path/to/distilled_policy_kl_std_jit2.pt",
            map_location=device
        )
        self.script.eval()

        # std 用のパラメータを用意
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions, device=device))
        else:  # "log"
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions, device=device)))

        # 分布用のフラグ
        Normal.set_default_validate_args(False)
        self.distribution = None

    def forward(self, observations, hidden_states=None, seq_lens=None):
        # ActorCritic の forward は未実装なので、ScriptModule を直接使う
        # ScriptModule は mean のみ返す想定
        mean: torch.Tensor = self.script(observations)
        # std を計算
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        # 分布を作成
        self.distribution = Normal(mean, std)
        # value head がないので 0 を返す
        value = torch.zeros(mean.shape[0], device=mean.device)
        return self.distribution, value

    # 以下は ActorCritic と同じＩ/F を提供
    def reset(self, dones=None, hidden_states=None):
        pass

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        # 決定的に mean を返す
        return self.script(observations)

    def evaluate(self, critic_observations, **kwargs):
        # value head がないので 0 を返す
        return torch.zeros(critic_observations.shape[0], device=critic_observations.device)

    def load_state_dict(self, state_dict, strict=True):
        # 学習済み JIT モデルではなく、PPO 続行時の再開用なら
        # ActorCritic と同じく state_dict をロードできるように
        super().load_state_dict(state_dict, strict=strict)
        return True
