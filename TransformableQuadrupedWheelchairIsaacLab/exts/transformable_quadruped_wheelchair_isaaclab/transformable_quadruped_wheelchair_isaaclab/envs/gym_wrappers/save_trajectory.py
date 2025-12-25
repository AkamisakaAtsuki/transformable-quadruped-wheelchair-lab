import gym
import torch
import os

class MultiEnvDistillWrapper(gym.Wrapper):
    def __init__(self, env, save_dir="rollouts", base_filename="traj"):
        super().__init__(env)
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.base = base_filename

        obs, _ = self.env.reset()
        self.num_envs = obs['policy'].shape[0]

        self.buffers = [ [] for _ in range(self.num_envs) ]
        self.file_counters = [0] * self.num_envs

        self.prev_obs_batch = obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs_batch = obs
        self.buffers = [ [] for _ in range(self.num_envs) ]
        return obs, info

    def step(self, action_batch):
        for i in range(self.num_envs):
            obs_i   = self.prev_obs_batch['policy'][i].cpu().clone()
            act_i   = action_batch[i].cpu().clone()
            self.buffers[i].append({'obs': obs_i, 'act': act_i})

        obs_batch, rewards, terminated, truncated, info = self.env.step(action_batch)
        done_batch = (terminated | truncated).cpu().numpy()  # bool 配列

        for i, done in enumerate(done_batch):
            if done:
                buf = self.buffers[i]
                if buf:
                    fn = f"{self.base}_env{i:02d}_{self.file_counters[i]:03d}.pt"
                    path = os.path.join(self.save_dir, fn)
                    torch.save(buf, path)
                    print(f"[MultiEnvDistillWrapper] env#{i} dumped {len(buf)} steps → {fn}")
                    self.file_counters[i] += 1
                    self.buffers[i] = []

        self.prev_obs_batch = obs_batch

        return obs_batch, rewards, terminated, truncated, info
