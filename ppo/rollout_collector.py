import torch
from torch.distributions import Normal

class RolloutCollector:
    def __init__(self, buffer, actor, critic, env, gamma=0.99):
        # object instance of RolloutBuffer class
        self.buffer = buffer
        self.actor = actor
        self.critic = critic
        self.env = env
        self.gamma = gamma
        self.episode_return = torch.zeros(
            buffer.num_env,
            device=buffer.device,
        )
        self.episode_length = torch.zeros(
            buffer.num_env,
            device=buffer.device,
        )

    @torch.no_grad()
    def collect(self, obs):
        self.buffer.reset()
        completed_length = 0
        completed_returns = 0
        num_done = 0

        for t in range(self.buffer.num_steps):
            # Actor
            mean, std = self.actor(obs) # inference
            dist = Normal(mean, std) # construct action distribution
            action = dist.sample()  # sample action from distribution
            log_prob = dist.log_prob(action).sum(dim=-1)

            # Critic
            value = self.critic(obs).squeeze(-1) # inference

            # Environment Step
            next_obs, reward, terminated, truncated, extras = (
                self.env.step(action)
            )
            next_obs = next_obs["policy"]

            # Episode Boundary
            done = terminated | truncated # true/false inference

            # record average episode length and return per rollout
            self.episode_return += reward
            self.episode_length += 1
            if done.any():
                completed_returns += self.episode_return[done].sum().item()
                completed_length += self.episode_length[done].sum().item()
                num_done += done.sum().item()
                self.episode_length[done] = 0
                self.episode_return[done] = 0

            # Timeout Correction by taking into account of truncation episodes
            reward = reward.clone() # duplicate to prevent changing raw env reward
            if "time_outs" in extras:
                reward += self.gamma * value * extras["time_outs"].float()
            else:
                reward += self.gamma * value * truncated.float()

            # Store transition
            self.buffer.add(
                obs=obs, action=action,
                value=value, reward=reward,
                log_prob=log_prob, done=done,
            )

            # Advance observation
            obs = next_obs

            if t == 0:
                print("obs:", obs.shape)
                print("mean:", mean.shape)
                print("std:", std.shape)
                print("action:", action.shape)
                print("log_prob:", log_prob.shape)
                print("value:", value.shape)
                print("reward:", reward.shape)
                print("done:", done.shape)

        # Value beyond rollout boundary (2048,)
        last_value = self.critic(obs).squeeze(dim=-1)

        # GAE + returns
        self.buffer.compute_gae(
            last_value = last_value,
            gamma=self.gamma,
        )

        # compute episode returns and length average
        if num_done > 0:
            completed_returns /= num_done
            completed_length /= num_done
        else:
            completed_returns = None
            completed_length = None
            
        # inspecting tensor shapes
        print("Buffer obs:", self.buffer.obs.shape)
        print("Advantage:", self.buffer.advantage.shape)
        print("Returns:", self.buffer.returns.shape)

        return obs, completed_length, completed_returns

