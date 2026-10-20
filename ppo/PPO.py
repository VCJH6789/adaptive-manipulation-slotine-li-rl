import torch
from torch.distributions import Normal

class PPO: 
    def __init__(self, actor, critic, actor_optimizer, critic_optimizer,
                 clip_epsilon=0.2, entropy_coef=0.01, 
                 num_epoch=5, minibatch_size=8192):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.num_epoch = num_epoch
        self.minibatch_size = minibatch_size

    def update(self, buffer):
        # metrics
        actor_losses = []
        critic_losses = []
        entropies = []
        approx_kls = []
        clip_fractions = []
        for _ in range(self.num_epoch):
            # outer loop
            for batch in buffer.minibatches(self.minibatch_size):
                # training loop for new policy
                (mb_obs, mb_action, mb_old_log_prob,
                 mb_advantage, mb_returns, mb_value) = batch

                # actor
                mean, std = self.actor(mb_obs)
                dist = Normal(mean, std)
                new_log_prob = dist.log_prob(mb_action).sum(dim=-1)
                # compute ratio between old and new policy
                ratio = torch.exp(new_log_prob - mb_old_log_prob)
                # clipping ratio
                surrogate1 = ratio * mb_advantage
                surrogate2 = torch.clamp(ratio,
                                     1.0 - self.clip_epsilon,
                                     1.0 + self.clip_epsilon) \
                                     * mb_advantage
                # compute actor loss
                actor_loss = -torch.min(surrogate1,surrogate2).mean()
                # entropy --> standard deviation
                entropy = dist.entropy().sum(dim=-1).mean()
                # actor loss + entropy --> encourage exploration
                actor_total_loss = actor_loss - self.entropy_coef * entropy
                # backprog
                self.actor_optimizer.zero_grad() # reset parameter gradient within chain rule
                actor_total_loss.backward()
                # clip total gradient norm to prevent excessively large updates
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=0.5, # max grad 0.5
                )
                self.actor_optimizer.step()

                # critic
                value = self.critic(mb_obs).squeeze(dim=-1)
                # compute critic loss
                critic_loss = ((value - mb_returns)**2).mean()
                # backprog
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # clipping gradient
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    max_norm=0.5,
                )
                self.critic_optimizer.step()

                # metrics --> monitoring purposes
                with torch.no_grad():
                    # KL Divergence
                    approx_kl = (new_log_prob - mb_old_log_prob)
                    approx_kl = ((ratio - 1.0) - approx_kl).mean()
                    # Clip Fraction
                    clip_fraction = (torch.abs(ratio - 1.0) > 
                                     self.clip_epsilon).float().mean()
                actor_losses.append(actor_loss.item()) # item convert to Python number from Pytorch
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())
                approx_kls.append(approx_kl.item())
                clip_fractions.append(clip_fraction.item())
        # outer loop
        metrics = {
            "actor_loss": sum(actor_losses) / len(actor_losses),
            "critic_loss": sum(critic_losses) / len(critic_losses),
            "entropy": sum(entropies) / len(entropies),
            "approx_kl": sum(approx_kls) / len(approx_kls),
            "clip_fraction": (
                sum(clip_fractions)
                / len(clip_fractions)
            ),
        }
        return metrics
                

                