"""PPO training for Newton Ant — adapted from minimal-stable-PPO with mineral paper configs.

Reference: "Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation" (ICLR 2025)
PPO on AntRun target: 2048.7 ± 36.6 episode return.
Based on: https://github.com/ToruOwO/minimal-stable-PPO
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from ant_env import AntEnv


# ---------------------------------------------------------------------------
# Running Mean/Std (from minimal-stable-PPO/ppo/utils.py)
# Updates stats during forward() when in training mode.
# ---------------------------------------------------------------------------
class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(insize, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(insize, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        return new_mean, new_var, tot_count

    def forward(self, x, unnorm=False):
        if self.training:
            mean = x.mean(0)
            var = x.var(0)
            self.running_mean, self.running_var, self.count = self._update(
                self.running_mean, self.running_var, self.count,
                mean, var, x.size(0),
            )
        if unnorm:
            y = torch.clamp(x, -5.0, 5.0)
            y = torch.sqrt(self.running_var.float() + self.epsilon) * y + self.running_mean.float()
        else:
            y = (x - self.running_mean.float()) / torch.sqrt(self.running_var.float() + self.epsilon)
            y = torch.clamp(y, -5.0, 5.0)
        return y


# ---------------------------------------------------------------------------
# Experience Buffer (from minimal-stable-PPO/ppo/experience.py)
# ---------------------------------------------------------------------------
class ExperienceBuffer:
    def __init__(self, num_envs, horizon, obs_dim, act_dim, device):
        self.num_envs = num_envs
        self.horizon = horizon
        self.device = device
        self.storage = {
            "obses": torch.zeros((horizon, num_envs, obs_dim), device=device),
            "rewards": torch.zeros((horizon, num_envs, 1), device=device),
            "values": torch.zeros((horizon, num_envs, 1), device=device),
            "neglogpacs": torch.zeros((horizon, num_envs), device=device),
            "dones": torch.zeros((horizon, num_envs), dtype=torch.uint8, device=device),
            "actions": torch.zeros((horizon, num_envs, act_dim), device=device),
            "mus": torch.zeros((horizon, num_envs, act_dim), device=device),
            "sigmas": torch.zeros((horizon, num_envs, act_dim), device=device),
            "returns": torch.zeros((horizon, num_envs, 1), device=device),
        }
        self.data_dict = None

    def update(self, name, idx, val):
        self.storage[name][idx, :] = val

    def compute_return(self, last_values, gamma, tau):
        last_gae_lam = 0
        mb_advs = torch.zeros_like(self.storage["rewards"])
        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                next_values = last_values
            else:
                next_values = self.storage["values"][t + 1]
            not_done = 1.0 - self.storage["dones"][t].float().unsqueeze(1)
            delta = self.storage["rewards"][t] + gamma * next_values * not_done - self.storage["values"][t]
            mb_advs[t] = last_gae_lam = delta + gamma * tau * not_done * last_gae_lam
            self.storage["returns"][t] = mb_advs[t] + self.storage["values"][t]

    def prepare_training(self):
        # Flatten (horizon, num_envs, ...) -> (batch, ...)
        self.data_dict = {}
        for k, v in self.storage.items():
            s = v.size()
            self.data_dict[k] = v.transpose(0, 1).reshape(s[0] * s[1], *s[2:])
        advantages = self.data_dict["returns"] - self.data_dict["values"]
        self.data_dict["advantages"] = (
            (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ).squeeze(1)

    def get_minibatch(self, start, end):
        return {k: v[start:end] for k, v in self.data_dict.items()}

    def update_mu_sigma(self, mu, sigma, start, end):
        self.data_dict["mus"][start:end] = mu
        self.data_dict["sigmas"][start:end] = sigma


# ---------------------------------------------------------------------------
# Network Architecture (mineral DFlexAntPPO config)
# ---------------------------------------------------------------------------
def orthogonal_init(module, gain=1.0):
    nn.init.orthogonal_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class MLP(nn.Module):
    def __init__(self, in_dim, units=(512, 256, 128)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in units:
            linear = orthogonal_init(nn.Linear(prev, h), gain=np.sqrt(2))
            layers.extend([linear, nn.LayerNorm(h), nn.SiLU()])
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, units=(512, 256, 128)):
        super().__init__()
        self.actor_mlp = MLP(obs_dim, units)
        self.critic_mlp = MLP(obs_dim, units)

        out_size = units[-1]
        self.mu = orthogonal_init(nn.Linear(out_size, act_dim), gain=0.01)
        self.sigma = orthogonal_init(nn.Linear(out_size, act_dim), gain=0.01)
        self.value = orthogonal_init(nn.Linear(out_size, 1), gain=1.0)

    def _actor_critic(self, obs):
        a_feat = self.actor_mlp(obs)
        c_feat = self.critic_mlp(obs)
        raw_mu = self.mu(a_feat)
        raw_sigma = self.sigma(a_feat)
        # dreamerv3_normal
        mu = torch.tanh(raw_mu)
        std = 0.9 * torch.sigmoid(raw_sigma + 2.0) + 0.1
        value = self.value(c_feat)
        return mu, std, value

    @torch.no_grad()
    def act(self, obs):
        mu, sigma, value = self._actor_critic(obs)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        neglogp = -dist.log_prob(action).sum(-1)
        return {
            "actions": action,
            "neglogpacs": neglogp,
            "values": value,
            "mus": mu,
            "sigmas": sigma,
        }

    @torch.no_grad()
    def act_inference(self, obs):
        mu, _, _ = self._actor_critic(obs)
        return mu

    def forward(self, obs, prev_actions):
        mu, sigma, value = self._actor_critic(obs)
        dist = torch.distributions.Normal(mu, sigma)
        neglogp = -dist.log_prob(prev_actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return {
            "prev_neglogp": neglogp,
            "values": value,
            "entropy": entropy,
            "mus": mu,
            "sigmas": sigma,
        }


# ---------------------------------------------------------------------------
# KL and Schedulers (from minimal-stable-PPO)
# ---------------------------------------------------------------------------
def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -0.5
    kl = (c1 + c2 + c3).sum(dim=-1)
    return kl.mean()


class AdaptiveScheduler:
    def __init__(self, kl_threshold=0.008):
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > 2.0 * self.kl_threshold:
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < 0.5 * self.kl_threshold:
            lr = min(current_lr * 1.5, self.max_lr)
        return lr


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    device = args.device
    env = AntEnv(num_envs=args.num_envs, device=device)
    model = ActorCritic(env.num_obs, env.num_act).to(device)
    obs_rms = RunningMeanStd(env.num_obs).to(device)
    value_rms = RunningMeanStd(1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    scheduler = AdaptiveScheduler(kl_threshold=args.kl_threshold)

    horizon = args.horizon
    num_envs = args.num_envs
    batch_size = num_envs * horizon
    minibatch_size = args.minibatch_size
    num_minibatches = batch_size // minibatch_size
    mini_epochs = args.mini_epochs
    gamma = args.gamma
    tau = args.tau
    e_clip = args.e_clip
    critic_coef = args.critic_coef
    bounds_loss_coef = args.bounds_loss_coef
    max_agent_steps = args.max_agent_steps
    last_lr = args.lr

    storage = ExperienceBuffer(num_envs, horizon, env.num_obs, env.num_act, device)

    obs = env.reset()
    current_rewards = torch.zeros((num_envs, 1), device=device)
    current_lengths = torch.zeros(num_envs, device=device)
    episode_rewards = []
    episode_lengths = []
    agent_steps = 0
    best_rewards = -1e9
    epoch_num = 0
    t_start = time.time()

    print(f"PPO Training (minimal-stable-PPO + mineral config)")
    print(f"  {num_envs} envs | horizon={horizon} | batch={batch_size} | minibatch={minibatch_size}")
    print(f"  max_agent_steps={max_agent_steps}")
    print(f"{'Epoch':>6} {'Steps':>10} {'MeanRew':>10} {'MeanLen':>8} {'ALoss':>10} {'CLoss':>10} {'KL':>10} {'LR':>10} {'Best':>10}")
    print("-" * 96)

    while agent_steps < max_agent_steps:
        epoch_num += 1

        # ==================================================================
        # Collect rollout (play_steps)
        # ==================================================================
        model.eval()
        obs_rms.eval()
        value_rms.eval()

        for n in range(horizon):
            # Normalize obs (eval mode — no stats update)
            proc_obs = obs_rms(obs)
            res = model.act(proc_obs)
            # Unnormalize value for GAE (eval mode — no stats update)
            res["values"] = value_rms(res["values"], unnorm=True)

            storage.update("obses", n, obs)
            storage.update("actions", n, res["actions"])
            storage.update("neglogpacs", n, res["neglogpacs"])
            storage.update("values", n, res["values"])
            storage.update("mus", n, res["mus"])
            storage.update("sigmas", n, res["sigmas"])

            actions = torch.clamp(res["actions"], -1.0, 1.0)
            obs, reward, done, info = env.step(actions)

            storage.update("dones", n, done)

            # Value bootstrap on truncation
            shaped_reward = reward.unsqueeze(1)
            if "time_outs" in info:
                shaped_reward = shaped_reward + gamma * res["values"] * info["time_outs"].unsqueeze(1).float()
            storage.update("rewards", n, shaped_reward)

            # Track episode stats
            current_rewards += reward.unsqueeze(1)
            current_lengths += 1
            done_idx = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_idx) > 0:
                for i in done_idx:
                    episode_rewards.append(current_rewards[i].item())
                    episode_lengths.append(current_lengths[i].item())
                current_rewards[done_idx] = 0
                current_lengths[done_idx] = 0

        # Last values for GAE
        proc_obs = obs_rms(obs)
        last_res = model.act(proc_obs)
        last_values = value_rms(last_res["values"], unnorm=True)

        agent_steps += batch_size

        # GAE
        storage.compute_return(last_values, gamma, tau)
        storage.prepare_training()

        # Normalize values and returns (train mode — updates stats)
        value_rms.train()
        storage.data_dict["values"] = value_rms(storage.data_dict["values"])
        storage.data_dict["returns"] = value_rms(storage.data_dict["returns"])
        value_rms.eval()

        # Recompute advantages after normalization
        advantages = storage.data_dict["returns"] - storage.data_dict["values"]
        storage.data_dict["advantages"] = (
            (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ).squeeze(1)

        # ==================================================================
        # Train epoch
        # ==================================================================
        model.train()
        obs_rms.train()

        a_losses, c_losses, kls = [], [], []

        for mini_ep in range(mini_epochs):
            ep_kls = []
            for i in range(num_minibatches):
                start = i * minibatch_size
                end = (i + 1) * minibatch_size
                mb = storage.get_minibatch(start, end)

                mb_obs = obs_rms(mb["obses"])  # train mode — updates stats
                res = model(mb_obs, mb["actions"])

                old_neglogp = mb["neglogpacs"]
                advantage = mb["advantages"]
                old_mu = mb["mus"]
                old_sigma = mb["sigmas"]

                # Actor loss
                ratio = torch.exp(old_neglogp - res["prev_neglogp"])
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - e_clip, 1.0 + e_clip)
                a_loss = torch.max(-surr1, -surr2).mean()

                # Critic loss (unclipped)
                c_loss = (res["values"] - mb["returns"]).pow(2).mean()

                # Bounds loss
                mu = res["mus"]
                b_loss = (torch.clamp(mu - 1.1, min=0.0).pow(2) +
                          torch.clamp(mu + 1.1, max=0.0).pow(2)).sum(-1).mean()

                loss = a_loss + 0.5 * critic_coef * c_loss + bounds_loss_coef * b_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    kl = policy_kl(mu.detach(), res["sigmas"].detach(), old_mu, old_sigma)

                a_losses.append(a_loss.item())
                c_losses.append(c_loss.item())
                ep_kls.append(kl)

                # Update mu/sigma for next minibatch KL (matching minimal-stable-PPO)
                storage.update_mu_sigma(mu.detach(), res["sigmas"].detach(), start, end)

            av_kl = torch.mean(torch.stack(ep_kls)).item()
            kls.append(av_kl)

            # KL-adaptive LR per mini-epoch (matching minimal-stable-PPO)
            last_lr = scheduler.update(last_lr, av_kl)
            for pg in optimizer.param_groups:
                pg["lr"] = last_lr

        model.eval()
        obs_rms.eval()

        # ==================================================================
        # Logging
        # ==================================================================
        avg_a = sum(a_losses) / len(a_losses) if a_losses else 0
        avg_c = sum(c_losses) / len(c_losses) if c_losses else 0
        avg_kl = sum(kls) / len(kls) if kls else 0

        if episode_rewards:
            recent = episode_rewards[-200:]
            mean_rew = sum(recent) / len(recent)
            recent_len = episode_lengths[-200:]
            mean_len = sum(recent_len) / len(recent_len)
        else:
            mean_rew = 0.0
            mean_len = 0.0

        if epoch_num % 10 == 0 or epoch_num == 1:
            print(f"{epoch_num:>6} {agent_steps:>10} {mean_rew:>10.1f} {mean_len:>8.0f} "
                  f"{avg_a:>10.4f} {avg_c:>10.4f} {avg_kl:>10.5f} {last_lr:>10.6f} {best_rewards:>10.1f}")

        # Save checkpoints
        if epoch_num % 100 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt = {
                "model": model.state_dict(),
                "running_mean_std": obs_rms.state_dict(),
                "value_mean_std": value_rms.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_num,
            }
            torch.save(ckpt, f"checkpoints/ppo_{epoch_num}.pt")
            print(f"  Saved checkpoint: checkpoints/ppo_{epoch_num}.pt")

        if mean_rew > best_rewards and len(episode_rewards) > 10:
            best_rewards = mean_rew
            os.makedirs("checkpoints", exist_ok=True)
            ckpt = {
                "model": model.state_dict(),
                "running_mean_std": obs_rms.state_dict(),
                "value_mean_std": value_rms.state_dict(),
                "epoch": epoch_num,
                "mean_reward": mean_rew,
            }
            torch.save(ckpt, "checkpoints/best.pt")

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed / 60:.1f} min. Best mean reward: {best_rewards:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--mini-epochs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--e-clip", type=float, default=0.2)
    parser.add_argument("--critic-coef", type=float, default=4.0)
    parser.add_argument("--bounds-loss-coef", type=float, default=0.0001)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--kl-threshold", type=float, default=0.008)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-agent-steps", type=int, default=4_100_000)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    train(args)
