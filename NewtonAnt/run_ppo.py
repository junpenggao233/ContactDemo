"""Minimal PPO training for Newton Ant."""
import argparse
import os
import torch
import torch.nn as nn
from torch.distributions import Normal
from ant_env import AntEnv


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.actor_mean = MLP(obs_dim, act_dim, hidden)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = MLP(obs_dim, 1, hidden)

    def forward(self, obs):
        mean = self.actor_mean(obs)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(obs).squeeze(-1)
        return mean, std, value

    def act(self, obs):
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(-1, 1), log_prob, value

    def evaluate(self, obs, actions):
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * (~dones[t]).float() - values[t]
        last_gae = delta + gamma * lam * (~dones[t]).float() * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def train(args):
    env = AntEnv(num_envs=args.num_envs)
    policy = ActorCritic(env.num_obs, env.num_act, args.hidden).to(args.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    obs = env.reset()
    ep_rewards = torch.zeros(args.num_envs, device=args.device)
    ep_lengths = torch.zeros(args.num_envs, device=args.device)
    completed_rewards = []
    completed_lengths = []

    print(f"PPO Training | {args.num_envs} envs | {args.iterations} iterations")
    print(f"{'Iter':>6} {'MeanRew':>10} {'MeanLen':>10} {'PolicyLoss':>12} {'ValueLoss':>12}")
    print("-" * 56)

    for iteration in range(args.iterations):
        # Collect rollout
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []

        with torch.no_grad():
            for _ in range(args.horizon):
                action, log_prob, value = policy.act(obs)
                next_obs, reward, done, info = env.step(action)

                obs_buf.append(obs)
                act_buf.append(action)
                logp_buf.append(log_prob)
                rew_buf.append(reward)
                val_buf.append(value)
                done_buf.append(done)

                ep_rewards += reward
                ep_lengths += 1

                # Track completed episodes
                finished = done.nonzero(as_tuple=False).squeeze(-1)
                if len(finished) > 0:
                    completed_rewards.extend(ep_rewards[finished].tolist())
                    completed_lengths.extend(ep_lengths[finished].tolist())
                    ep_rewards[finished] = 0.0
                    ep_lengths[finished] = 0.0

                obs = next_obs

            # Bootstrap value
            _, _, next_value = policy(obs)

        # Stack buffers
        obs_t = torch.stack(obs_buf)       # (T, N, obs_dim)
        act_t = torch.stack(act_buf)       # (T, N, act_dim)
        logp_t = torch.stack(logp_buf)     # (T, N)
        rew_t = torch.stack(rew_buf)       # (T, N)
        val_t = torch.stack(val_buf)       # (T, N)
        done_t = torch.stack(done_buf)     # (T, N)

        # GAE
        advantages, returns = compute_gae(rew_t, val_t, done_t, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten
        T, N = obs_t.shape[:2]
        flat_obs = obs_t.reshape(T * N, -1)
        flat_act = act_t.reshape(T * N, -1)
        flat_logp = logp_t.reshape(T * N)
        flat_adv = advantages.reshape(T * N)
        flat_ret = returns.reshape(T * N)

        # PPO update
        batch_size = T * N
        minibatch_size = batch_size // args.num_minibatches

        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(args.epochs):
            perm = torch.randperm(batch_size, device=args.device)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start:start + minibatch_size]
                mb_obs = flat_obs[idx]
                mb_act = flat_act[idx]
                mb_logp_old = flat_logp[idx]
                mb_adv = flat_adv[idx]
                mb_ret = flat_ret[idx]

                new_logp, new_val, entropy = policy.evaluate(mb_obs, mb_act)
                ratio = (new_logp - mb_logp_old).exp()
                clipped = ratio.clamp(1 - args.clip, 1 + args.clip)
                policy_loss = -torch.min(ratio * mb_adv, clipped * mb_adv).mean()
                value_loss = 0.5 * (new_val - mb_ret).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + args.vf_coef * value_loss + args.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        num_updates = args.epochs * (batch_size // minibatch_size)

        # Logging
        if (iteration + 1) % 10 == 0 or iteration == 0:
            if completed_rewards:
                recent = completed_rewards[-100:]
                mean_rew = sum(recent) / len(recent)
                recent_len = completed_lengths[-100:]
                mean_len = sum(recent_len) / len(recent_len)
            else:
                mean_rew = 0.0
                mean_len = 0.0
            print(f"{iteration + 1:>6} {mean_rew:>10.2f} {mean_len:>10.1f} "
                  f"{total_policy_loss / num_updates:>12.4f} "
                  f"{total_value_loss / num_updates:>12.4f}")

        # Save checkpoint
        if (iteration + 1) % 100 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            path = f"checkpoints/ppo_{iteration + 1}.pt"
            torch.save(policy.state_dict(), path)
            print(f"  Saved checkpoint: {path}")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    train(args)
