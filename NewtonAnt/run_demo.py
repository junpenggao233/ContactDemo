"""Physics demo: run Ant with random actions and print stats."""
import torch
from ant_env import AntEnv


def main():
    num_envs = 4
    num_steps = 500

    env = AntEnv(num_envs=num_envs)
    obs = env.reset()

    print(f"Running {num_steps} steps with {num_envs} environments...")
    print(f"{'Step':>6} {'Height':>10} {'Fwd Vel':>10} {'Reward':>10}")
    print("-" * 42)

    total_reward = torch.zeros(num_envs, device=env.device)

    for step in range(num_steps):
        actions = torch.randn(num_envs, env.num_act, device=env.device).clamp(-1, 1)
        obs, rew, done, info = env.step(actions)
        total_reward += rew

        if (step + 1) % 50 == 0:
            height = obs[:, 0].mean().item()
            fwd_vel = obs[:, 5].mean().item()
            mean_rew = rew.mean().item()
            print(f"{step + 1:>6} {height:>10.4f} {fwd_vel:>10.4f} {mean_rew:>10.4f}")

    print(f"\nTotal reward (mean): {total_reward.mean().item():.2f}")


if __name__ == "__main__":
    main()
