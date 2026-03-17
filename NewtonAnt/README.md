# Newton Ant

Ant quadruped locomotion using [Newton Physics](https://newton-physics.github.io/newton/).
Replicates the [rewarped DFlex Ant](https://github.com/rewarped/rewarped/blob/main/rewarped/envs/dflex/ant.py) environment.

## Requirements

- NVIDIA GPU (Maxwell+), driver 545+, Python 3.10+

```bash
pip install newton torch
```

## Quick Start

```bash
cd NewtonAnt
uv sync

# Train PPO (~16 min on RTX 5080)
uv run python run_ppo.py --num-envs 64 --max-agent-steps 6200000

# Evaluate the pretrained checkpoint
uv run python run_eval.py checkpoints/best.pt --num-steps 1000

# Export USD visualization (viewable in Omniverse / usdview)
uv run python run_eval.py checkpoints/best.pt --usd eval.usd --num-steps 1000
```

## Results

PPO training reaches **~2500 mean reward**, exceeding the reference PPO result of
**2048.7 +/- 36.6** reported in the
[Stabilizing RL in Differentiable Multiphysics Simulation](https://arxiv.org/abs/2412.12089)
paper (ICLR 2025).

| Metric | Paper (PPO) | Ours |
|--------|-------------|------|
| Mean reward | 2048.7 +/- 36.6 | ~2500 |
| Forward velocity | -- | 0.6-1.4 m/s |
| Displacement / episode | -- | ~15 m |
| Training steps | 6M | 6.2M |

### Key implementation details

The following discrepancies between Newton and rewarped required correction:

1. **Action inversion** -- rewarped negates actions (`acts = -acts`) to match DFlex joint conventions
2. **Joint limit parameters** -- Newton's MJCF parser derives `limit_ke=2500, limit_kd=100` from MuJoCo's `solref` defaults; rewarped uses `limit_ke=1000, limit_kd=10`
3. **Velocity ordering** -- Newton `joint_qd[0:3]` = spatial linear, `[3:6]` = angular; rewarped is the opposite
4. **Body mass correction** -- rewarped's MJCF parser has a bug where capsule shapes use `density=1000` (the function parameter) instead of the XML default `density=5.0`; we post-correct Newton's body masses to match

### PPO arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--num-envs` | 64 | Parallel environments |
| `--horizon` | 32 | Rollout steps per env before each PPO update |
| `--minibatch-size` | 2048 | Minibatch size |
| `--mini-epochs` | 5 | PPO epochs per update |
| `--lr` | 5e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--tau` | 0.95 | GAE lambda |
| `--e-clip` | 0.2 | PPO surrogate clip range |
| `--critic-coef` | 4.0 | Value loss coefficient |
| `--kl-threshold` | 0.008 | KL-adaptive LR threshold |
| `--max-agent-steps` | 6,200,000 | Total steps before stopping |
| `--device` | `cuda:0` | Device |

## Files

| File | Description |
|------|-------------|
| `ant_env.py` | Gym-like `AntEnv` (Featherstone solver, 37-dim obs, 8-dim act) |
| `assets/ant.xml` | Ant MJCF (identical to rewarped, density=5.0) |
| `run_ppo.py` | Standalone PPO training |
| `run_eval.py` | Evaluate a trained checkpoint, optionally export USD |
| `checkpoints/best.pt` | Pretrained PPO checkpoint (reward ~2500) |
