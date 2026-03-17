# Newton Ant

Ant quadruped locomotion using [Newton Physics](https://newton-physics.github.io/newton/).
Replicates the [rewarped DFlex Ant](https://github.com/rewarped/rewarped/blob/main/rewarped/envs/dflex/ant.py) environment.

## Requirements

- NVIDIA GPU (Maxwell+), driver 545+, Python 3.10+

```bash
pip install newton torch
```

## Usage

```bash
cd NewtonAnt

# PPO training
uv run python run_ppo.py --num-envs 64
# Evaluate a trained checkpoint
uv run python run_eval.py checkpoints/xxx.pt
uv run python run_eval.py checkpoints/xxx.pt --usd eval.usd   # export USD
```

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
| `--max-agent-steps` | 4,100,000 | Total steps before stopping |
| `--device` | `cuda:0` | Device |

## Files

| File | Description |
|------|-------------|
| `ant_env.py` | Gym-like `AntEnv` (Featherstone solver, 37-dim obs, 8-dim act) |
| `assets/ant.xml` | Ant MJCF (from rewarped, density=1000) |
| `run_ppo.py` | Standalone PPO training |
| `run_eval.py` | Evaluate a trained checkpoint, optionally export USD |
