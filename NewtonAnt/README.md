# Newton Ant

Ant quadruped locomotion using [Newton Physics](https://newton-physics.github.io/newton/) (`pip install newton`).
Replicates the [rewarped DFlex Ant](https://github.com/rewarped/rewarped/blob/main/rewarped/envs/dflex/ant.py) environment,
with RL configs referencing [mineral](https://github.com/etaoxing/mineral).

## Quick Start

```bash
pip install newton torch
cd NewtonAnt

python run_demo.py                                          # physics demo
python run_ppo.py --num-envs 64 --iterations 1000           # PPO training
python run_trajopt.py --num-envs 16 --horizon 64            # trajectory optimization
```

Requires: NVIDIA GPU (Maxwell+), driver 545+, Python 3.10+.

## Files

| File | Description |
|------|-------------|
| `ant_env.py` | Gym-like `AntEnv` (Featherstone solver, 37-dim obs, 8-dim act) |
| `assets/ant.xml` | DFlex Ant MJCF (from rewarped, density=1000) |
| `run_demo.py` | Random actions, prints height/velocity/reward |
| `run_ppo.py` | Standalone minimal PPO |
| `run_trajopt.py` | Differentiable trajectory optimization via `wp.Tape` |

## Environment — Matches Rewarped Exactly

All physics and reward parameters match [`rewarped/envs/dflex/ant.py`](https://github.com/rewarped/rewarped/blob/main/rewarped/envs/dflex/ant.py):

- **Solver:** Featherstone, 16 substeps, frame_dt=1/60, angular_damping=0.0
- **Contacts:** ke=4e3, kd=1e3, kf=300, mu=0.75, restitution=0.0
- **Joints:** limit_ke=1e3, limit_kd=10, armature_scale=5, density=1000
- **Actions:** scale=200, penalty=0.0, clipped to [-1, 1]
- **Obs (37):** height, quat, lin_vel, ang_vel, joint_pos(8), joint_vel(8, scaled 0.1x), up, heading, prev_actions
- **Reward:** `progress_vel + 0.1*up + heading + (height - 0.27)`
- **Termination:** height < 0.27 or step >= 1000
- **MJCF:** Same `dflex/ant.xml`, initial joint angles [0,1,0,-1,0,-1,0,1]

## PPO — Simplified from Mineral

Standalone PPO; compare with [mineral DFlexAntPPO](https://github.com/etaoxing/mineral/blob/main/mineral/cfgs/agent/DFlexAntPPO.yaml).
To approximate mineral defaults: `--num-envs 64 --horizon 32 --epochs 5 --lr 5e-4 --ent-coef 0.0 --vf-coef 4.0`.
Mineral additionally uses AdamW, KL-adaptive LR, LayerNorm, input normalization, and a [512,256,128] MLP.

## Newton-Specific Adaptations

Newton (v1.0.0, Warp 1.12.0) differs from rewarped's `warp.sim` (Warp 1.3.3):

| Aspect | Rewarped | Newton |
|--------|----------|--------|
| MJCF | `wp.sim.parse_mjcf(path, builder, density=1000)` | `builder.add_mjcf(path)` + density baked in XML |
| Solver | `wp.sim.FeatherstoneIntegrator` | `newton.solvers.SolverFeatherstone` |
| Force control | `control.joint_act` | `control.joint_f` |
| Multi-env | manual | `builder.replicate(template, N)` |
| Coordinates | Y-up | Z-up (auto-converted) |
