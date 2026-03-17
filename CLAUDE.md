# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase demonstrating contact mechanics approaches in physics simulation:
- **BarrierContact**: 2D FEM with Incremental Potential Contact (IPC) log-barrier — guaranteed zero penetration, implicit Euler + Newton solver
- **PenaltyContact**: 3D rigid body penalty contact matching Newton Physics Engine — semi-implicit Euler, spring-damper + Huber friction
- **NewtonAnt**: Ant quadruped locomotion using Newton Physics Engine with PPO reinforcement learning

## Commands

```bash
# Dependencies (uses uv)
uv sync                          # core: numpy, scipy, matplotlib
uv sync --extra vis              # + polyscope visualization
uv sync --extra dev              # + pytest, ruff

# Linting
ruff check .                     # lint (line-length=100)
ruff format .                    # format

# BarrierContact demos
python BarrierContact/run_demo.py --experiment falling [--adaptive] [--dt 0.01] [--save out.gif]
python BarrierContact/run_demo.py --experiment sliding
python BarrierContact/experiments/falling_block.py   # standalone

# PenaltyContact demos (run as module)
python -m PenaltyContact.experiment --experiment falling [--save-gif out.gif]
python -m PenaltyContact.experiment --experiment spinning
python -m PenaltyContact.experiment --experiment sliding

# NewtonAnt (requires NVIDIA GPU, driver 545+)
cd NewtonAnt && uv sync          # separate dependency set (newton, torch, usd-core)
python run_ppo.py --num-envs 64
python run_eval.py checkpoints/xxx.pt [--usd eval.usd]
```

## Architecture

### Energy-based formulation (BarrierContact)
All forces/constraints are expressed as energy classes with `val()`, `grad()`, `hess()` methods inheriting from `Energy2D`. The implicit Euler integrator minimizes the Incremental Potential IP(x) = ½Σmᵢ‖xᵢ - x̃ᵢ‖² + h²ΣEₖ(x) using Newton's method with filtered line search to maintain barrier feasibility.

Key files: `energies.py` (NeoHookean, barrier, gravity), `integrators.py` (Newton solver), `mesh.py` (mesh generation via triangle library).

### Newton-exact contact model (PenaltyContact)
`contact_model.py` implements the full Newton contact force: elastic penalty + damping + Huber-norm smoothed Coulomb friction. `simulation.py` runs semi-implicit Euler with 8-corner collision detection against ground plane. Stability requires Δt < 2√(m/kₑ). Mathematical derivations are in `theory.md`.

### PPO training (NewtonAnt)
`ant_env.py` wraps Newton Physics with 37-dim observations and 8-dim actions (joint torques). `run_ppo.py` is a self-contained PPO implementation with RunningMeanStd normalization, GAE, and clipped surrogate objective. Uses 64 parallel environments, Featherstone articulation solver, 16 physics substeps per frame.

## Key Conventions

- BarrierContact and PenaltyContact are CPU-compatible; NewtonAnt requires NVIDIA GPU
- NewtonAnt has its own `pyproject.toml` and `uv.lock` (separate from root)
- Visualization uses polyscope (3D interactive) and matplotlib (2D/GIF export)
- Results files (`.gif`, `.mp4`, `results/`) are gitignored
