# ContactsDemo

2D FEM barrier-contact simulation with fixed and adaptive implicit time stepping.

## Project structure

```
ContactsDemo/
├── BarrierContact/                  # Core simulation library
│   ├── energies.py                  # Energy classes (NeoHookean, IPC barrier, gravity, inertia)
│   ├── integrators.py               # Implicit Euler with Newton solver + line search
│   ├── mesh.py                      # Mesh dataclass and structured rectangle loader
│   ├── simple_simulator.py          # Experiment setups and fixed/adaptive time steppers
│   ├── visualize.py                 # Polyscope animation + GIF export (matplotlib)
│   ├── run_demo.py                  # CLI entry point
│   └── experiments/
│       ├── falling_block.py         # Standalone: 10×10 block falling under gravity
│       └── sliding_block.py         # Standalone: 2×2 block sliding with initial velocity
└── adaptive_stepping_2d/            # Reference implementation (Polyscope-based)
```

## Experiments

Parameters match the reference experiments in `adaptive_stepping_2d/experiments/`.

| Experiment | Block | E (Pa) | ρ (kg/m²) | dt | Contact |
|---|---|---|---|---|---|
| Falling block | 10×10, nx=ny=10 | 1800 | 1.0 | Courant ≈ 0.0177 s | IPC barrier (κ=180, d̂=1.0) |
| Sliding block | 2×2, nx=ny=8 | 200 000 | 1000 | 0.01 s | Point-edge IPC (κ=20 000, d̂=0.02) |

## How to run

```bash
# Falling block → saves falling_block.gif
uv run python BarrierContact/experiments/falling_block.py

# Sliding block → saves sliding_block.gif
uv run python BarrierContact/experiments/sliding_block.py

# Options (both experiments)
--T 3.0          # end time (default: 6.0 / 2.0)
--dt 0.01        # time step (falling defaults to Courant condition)
--adaptive       # PI-controller adaptive stepping
--out out.gif    # output path
```

CLI runner with Polyscope animation:

```bash
uv run python BarrierContact/run_demo.py --experiment falling
uv run python BarrierContact/run_demo.py --experiment sliding --adaptive
```

## Dependencies

```bash
uv sync           # core (numpy, scipy, matplotlib)
uv sync --extra vis   # + polyscope for interactive animation
```
