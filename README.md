# ContactsDemo

Demonstrations of contact mechanics methods for physics simulation, comparing
**barrier (IPC)** and **penalty** approaches side by side.

## Methods

### [BarrierContact/](BarrierContact/) — Incremental Potential Contact (IPC)

2D FEM simulation with log-barrier contact forces. Uses implicit integration with
Newton's method and filtered line search. Guarantees zero penetration ($d > 0$ always)
but requires solving nonlinear systems each step.

- Deformable bodies (NeoHookean elasticity)
- Point-edge IPC barrier potential
- Fixed and adaptive time stepping

### [PenaltyContact/](PenaltyContact/) — Penalty Contact (Newton Physics Engine Model)

3D rigid body simulation with spring-damper contact forces. Uses explicit (semi-implicit
Euler) integration. Allows small penetration proportional to $1/k_e$ but is simple and
fast.

- Rigid body with quaternion rotations
- Contact model matching [Newton physics engine](https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/semi_implicit) exactly
- Huber-norm smoothed Coulomb friction

## Quick comparison

| | Barrier (IPC) | Penalty (Newton) |
|---|---|---|
| **Penetration** | Zero (guaranteed) | Small, controlled by stiffness $k_e$ |
| **Integration** | Implicit (Newton solver) | Semi-implicit Euler |
| **Stability** | Unconditional | $\Delta t < 2\sqrt{m/k_e}$ |
| **Cost per step** | Expensive (nonlinear solve) | Cheap (explicit force eval) |
| **Bodies** | Deformable (FEM) | Rigid |
| **Dimension** | 2D | 3D |
| **Visualization** | Polyscope + matplotlib GIF | Polyscope + Pillow GIF |

## How to run

```bash
# Barrier contact (2D FEM)
python -m BarrierContact.run_demo --experiment falling
python BarrierContact/experiments/falling_block.py

# Penalty contact (3D rigid body)
python -m PenaltyContact.experiment --experiment falling
python -m PenaltyContact.experiment --experiment falling --save-gif falling.gif
```

## Dependencies

```bash
uv sync                   # core (numpy, scipy, matplotlib)
uv sync --extra vis       # + polyscope for interactive animation
pip install Pillow        # for penalty contact GIF export
```
