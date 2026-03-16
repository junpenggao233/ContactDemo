# PenaltyContact

3D rigid cube penalty contact simulation using the Newton physics engine's exact contact model.

## Contact model

Implements Newton's `eval_body_contact` kernel from
[`kernels_contact.py`](https://github.com/newton-physics/newton/blob/main/newton/_src/solvers/semi_implicit/kernels_contact.py):

- **Normal force:** $f_n = d \cdot k_e$ (elastic) + $\min(v_n, 0) \cdot k_d \cdot \text{step}(d)$ (damping)
- **Friction:** Huber-norm smoothed Coulomb — $\mathbf{f}_t = \frac{\mathbf{v}_t}{v_s} \cdot \min(k_f \cdot v_s,\; -\mu(f_n + f_d))$
- **Integration:** semi-implicit Euler with angular damping, matching Newton's `integrate_rigid_body`


## Default parameters (Newton's `ShapeConfig` defaults)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ke` | 2500 N/m | Elastic stiffness |
| `kd` | 100 N·s/m | Contact damping |
| `kf` | 1000 N·s/m | Friction stiffness |
| `mu` | 1.0 | Coulomb friction coefficient |
| `friction_smoothing` | 1.0 | Huber norm delta |

## File structure

```
PenaltyContact/
├── contact_model.py       # Newton-exact force computation (ContactParams, huber_norm, compute_contact_force)
├── simulation.py          # 3D rigid body, quaternion utils, semi-implicit Euler loop
├── experiment.py          # Polyscope visualization + GIF export, CLI entry point
├── penalty_cube_demo.py   # Original 2D matplotlib demo (reference)
├── theory.md              # Penalty method theory and comparison with barrier methods
└── newton_comparison.md   # Line-by-line verification against Newton source code
```

## How to run

```bash
# Interactive Polyscope viewer
python -m PenaltyContact.experiment --experiment falling
python -m PenaltyContact.experiment --experiment spinning
python -m PenaltyContact.experiment --experiment sliding

# Save as GIF
python -m PenaltyContact.experiment --experiment falling --save-gif falling.gif

# Override parameters
python -m PenaltyContact.experiment --experiment falling --ke 5000 --mu 0.5 --dt 1e-4
```

## Experiments

| Experiment | Setup | What it demonstrates |
|-----------|-------|---------------------|
| `falling` | Cube at y=2m, slight tilt, vx=0.5 | Bouncing, damping, settling |
| `spinning` | Cube at y=3m, $\omega$=[0, 5, 2] | Friction slowing angular motion |
| `sliding` | Cube on ground, vx=3.0 | Friction deceleration to rest |

## Collision detection

For a convex box vs flat ground plane, collision detection reduces to checking the
y-coordinate of each of the 8 corner vertices against the ground height. Corners are
always the deepest-penetrating points regardless of cube orientation (edges and faces
are linear interpolations of corners). This matches Newton's contact generation for
box shapes.

## Dependencies

- `numpy` (simulation)
- `polyscope` (interactive visualization)
- `Pillow` (GIF export)
