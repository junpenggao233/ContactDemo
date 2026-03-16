# Penalty-Based Contact Method: Theory

## 1. Core Idea

The penalty method enforces the non-penetration constraint **approximately** by adding a
repulsive force that grows with penetration depth. Instead of enforcing $d \geq 0$ exactly
(where $d$ is the gap between two bodies), we allow slight penetration and penalize it with
a stiff spring:

$$
f_n = \begin{cases}
k_e \cdot \delta & \text{if } \delta > 0 \quad (\text{penetrating}) \\
0 & \text{otherwise}
\end{cases}
$$

where $\delta = -d$ is the penetration depth and $k_e$ is the penalty stiffness.

**Analogy:** Imagine the ground is covered by an invisible layer of very stiff springs.
When an object pushes into the ground, the springs push back proportionally to how far
the object has gone in.

## 2. Mathematical Formulation

### 2.1 Gap Function

For a point $\mathbf{x}$ above a ground plane at $y = 0$:

$$
d(\mathbf{x}) = y - y_{\text{ground}}
$$

Penetration depth:

$$
\delta = \max(0, -d) = \max(0, y_{\text{ground}} - y)
$$

### 2.2 Contact Energy (Potential)

The penalty method can be viewed as adding a **contact potential energy**:

$$
E_{\text{contact}} = \frac{1}{2} k_e \, \delta^2
$$

The force is then the negative gradient of this energy:

$$
f_n = -\frac{\partial E_{\text{contact}}}{\partial \mathbf{x}} = k_e \, \delta
$$

This is just Hooke's law — a linear spring with rest length at the contact surface.

### 2.3 Adding Damping (Newton's Model)

A pure spring causes indefinite bouncing. To dissipate energy, we add a velocity-dependent
damping term. Newton's contact model uses:

$$
f_n = d \cdot k_e, \qquad f_d = \min(v_n, 0) \cdot k_d \cdot \text{step}(d)
$$

where $v_n = \mathbf{n} \cdot \mathbf{v}_{\text{rel}}$ is the normal component of relative
velocity, and $\text{step}(d) = 1$ when $d < 0$ (penetrating), $0$ otherwise.

The $\min(v_n, 0)$ ensures damping only activates during approach (not separation),
preventing "sticky" contacts. The $\text{step}(d)$ ensures damping only acts during
penetration.

### 2.4 Coulomb Friction

The tangential velocity is obtained by removing the normal component from the relative
velocity:

$$
v_n = \mathbf{n} \cdot \mathbf{v}_{\text{rel}}, \qquad \mathbf{v}_t = \mathbf{v}_{\text{rel}} - \mathbf{n}\,v_n
$$

Coulomb's law states that friction opposes sliding with magnitude bounded by the normal
force:

$$
\|\mathbf{f}_t\| \leq \mu \, |f_n + f_d|
$$

The idealized Coulomb model has a discontinuity at $\|\mathbf{v}_t\| = 0$. A simple
regularization replaces $\|\mathbf{v}_t\|$ with $\max(\|\mathbf{v}_t\|, \epsilon)$,
but this introduces a sharp kink at $\epsilon$.

### 2.5 Huber-Norm Smoothed Friction (Newton Physics Engine)

The simple $\max(\|\mathbf{v}_t\|, \epsilon)$ regularization above has a sharp kink at
$\|\mathbf{v}_t\| = \epsilon$, which causes gradient discontinuity. The Newton physics
engine uses a more principled approach: the **Huber norm** as the friction velocity
measure.

#### The Huber Norm

The Huber norm of a vector $\mathbf{v}$ with smoothing parameter $\delta$ is defined as:

$$
H_\delta(\mathbf{v}) = \begin{cases}
\frac{1}{2}\|\mathbf{v}\|^2 & \text{if } \|\mathbf{v}\| \leq \delta \\[6pt]
\delta\!\left(\|\mathbf{v}\| - \frac{1}{2}\delta\right) & \text{if } \|\mathbf{v}\| > \delta
\end{cases}
$$

This is the classical Huber loss applied to $\|\mathbf{v}\|$. It transitions smoothly
from quadratic (near zero) to linear (far from zero), providing $C^1$ continuity at
$\|\mathbf{v}\| = \delta$.

**Key properties:**

- At $\|\mathbf{v}\| = \delta$: both branches give $\frac{1}{2}\delta^2$ (continuous)
- Gradient at $\|\mathbf{v}\| = \delta$: both branches give $\delta \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|}$ ($C^1$ smooth)
- For $\|\mathbf{v}\| \gg \delta$: $H_\delta \approx \delta\|\mathbf{v}\|$ (linear in norm)
- For $\|\mathbf{v}\| \ll \delta$: $H_\delta \approx \frac{1}{2}\|\mathbf{v}\|^2$ (quadratic)

#### Friction Force with Huber Norm

Using $v_s = H_\delta(\mathbf{v}_t)$, the friction force is:

$$
\mathbf{f}_t = \frac{\mathbf{v}_t}{v_s} \cdot \min\!\left(k_f \cdot v_s,\; -\mu(f_n + f_d)\right)
$$

This has two regimes:

1. **Stiffness regime** ($k_f v_s < -\mu(f_n + f_d)$): The $v_s$ cancels —
   $\mathbf{f}_t = k_f \mathbf{v}_t$, a viscous friction proportional to tangential velocity.

2. **Coulomb regime** ($k_f v_s \geq -\mu(f_n + f_d)$): For $\|\mathbf{v}_t\| \gg \delta$,
   $\frac{\mathbf{v}_t}{v_s} \approx \frac{\mathbf{v}_t}{\delta\|\mathbf{v}_t\|}$, so the
   friction magnitude approaches $\frac{\mu|f_n + f_d|}{\delta}$. With Newton's default
   $\delta = 1.0$, this recovers standard Coulomb friction for velocities above 1 m/s.

The advantage over $\epsilon$-clamping is that the Huber norm provides a smooth ($C^1$)
transition everywhere, making it suitable for gradient-based optimization and
differentiable simulation.

#### Implementation

From [Warp](https://github.com/NVIDIA/warp)'s `warp/_src/math.py`:

```python
def norm_huber(v, delta=1.0):
    a = dot(v, v)           # ||v||^2
    if a <= delta * delta:
        return 0.5 * a      # quadratic regime
    return delta * (sqrt(a) - 0.5 * delta)  # linear regime
```

## 3. Rigid Cube Dynamics

### 3.1 Cube State

The cube is described by position $\mathbf{x} \in \mathbb{R}^3$ (center of mass),
orientation quaternion $\mathbf{q}$ ($\|\mathbf{q}\| = 1$), linear velocity
$\mathbf{v}$, and angular velocity $\boldsymbol{\omega}$.

For a uniform cube with mass $m$ and side length $s$, the inertia is scalar:
$I = ms^2/6$. This means the gyroscopic torque
$\boldsymbol{\omega} \times (I\boldsymbol{\omega}) = \mathbf{0}$ vanishes, and angular
dynamics can be computed directly in world frame.

### 3.2 Contact Points

The 8 corners in body-local coordinates are $\mathbf{c}_i \in \{(\pm h, \pm h, \pm h)\}$
where $h = s/2$. In world frame:

$$
\mathbf{p}_i = \mathbf{x} + R(\mathbf{q})\,\mathbf{c}_i, \qquad
\mathbf{v}_i = \mathbf{v} + \boldsymbol{\omega} \times (\mathbf{p}_i - \mathbf{x})
$$

For a convex box against a flat ground plane, corners are always the deepest-penetrating
points (edges and faces are linear interpolations of corners), so checking only 8 corners
suffices.

### 3.3 Force Accumulation

For each corner $i$ with $d_i = (\mathbf{p}_i)_y - y_{\text{ground}} < 0$, the contact
force $\mathbf{f}_i$ is computed per §2. Total force and torque:

$$
\mathbf{F} = m\mathbf{g} + \sum_i \mathbf{f}_i, \qquad
\boldsymbol{\tau} = \sum_i (\mathbf{p}_i - \mathbf{x}) \times \mathbf{f}_i
$$

### 3.4 Semi-Implicit Euler Integration

Velocities are updated first, then positions advance using the new velocities
(matching Newton's `integrate_rigid_body`):

$$
\mathbf{v}^{n+1} = \mathbf{v}^n + \Delta t \, \frac{\mathbf{F}}{m}, \qquad
\mathbf{x}^{n+1} = \mathbf{x}^n + \Delta t \, \mathbf{v}^{n+1}
$$

$$
\boldsymbol{\omega}^{n+1} = \boldsymbol{\omega}^n + \Delta t \, \frac{\boldsymbol{\tau}}{I}, \qquad
\mathbf{q}^{n+1} = \text{normalize}\!\left(\mathbf{q}^n + \frac{\Delta t}{2}(0, \boldsymbol{\omega}^{n+1}) \otimes \mathbf{q}^n\right)
$$

Angular damping (Newton default $\gamma = 0.05$):
$\boldsymbol{\omega}^{n+1} \leftarrow \boldsymbol{\omega}^{n+1}(1 - \gamma\Delta t)$.

### 3.5 Stability Condition

When a corner penetrates the ground, the penalty spring creates a 1D harmonic
oscillator: $m\ddot{x} = -k_e x$. The semi-implicit Euler update for this system is:

$$
v^{n+1} = v^n - \frac{k_e}{m}\,x^n\,\Delta t, \qquad
x^{n+1} = x^n + v^{n+1}\,\Delta t
$$

This can be written as a matrix iteration $\mathbf{z}^{n+1} = A\,\mathbf{z}^n$ where
$\mathbf{z} = (x, v)^T$ and:

$$
A = \begin{pmatrix} 1 - \omega^2\Delta t^2 & \Delta t \\ -\omega^2\Delta t & 1 \end{pmatrix},
\qquad \omega = \sqrt{k_e / m}
$$

The eigenvalues of $A$ satisfy $\lambda^2 - (2 - \omega^2\Delta t^2)\lambda + 1 = 0$.
Since $\det(A) = 1$, the two eigenvalues have product 1. For stability ($|\lambda| \leq 1$),
they must be complex conjugates on the unit circle, which requires the discriminant to be
negative:

$$
(2 - \omega^2\Delta t^2)^2 - 4 < 0 \quad \Longrightarrow \quad
|2 - \omega^2\Delta t^2| < 2 \quad \Longrightarrow \quad
\omega\,\Delta t < 2
$$

Therefore:

$$
\Delta t < \frac{2}{\omega} = 2\sqrt{\frac{m}{k_e}}
$$

With $k_e = 2500$ N/m, $m = 1$ kg: $\Delta t < 0.04$ s. We use $\Delta t = 5 \times 10^{-4}$ s.

## 4. Characteristics of the Penalty Method

**Strengths:**
- Simple to implement — contact is just a force evaluation, no nonlinear solve needed
- Works with explicit/semi-implicit integrators (cheap per step, GPU-friendly)
- Easy to add friction, damping, and restitution as additional force terms

**Limitations:**
- Penetration is never truly zero — only reduced by increasing $k_e$
- High $k_e$ makes the ODE stiff, requiring small $\Delta t$ ($< 2\sqrt{m/k_e}$)
- This stiffness-timestep tradeoff is the fundamental limitation: accuracy (small
  penetration) and performance (large $\Delta t$) are in direct tension

## 5. References

- Bender, Erleben, Trinkle, "Interactive Simulation of Rigid Body Dynamics in
  Computer Graphics," *Eurographics STAR*, 2014.
- Li et al., "Incremental Potential Contact," *ACM SIGGRAPH*, 2020.
