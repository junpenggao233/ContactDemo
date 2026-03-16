"""Minimal 2D rigid cube hitting the ground using penalty-based contact.

Demonstrates the spring-damper penalty method:
  - A rigid square falls under gravity onto a flat ground at y=0
  - Contact force: f_n = k * penetration + c * penetration_rate
  - Coulomb friction: f_t = -mu * f_n * v_t / |v_t|
  - Integration: symplectic Euler

Run:
    python -m PenaltyContact.penalty_cube_demo
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------------
# Rigid body state
# ---------------------------------------------------------------------------

class RigidBody2D:
    """A 2D rigid square with position, angle, and velocities."""

    def __init__(
        self,
        cx: float,
        cy: float,
        size: float,
        mass: float,
        angle: float = 0.0,
    ):
        self.cx = cx          # center x
        self.cy = cy          # center y
        self.size = size      # side length
        self.mass = mass
        self.angle = angle    # rotation angle (radians)

        self.vx = 0.0         # translational velocity x
        self.vy = 0.0         # translational velocity y
        self.omega = 0.0      # angular velocity

        # Moment of inertia for a square: I = m * s^2 / 6
        self.I = mass * size**2 / 6.0

    def corners(self) -> np.ndarray:
        """Return 4 corner positions in world frame, shape (4, 2)."""
        h = self.size / 2.0
        # Local corners (CCW)
        local = np.array([
            [-h, -h],
            [ h, -h],
            [ h,  h],
            [-h,  h],
        ])
        c, s = np.cos(self.angle), np.sin(self.angle)
        R = np.array([[c, -s], [s, c]])
        world = (R @ local.T).T + np.array([self.cx, self.cy])
        return world


# ---------------------------------------------------------------------------
# Penalty contact
# ---------------------------------------------------------------------------

def compute_penalty_forces(
    body: RigidBody2D,
    y_ground: float,
    k: float,
    c: float,
    mu: float,
) -> tuple[float, float, float]:
    """Compute total contact force and torque from penalty springs at corners.

    For each corner below the ground plane:
      - Normal force:  f_n = k * delta + c * max(0, delta_dot)
      - Friction:      f_t = -mu * f_n * sign(v_t)  (smoothed)

    Returns (fx, fy, torque) in world frame.
    """
    corners = body.corners()
    fx_total = 0.0
    fy_total = 0.0
    tau_total = 0.0

    for corner in corners:
        penetration = y_ground - corner[1]   # delta > 0 means penetrating
        if penetration <= 0:
            continue

        # Corner velocity = body velocity + omega x r
        rx = corner[0] - body.cx
        ry = corner[1] - body.cy
        vx_corner = body.vx - body.omega * ry
        vy_corner = body.vy + body.omega * rx

        # Normal direction is +y (ground pushes up)
        pen_rate = -vy_corner   # delta_dot > 0 when approaching

        # Normal force (spring + damping, only damp during approach)
        fn = k * penetration + c * max(0.0, pen_rate)
        fn = max(fn, 0.0)  # no adhesion

        # Friction force (smoothed Coulomb)
        vt = vx_corner  # tangential velocity (x-component on flat ground)
        eps = 1e-4
        ft = -mu * fn * vt / max(abs(vt), eps)

        # Accumulate
        fx_total += ft
        fy_total += fn
        tau_total += rx * fn - ry * ft  # r x f

    return fx_total, fy_total, tau_total


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    body: RigidBody2D,
    dt: float = 1e-4,
    T_end: float = 2.0,
    gravity: float = -9.81,
    k: float = 1e5,
    c: float = 500.0,
    mu: float = 0.3,
    y_ground: float = 0.0,
    save_every: int = 100,
) -> dict:
    """Run penalty contact simulation with symplectic Euler.

    Parameters
    ----------
    body : RigidBody2D
        Initial state of the rigid cube.
    dt : float
        Time step (must be small enough for stiffness k).
    T_end : float
        Total simulation time.
    gravity : float
        Gravitational acceleration (negative = downward).
    k : float
        Penalty stiffness (N/m).
    c : float
        Damping coefficient (N·s/m).
    mu : float
        Coulomb friction coefficient.
    y_ground : float
        Ground plane y-coordinate.
    save_every : int
        Save state every N steps for visualization.
    """
    n_steps = int(T_end / dt)
    history = {
        "t": [], "cx": [], "cy": [], "angle": [],
        "vy": [], "corners": [], "ke": [], "pe": [],
    }

    for step in range(n_steps):
        # Save state
        if step % save_every == 0:
            ke = 0.5 * body.mass * (body.vx**2 + body.vy**2) + 0.5 * body.I * body.omega**2
            pe = body.mass * abs(gravity) * body.cy
            history["t"].append(step * dt)
            history["cx"].append(body.cx)
            history["cy"].append(body.cy)
            history["angle"].append(body.angle)
            history["vy"].append(body.vy)
            history["corners"].append(body.corners().copy())
            history["ke"].append(ke)
            history["pe"].append(pe)

        # Contact forces
        fx, fy, tau = compute_penalty_forces(body, y_ground, k, c, mu)

        # Gravity
        fy += body.mass * gravity

        # Symplectic Euler: update velocity first, then position
        body.vx += dt * fx / body.mass
        body.vy += dt * fy / body.mass
        body.omega += dt * tau / body.I

        body.cx += dt * body.vx
        body.cy += dt * body.vy
        body.angle += dt * body.omega

    # Convert to arrays
    for key in history:
        if key != "corners":
            history[key] = np.array(history[key])

    return history


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def animate_result(history: dict, y_ground: float = 0.0, save_path: str | None = None):
    """Animate the cube trajectory."""
    fig, (ax_main, ax_energy) = plt.subplots(1, 2, figsize=(12, 5))

    # Main animation panel
    ax_main.set_xlim(-2, 2)
    ax_main.set_ylim(-0.5, 4)
    ax_main.set_aspect("equal")
    ax_main.set_xlabel("x (m)")
    ax_main.set_ylabel("y (m)")
    ax_main.set_title("Penalty Contact: Rigid Cube on Ground")

    # Ground
    ax_main.axhline(y=y_ground, color="brown", linewidth=3)
    ax_main.fill_between([-3, 3], y_ground - 0.5, y_ground, color="burlywood", alpha=0.5)

    # Cube patch
    cube_patch = plt.Polygon(history["corners"][0], closed=True,
                              facecolor="steelblue", edgecolor="navy", linewidth=2)
    ax_main.add_patch(cube_patch)
    time_text = ax_main.text(0.02, 0.95, "", transform=ax_main.transAxes, fontsize=10)

    # Energy panel
    t = history["t"]
    ke = history["ke"]
    pe = history["pe"]
    total_e = ke + pe
    ax_energy.plot(t, ke, label="Kinetic", color="red", alpha=0.7)
    ax_energy.plot(t, pe, label="Potential", color="blue", alpha=0.7)
    ax_energy.plot(t, total_e, label="Total", color="black", linewidth=2)
    ax_energy.set_xlabel("Time (s)")
    ax_energy.set_ylabel("Energy (J)")
    ax_energy.set_title("Energy vs Time")
    ax_energy.legend()
    energy_marker, = ax_energy.plot([], [], "ro", markersize=8)

    def update(frame):
        cube_patch.set_xy(history["corners"][frame])
        time_text.set_text(f"t = {history['t'][frame]:.3f} s")
        energy_marker.set_data([t[frame]], [total_e[frame]])
        return cube_patch, time_text, energy_marker

    anim = FuncAnimation(fig, update, frames=len(history["t"]),
                         interval=20, blit=True)

    plt.tight_layout()

    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer="pillow", fps=50)
        print("Done.")
    else:
        plt.show()

    return anim


def plot_summary(history: dict, y_ground: float = 0.0):
    """Static summary plot: trajectory, velocity, and energy."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    t = history["t"]

    # (0,0) Height vs time
    axes[0, 0].plot(t, history["cy"], "b-")
    axes[0, 0].axhline(y=y_ground, color="brown", linestyle="--", label="ground")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Center y (m)")
    axes[0, 0].set_title("Height of cube center")
    axes[0, 0].legend()

    # (0,1) Vertical velocity
    axes[0, 1].plot(t, history["vy"], "r-")
    axes[0, 1].axhline(y=0, color="gray", linestyle="--")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("vy (m/s)")
    axes[0, 1].set_title("Vertical velocity")

    # (1,0) Angle
    axes[1, 0].plot(t, np.degrees(history["angle"]), "g-")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Angle (deg)")
    axes[1, 0].set_title("Rotation angle")

    # (1,1) Energy
    ke = history["ke"]
    pe = history["pe"]
    axes[1, 1].plot(t, ke, label="Kinetic", alpha=0.7)
    axes[1, 1].plot(t, pe, label="Potential", alpha=0.7)
    axes[1, 1].plot(t, ke + pe, label="Total", color="black", linewidth=2)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Energy (J)")
    axes[1, 1].set_title("Energy")
    axes[1, 1].legend()

    plt.suptitle("Penalty Contact Demo: Rigid Cube Falling on Ground", fontsize=13)
    plt.tight_layout()
    plt.savefig("PenaltyContact/penalty_contact_summary.png", dpi=150)
    print("Saved summary plot to PenaltyContact/penalty_contact_summary.png")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a cube: 0.5m side, 1kg, dropped from 2m with a slight tilt
    cube = RigidBody2D(cx=0.0, cy=2.0, size=0.5, mass=1.0, angle=0.15)
    cube.vy = 0.0
    cube.vx = 0.5   # slight horizontal velocity for interesting dynamics

    print("=== Penalty Contact Demo ===")
    print(f"Cube: {cube.size}m side, {cube.mass}kg, dropped from y={cube.cy}m")
    print(f"Initial angle: {np.degrees(cube.angle):.1f} deg")
    print(f"Initial velocity: vx={cube.vx}, vy={cube.vy}")
    print()

    # Simulate
    # k=1e5 is stiff enough to limit penetration to ~0.1mm
    # dt=1e-4 is needed for stability: dt < 2*sqrt(m/k) ≈ 6e-3
    history = simulate(
        cube,
        dt=1e-4,
        T_end=3.0,
        gravity=-9.81,
        k=1e5,       # penalty stiffness (N/m)
        c=200.0,     # damping coefficient
        mu=0.3,      # friction coefficient
        save_every=200,
    )

    print(f"Simulated {len(history['t'])} saved frames")
    print(f"Final height: {history['cy'][-1]:.4f} m")
    print(f"Max penetration below ground: {max(0, -history['cy'].min() + 0.25):.5f} m")
    print()

    # Plot summary
    plot_summary(history)

    # Uncomment to save animation as GIF:
    # animate_result(history, save_path="PenaltyContact/penalty_cube.gif")
