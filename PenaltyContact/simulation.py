"""3D rigid body simulation with penalty contact.

Semi-implicit Euler integration of a rigid cube falling onto a ground plane,
using the Newton-exact contact model from contact_model.py.
"""

from __future__ import annotations

import numpy as np

from .contact_model import ContactParams, compute_contact_force


# ---------------------------------------------------------------------------
# Quaternion utilities  (convention: [w, x, y, z])
# ---------------------------------------------------------------------------

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    result = quat_multiply(quat_multiply(q, qv), q_conj)
    return result[1:]


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


# ---------------------------------------------------------------------------
# Cube face triangles (for 8 vertices)
# ---------------------------------------------------------------------------

# Vertices indexed 0..7 as: [±h, ±h, ±h] in binary order
# 0: (-,-,-), 1: (+,-,-), 2: (-,+,-), 3: (+,+,-),
# 4: (-,-,+), 5: (+,-,+), 6: (-,+,+), 7: (+,+,+)
CUBE_FACES = np.array([
    # -z face (0,1,3,2)
    [0, 1, 3], [0, 3, 2],
    # +z face (4,6,7,5)
    [4, 6, 7], [4, 7, 5],
    # -y face (0,4,5,1)
    [0, 4, 5], [0, 5, 1],
    # +y face (2,3,7,6)
    [2, 3, 7], [2, 7, 6],
    # -x face (0,2,6,4)
    [0, 2, 6], [0, 6, 4],
    # +x face (1,5,7,3)
    [1, 5, 7], [1, 7, 3],
], dtype=np.int32)


# ---------------------------------------------------------------------------
# Rigid body
# ---------------------------------------------------------------------------

class RigidBody3D:
    """3D rigid cube with position, orientation, and velocities."""

    def __init__(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        mass: float = 1.0,
        size: float = 0.5,
    ):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.quat = quat_normalize(np.array(quat, dtype=float))
        self.omega = np.array(omega, dtype=float)
        self.mass = mass
        self.size = size
        # Moment of inertia for uniform cube: I = m * s^2 / 6
        self.inertia = mass * size**2 / 6.0

    def corners(self) -> np.ndarray:
        """Return 8 corner positions in world frame, shape (8, 3)."""
        h = self.size / 2.0
        local = np.array([
            [-h, -h, -h], [+h, -h, -h], [-h, +h, -h], [+h, +h, -h],
            [-h, -h, +h], [+h, -h, +h], [-h, +h, +h], [+h, +h, +h],
        ])
        world = np.array([quat_rotate(self.quat, c) for c in local]) + self.pos
        return world

    def corner_velocities(self) -> np.ndarray:
        """Return velocity of each corner, shape (8, 3)."""
        h = self.size / 2.0
        local = np.array([
            [-h, -h, -h], [+h, -h, -h], [-h, +h, -h], [+h, +h, -h],
            [-h, -h, +h], [+h, -h, +h], [-h, +h, +h], [+h, +h, +h],
        ])
        vels = np.zeros((8, 3))
        for i, c in enumerate(local):
            r = quat_rotate(self.quat, c)
            vels[i] = self.vel + np.cross(self.omega, r)
        return vels


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    body: RigidBody3D,
    params: ContactParams | None = None,
    dt: float = 5e-4,
    T_end: float = 3.0,
    gravity: float = -9.81,
    ground_y: float = 0.0,
    save_every: int = 20,
    angular_damping: float = 0.05,
) -> dict:
    """Run 3D rigid body simulation with penalty contact.

    Uses semi-implicit Euler: update velocities first, then positions.

    Parameters
    ----------
    body : RigidBody3D
        Initial state.
    params : ContactParams
        Contact material parameters.
    dt : float
        Time step.
    T_end : float
        End time.
    gravity : float
        Gravitational acceleration (applied in -y).
    ground_y : float
        Ground plane y-coordinate.
    save_every : int
        Save state every N steps.

    Returns
    -------
    dict with keys: t, vertices_history, cm_pos, cm_vel, omega_history
    """
    if params is None:
        params = ContactParams()

    n_steps = int(T_end / dt)
    normal = np.array([0.0, 1.0, 0.0])  # ground normal (up)

    # Storage
    t_hist = []
    verts_hist = []
    cm_pos_hist = []
    cm_vel_hist = []
    omega_hist = []

    for step in range(n_steps + 1):
        # Save state
        if step % save_every == 0:
            t_hist.append(step * dt)
            verts_hist.append(body.corners().copy())
            cm_pos_hist.append(body.pos.copy())
            cm_vel_hist.append(body.vel.copy())
            omega_hist.append(body.omega.copy())

        if step == n_steps:
            break

        # Contact at 8 corners. For a convex box vs a flat ground plane,
        # corners are always the deepest-penetrating points regardless of
        # orientation (edges/faces are linear interpolations of corners).
        # This matches Newton's contact generation for box shapes.
        corners = body.corners()
        corner_vels = body.corner_velocities()

        total_force = np.array([0.0, body.mass * gravity, 0.0])
        total_torque = np.zeros(3)

        for i in range(8):
            d = corners[i, 1] - ground_y  # signed distance (negative = penetrating)
            if d < 0:
                f = compute_contact_force(d, corner_vels[i], normal, params)
                total_force += f
                r = corners[i] - body.pos
                total_torque += np.cross(r, f)

        # Semi-implicit Euler: velocities first
        body.vel += dt * total_force / body.mass
        body.omega += dt * total_torque / body.inertia

        # Position update
        body.pos += dt * body.vel

        # Quaternion update: dq/dt = 0.5 * [0, omega] * q
        omega_quat = np.array([0.0, body.omega[0], body.omega[1], body.omega[2]])
        dq = 0.5 * quat_multiply(omega_quat, body.quat)
        body.quat += dt * dq
        body.quat = quat_normalize(body.quat)

        # Angular damping (matches Newton's integrate_rigid_body)
        body.omega *= 1.0 - angular_damping * dt

    return {
        "t": np.array(t_hist),
        "vertices_history": verts_hist,
        "cm_pos": np.array(cm_pos_hist),
        "cm_vel": np.array(cm_vel_hist),
        "omega_history": np.array(omega_hist),
    }
