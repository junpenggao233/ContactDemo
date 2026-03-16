"""Newton-exact penalty contact force computation.

Implements the body-body contact model from Newton's semi-implicit solver
(eval_body_contact), including Huber-norm smoothed Coulomb friction.

Reference: newton/_src/solvers/semi_implicit/kernels_contact.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContactParams:
    """Material parameters matching Newton's ShapeConfig defaults."""

    ke: float = 2500.0   # elastic stiffness (N/m)
    kd: float = 100.0    # damping (N·s/m)
    kf: float = 1000.0   # friction stiffness (N·s/m)
    mu: float = 1.0      # Coulomb friction coefficient
    friction_smoothing: float = 1.0  # Huber delta (Newton solver default)


def huber_norm(v: np.ndarray, delta: float) -> float:
    """Huber norm matching warp.math.norm_huber exactly.

    H(v) = 0.5 * ||v||^2           if ||v|| <= delta
           delta * (||v|| - 0.5*delta)  otherwise
    """
    a = np.dot(v, v)
    if a <= delta * delta:
        return 0.5 * a
    return delta * (np.sqrt(a) - 0.5 * delta)


def compute_contact_force(
    d: float,
    v_rel: np.ndarray,
    normal: np.ndarray,
    params: ContactParams,
) -> np.ndarray:
    """Compute contact force on the cube for a single contact point.

    Parameters
    ----------
    d : float
        Signed distance (negative = penetrating).
    v_rel : ndarray(3,)
        Relative velocity of the contact point (cube velocity at contact).
    normal : ndarray(3,)
        Contact normal pointing away from ground (into cube).
    params : ContactParams
        Material parameters.

    Returns
    -------
    force : ndarray(3,)
        Force applied to the cube.
    """
    # Normal velocity component
    vn = np.dot(normal, v_rel)

    # Elastic normal force
    fn = d * params.ke

    # Damping: only when penetrating (d < 0) and approaching (vn < 0)
    step_d = 1.0 if d < 0.0 else 0.0
    fd = min(vn, 0.0) * params.kd * step_d

    # Friction: only when penetrating
    f_friction = np.zeros(3)
    if d < 0.0:
        vt = v_rel - normal * vn
        vs = huber_norm(vt, params.friction_smoothing)
        if vs > 0.0:
            f_dir = vt / vs
            f_mag = min(params.kf * vs, -params.mu * (fn + fd))
            f_friction = f_dir * f_mag

    f_total = normal * (fn + fd) + f_friction

    # Negate: f_total is force on body A (ground-side convention),
    # we want force on the cube (body B).
    return -f_total
