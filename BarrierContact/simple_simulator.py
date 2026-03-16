"""Simplified 2D FEM simulator.

Fixed and adaptive time stepping for falling-block and sliding-block experiments.
Parameters match the reference experiments in adaptive_stepping_2d/experiments/.
No dependency on adaptive_stepping_1d or polyscope.

Usage (as a module from project root):
    from BarrierContact.simple_simulator import setup_falling_block, run_fixed
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from BarrierContact.energies import (
    BarrierEnergy2D,
    GravityEnergy2D,
    InertiaEnergy2D,
    NeoHookeanEnergy,
    PointEdgeBarrierEnergy,
    compute_contact_area,
    compute_lumped_mass,
)
from BarrierContact.integrators import ImplicitEuler2D
from BarrierContact.mesh import MeshLoader


# ---------------------------------------------------------------------------
# Ground edge helper (used by sliding block)
# ---------------------------------------------------------------------------


def _make_ground_edges(
    x_min: float = -5.0, x_max: float = 10.0, n_segments: int = 30
) -> ndarray:
    """Create flat ground edges along y=0.

    Returns
    -------
    ndarray, shape (n_segments, 2, 2)
    """
    xs = np.linspace(x_min, x_max, n_segments + 1)
    edges = np.zeros((n_segments, 2, 2))
    for i in range(n_segments):
        edges[i, 0] = [xs[i], 0.0]
        edges[i, 1] = [xs[i + 1], 0.0]
    return edges


# ---------------------------------------------------------------------------
# Experiment setups
# ---------------------------------------------------------------------------


def setup_falling_block(
    width: float = 10.0,
    height: float = 10.0,
    nx: int = 10,
    ny: int = 10,
    E_young: float = 1800.0,
    nu: float = 0.2,
    rho: float = 1.0,
    h0: float = 5.0,
    g_mag: float = 10.0,
    kappa: float | None = None,
    dhat: float | None = None,
) -> dict:
    """Block falling under gravity onto a flat ground at y=0.

    Matches the reference experiment in adaptive_stepping_2d/experiments/falling_block.py.
    Default parameters: 10×10 block, dx=1.0, κ = 0.1*E, d̂ = dx.

    Returns a setup dict with keys: ``x``, ``v``, ``mesh``, ``integrator``.
    """
    dx = width / nx
    if kappa is None:
        kappa = 0.1 * E_young
    if dhat is None:
        dhat = dx

    mesh = MeshLoader.rectangle(width, height, nx, ny)
    n = mesh.n_nodes

    # Shift so block bottom is at y = h0
    x = mesh.vertices.copy()
    x[:, 1] += h0 + height / 2.0

    m = compute_lumped_mass(mesh, rho)
    contact_area = compute_contact_area(mesh)

    inertia = InertiaEnergy2D(m)
    elastic = NeoHookeanEnergy(mesh, E_young, nu)
    gravity = GravityEnergy2D(m, np.array([0.0, -g_mag]))
    barrier = BarrierEnergy2D(
        y_ground=0.0,
        kappa=kappa,
        dhat=dhat,
        contact_area=contact_area,
    )

    integrator = ImplicitEuler2D(inertia, [elastic, gravity, barrier])

    print(
        f"[falling_block] {n} nodes, {mesh.n_triangles} tris | "
        f"E={E_young:.0f} nu={nu} rho={rho} h0={h0} kappa={kappa:.1f} dhat={dhat:.3f}"
    )
    return {"x": x.ravel(), "v": np.zeros(2 * n), "mesh": mesh, "integrator": integrator}


def setup_sliding_block(
    width: float = 2.0,
    height: float = 2.0,
    nx: int = 8,
    ny: int = 8,
    E_young: float = 200_000.0,
    nu: float = 0.3,
    rho: float = 1000.0,
    v0x: float = 2.0,
    g_mag: float = 10.0,
    kappa: float = 20_000.0,
    dhat: float = 0.02,
) -> dict:
    """Block sliding right with initial horizontal velocity on flat ground.

    Matches the reference experiment in adaptive_stepping_2d/experiments/sliding_block.py.
    Uses PointEdgeBarrierEnergy with 30 ground segments from x=-5 to x=10.

    Returns a setup dict with keys: ``x``, ``v``, ``mesh``, ``integrator``.
    """
    mesh = MeshLoader.rectangle(width, height, nx, ny)
    n = mesh.n_nodes

    # Block bottom starts at y = 0.5 (h0=0.5 matches reference)
    h0 = 0.5
    x = mesh.vertices.copy()
    x[:, 1] += h0 + height / 2.0

    m = compute_lumped_mass(mesh, rho)

    # Bottom boundary nodes and their contact weights
    bottom_nodes = mesh.find_bottom_nodes()
    contact_area_all = compute_contact_area(mesh)
    contact_area_bottom = contact_area_all[bottom_nodes]
    if contact_area_bottom.sum() == 0:
        contact_area_bottom = np.full(len(bottom_nodes), width / (nx + 1))

    # 30 ground segments from x=-5 to x=10 at y=0 (matches reference)
    ground_edges = _make_ground_edges(x_min=-5.0, x_max=10.0, n_segments=30)

    inertia = InertiaEnergy2D(m)
    elastic = NeoHookeanEnergy(mesh, E_young, nu)
    gravity = GravityEnergy2D(m, np.array([0.0, -g_mag]))
    barrier = PointEdgeBarrierEnergy(
        bottom_node_indices=bottom_nodes,
        ground_edges=ground_edges,
        n_block_nodes=n,
        kappa=kappa,
        dhat=dhat,
        contact_area=contact_area_bottom,
    )

    integrator = ImplicitEuler2D(inertia, [elastic, gravity, barrier])

    # Initial horizontal velocity for all nodes
    v = np.zeros(2 * n)
    v[0::2] = v0x

    print(
        f"[sliding_block] {n} nodes, {mesh.n_triangles} tris | "
        f"E={E_young:.0f} rho={rho} v0x={v0x} kappa={kappa:.0f} dhat={dhat}"
    )
    return {"x": x.ravel(), "v": v, "mesh": mesh, "integrator": integrator}


# ---------------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------------


def run_fixed(
    setup: dict,
    dt: float = 0.02,
    T_end: float = 6.0,
    tol: float = 1e-4,
    verbose: bool = True,
) -> dict:
    """Run simulation with a fixed time step.

    Returns a trajectory dict with keys:
    ``t``, ``x`` (list of (n,2) arrays), ``stats``, ``mesh``.
    """
    x = setup["x"].copy()
    v = setup["v"].copy()
    integrator = setup["integrator"]
    mesh = setup["mesh"]

    t = 0.0
    t_hist = [0.0]
    x_hist = [x.reshape(-1, 2).copy()]
    stats_hist: list[dict] = []

    n_steps = int(np.ceil(T_end / dt))
    report_every = max(1, n_steps // 20)

    for i in range(n_steps):
        dt_actual = min(dt, T_end - t)
        if dt_actual < 1e-14:
            break

        x, v, stats = integrator.step(x, v, dt_actual, tol)
        t += dt_actual

        t_hist.append(t)
        x_hist.append(x.reshape(-1, 2).copy())
        stats_hist.append(stats)

        if verbose and (i + 1) % report_every == 0:
            x2 = x.reshape(-1, 2)
            print(
                f"  t={t:.3f}/{T_end}  iters={stats['newton_iters']:2d}"
                f"  y_min={x2[:, 1].min():.4f}"
                f"  x_com={x2[:, 0].mean():.3f}"
            )

    return {
        "t": np.array(t_hist),
        "x": x_hist,
        "stats": stats_hist,
        "mesh": mesh,
    }


def run_adaptive(
    setup: dict,
    dt0: float = 0.02,
    T_end: float = 6.0,
    tol: float = 1e-4,
    dt_min: float = 1e-5,
    dt_max: float = 0.1,
    target_iters: int = 8,
    verbose: bool = True,
) -> dict:
    """Run simulation with adaptive time stepping.

    Step size is adjusted based on Newton iteration count using a simple
    PI controller:  h_new = h * (target_iters / actual_iters)^0.7

    A step is rejected (and dt halved) if Newton didn't converge.

    Returns a trajectory dict with keys:
    ``t``, ``x``, ``stats``, ``dt``, ``mesh``, ``accepted``, ``rejected``.
    """
    x = setup["x"].copy()
    v = setup["v"].copy()
    integrator = setup["integrator"]
    mesh = setup["mesh"]

    t = 0.0
    dt = dt0
    prev_iters: int | None = None

    t_hist = [0.0]
    x_hist = [x.reshape(-1, 2).copy()]
    stats_hist: list[dict] = []
    dt_hist: list[float] = []

    accepted = rejected = 0
    report_every = max(1, int(T_end / dt0 / 20))

    while t < T_end - 1e-14:
        dt = float(np.clip(dt, dt_min, min(dt_max, T_end - t)))

        x_new, v_new, stats = integrator.step(x, v, dt, tol)
        iters = stats["newton_iters"]

        # Reject if clearly unconverged (residual >> tol and many iters)
        converged = stats["final_residual"] < tol * 100 or iters < 50
        if not converged:
            rejected += 1
            dt = max(dt * 0.5, dt_min)
            if verbose:
                print(
                    f"  REJECT t={t:.4f} dt={dt:.5f}"
                    f" residual={stats['final_residual']:.2e}"
                )
            prev_iters = None
            continue

        # Accept step
        x, v = x_new, v_new
        t += dt
        accepted += 1

        t_hist.append(t)
        x_hist.append(x.reshape(-1, 2).copy())
        stats_hist.append(stats)
        dt_hist.append(dt)

        if verbose and accepted % report_every == 0:
            x2 = x.reshape(-1, 2)
            print(
                f"  t={t:.3f}/{T_end}  dt={dt:.4f}  iters={iters:2d}"
                f"  y_min={x2[:, 1].min():.4f}"
                f"  x_com={x2[:, 0].mean():.3f}"
            )

        # Adapt dt: PI controller on Newton iterations
        if iters == 0:
            factor = 2.0
        else:
            ratio = target_iters / iters
            if prev_iters is not None and prev_iters > 0:
                prev_ratio = target_iters / prev_iters
                factor = ratio**0.7 * (ratio / prev_ratio) ** 0.2
            else:
                factor = ratio**0.7
            factor = float(np.clip(factor, 0.3, 2.0))

        dt = float(np.clip(dt * factor, dt_min, dt_max))
        prev_iters = iters

    print(f"  Steps: {accepted} accepted, {rejected} rejected")
    return {
        "t": np.array(t_hist),
        "x": x_hist,
        "stats": stats_hist,
        "dt": np.array(dt_hist) if dt_hist else np.array([dt0]),
        "mesh": mesh,
        "accepted": accepted,
        "rejected": rejected,
    }
