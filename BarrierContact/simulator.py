"""Simulation driver for 2D FEM with fixed and adaptive time stepping.

Provides:
- ``setup_block()`` -- create a rectangular block simulation setup.
- ``run_fixed()`` -- run with a fixed time step (implicit Euler).
- ``run_adaptive()`` -- run with adaptive time stepping using controllers
  from ``adaptive_stepping_1d.adaptive_controller``.
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from adaptive_stepping_1d.adaptive_controller import ErrorEstimator
from adaptive_stepping_1d.simulator import _make_controller

from .energies import (
    BarrierEnergy2D,
    Energy2D,
    GravityEnergy2D,
    InertiaEnergy2D,
    NeoHookeanEnergy,
    compute_contact_area,
    compute_lumped_mass,
)
from .integrators import ImplicitEuler2D
from .mesh import Mesh, MeshLoader


def setup_block(
    width: float = 1.0,
    height: float = 0.5,
    nx: int = 4,
    ny: int = 4,
    E_young: float = 1e4,
    nu_poisson: float = 0.3,
    rho: float = 1000.0,
    h0: float = 0.5,
    v0: ndarray | None = None,
    g_mag: float = 10.0,
) -> dict:
    """Create a rectangular block simulation setup.

    The block is centered horizontally with its bottom edge at ``y = h0``.

    Parameters
    ----------
    width, height : float
        Block dimensions.
    nx, ny : int
        Mesh cells in x and y directions.
    E_young, nu_poisson : float
        Material properties.
    rho : float
        Density (kg/m^2 for 2D).
    h0 : float
        Initial height of block bottom above ground (y=0).
    v0 : ndarray (2,) or None
        Initial velocity applied uniformly to all nodes. None means zero.
    g_mag : float
        Gravity magnitude (positive downward).

    Returns
    -------
    dict
        Simulation setup with keys: ``mesh``, ``x``, ``v``, ``m``,
        ``E_young``, ``nu_poisson``, ``rho``, ``g_vec``, ``ground_y``.
    """
    mesh = MeshLoader.rectangle(width, height, nx, ny)
    n_nodes = mesh.n_nodes

    # MeshLoader.rectangle puts y in [-height/2, height/2].
    # Shift so bottom sits at y = h0.
    x_init = mesh.vertices.copy()
    x_init[:, 1] += h0 + height / 2.0

    m = compute_lumped_mass(mesh, rho)

    x_flat = x_init.ravel()
    v_flat = np.tile(v0, n_nodes) if v0 is not None else np.zeros(2 * n_nodes)

    return {
        "mesh": mesh,
        "x": x_flat,
        "v": v_flat,
        "m": m,
        "E_young": E_young,
        "nu_poisson": nu_poisson,
        "rho": rho,
        "g_vec": np.array([0.0, -g_mag]),
        "ground_y": 0.0,
    }


def _build_integrator(
    setup: dict,
    kappa: float,
    dhat: float,
) -> ImplicitEuler2D:
    """Build an ``ImplicitEuler2D`` integrator from a setup dict.

    Creates InertiaEnergy2D, NeoHookeanEnergy, GravityEnergy2D, and optionally
    BarrierEnergy2D (if ``kappa > 0``).

    Parameters
    ----------
    setup : dict
        From ``setup_block()``.
    kappa : float
        Barrier stiffness. If <= 0, no barrier is added.
    dhat : float
        Barrier activation distance.

    Returns
    -------
    ImplicitEuler2D
    """
    mesh: Mesh = setup["mesh"]

    inertia = InertiaEnergy2D(setup["m"])
    neo_hookean = NeoHookeanEnergy(mesh, setup["E_young"], setup["nu_poisson"])
    gravity = GravityEnergy2D(setup["m"], setup["g_vec"])

    potentials: list[Energy2D] = [neo_hookean, gravity]

    if kappa > 0:
        contact_area = compute_contact_area(mesh)
        barrier = BarrierEnergy2D(
            y_ground=setup["ground_y"],
            kappa=kappa,
            dhat=dhat,
            contact_area=contact_area,
        )
        potentials.append(barrier)

    return ImplicitEuler2D(inertia, potentials)


def _compute_physical_energy(
    x_flat: ndarray,
    v_flat: ndarray,
    m: ndarray,
    neo_hookean: NeoHookeanEnergy,
    gravity: GravityEnergy2D,
) -> float:
    """Compute physical energy: KE + elastic PE + gravity PE (no barrier)."""
    v_2d = v_flat.reshape(-1, 2)
    ke = 0.5 * np.sum(m[:, None] * v_2d**2)
    return ke + neo_hookean.val(x_flat) + gravity.val(x_flat)


def run_fixed(
    setup: dict,
    h: float,
    T_end: float,
    kappa: float,
    dhat: float,
    tol: float = 1e-6,
    verbose: bool = False,
) -> dict:
    """Run 2D FEM simulation with fixed time step.

    Parameters
    ----------
    setup : dict
        From ``setup_block()``.
    h : float
        Fixed time step size.
    T_end : float
        End time.
    kappa, dhat : float
        Barrier parameters.
    tol : float
        Newton convergence tolerance.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Results with keys: ``x_history``, ``t_history``, ``stats_history``, ``mesh``.
    """
    integrator = _build_integrator(setup, kappa, dhat)

    x_flat = setup["x"].copy()
    v_flat = setup["v"].copy()

    t = 0.0
    t_history = [t]
    x_history = [x_flat.reshape(-1, 2).copy()]
    stats_history: list[dict] = []

    while t < T_end - 1e-14:
        dt = min(h, T_end - t)
        x_flat, v_flat, stats = integrator.step(x_flat, v_flat, dt, tol)
        t += dt

        t_history.append(t)
        x_history.append(x_flat.reshape(-1, 2).copy())
        stats_history.append(stats)

        if verbose and len(stats_history) % 50 == 0:
            x_2d = x_flat.reshape(-1, 2)
            print(
                f"  t={t:.4f}, iters={stats['newton_iters']}, "
                f"y_min={x_2d[:, 1].min():.6f}"
            )

    return {
        "x_history": x_history,
        "t_history": np.array(t_history),
        "stats_history": stats_history,
        "mesh": setup["mesh"],
    }


def run_adaptive(
    setup: dict,
    h0: float = 0.01,
    T_end: float = 1.0,
    kappa: float = 1e3,
    dhat: float = 0.1,
    tol: float = 1e-6,
    controller_type: str = "gustafsson",
    target_iter: int = 10,
    energy_tol: float = 1e-3,
    h_min: float = 1e-8,
    h_max: float = 0.1,
    verbose: bool = False,
) -> dict:
    """Run 2D FEM simulation with adaptive time stepping.

    Uses controllers and error estimators from
    ``adaptive_stepping_1d.adaptive_controller``.

    Parameters
    ----------
    setup : dict
        From ``setup_block()``.
    h0 : float
        Initial time step size.
    T_end : float
        End time.
    kappa, dhat : float
        Barrier parameters.
    tol : float
        Newton convergence tolerance.
    controller_type : str
        One of ``"i"``, ``"pi"``, ``"pid"``, ``"gustafsson"``.
    target_iter : int
        Target Newton iteration count for the error estimator.
    energy_tol : float
        Relative energy drift tolerance.
    h_min, h_max : float
        Step size bounds.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Results with keys: ``x_history``, ``t_history``, ``h_history``,
        ``stats_history``, ``mesh``.
    """
    integrator = _build_integrator(setup, kappa, dhat)
    mesh: Mesh = setup["mesh"]

    # Build physical energy terms (no barrier) for drift tracking
    neo_hookean = NeoHookeanEnergy(mesh, setup["E_young"], setup["nu_poisson"])
    gravity = GravityEnergy2D(setup["m"], setup["g_vec"])

    controller = _make_controller(controller_type)
    estimator = ErrorEstimator(
        target_iter=target_iter,
        energy_tol=energy_tol,
    )

    x_flat = setup["x"].copy()
    v_flat = setup["v"].copy()
    m = setup["m"]

    t = 0.0
    h = h0
    t_history = [t]
    x_history = [x_flat.reshape(-1, 2).copy()]
    h_history: list[float] = []
    stats_history: list[dict] = []

    E_phys = _compute_physical_energy(x_flat, v_flat, m, neo_hookean, gravity)
    accepted = 0
    rejected = 0

    while t < T_end - 1e-14:
        h = min(h, T_end - t)
        h = max(h, h_min)

        x_new, v_new, stats = integrator.step(x_flat, v_flat, h, tol)

        E_phys_new = _compute_physical_energy(x_new, v_new, m, neo_hookean, gravity)
        energy_drift = E_phys_new - E_phys

        err = estimator(
            stats, energy_drift, E_phys if abs(E_phys) > 1e-12 else 1.0
        )

        if controller_type == "gustafsson":
            h_new, accept = controller.step(err, h, stats["newton_iters"])
        else:
            h_new, accept = controller.step(err, h)

        h_new = np.clip(h_new, h_min, h_max)

        if accept or h <= h_min * 1.01:
            x_flat, v_flat = x_new, v_new
            E_phys = E_phys_new
            t += h
            accepted += 1

            t_history.append(t)
            x_history.append(x_flat.reshape(-1, 2).copy())
            h_history.append(h)
            stats_history.append(stats)

            if verbose and accepted % 50 == 0:
                x_2d = x_flat.reshape(-1, 2)
                print(
                    f"  t={t:.4f}, h={h:.6f}, iters={stats['newton_iters']}, "
                    f"err={err:.4f}, y_min={x_2d[:, 1].min():.6f}"
                )
        else:
            rejected += 1
            if verbose:
                print(f"  REJECT t={t:.4f}, h={h:.6f}, err={err:.4f}")

        h = h_new

    return {
        "x_history": x_history,
        "t_history": np.array(t_history),
        "h_history": np.array(h_history),
        "stats_history": stats_history,
        "mesh": setup["mesh"],
        "accepted_steps": accepted,
        "rejected_steps": rejected,
    }
