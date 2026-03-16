#!/usr/bin/env python3
"""Demo runner for 2D FEM barrier-contact simulations.

Run from the ContactsDemo project root:

    python BarrierContact/run_demo.py --experiment falling
    python BarrierContact/run_demo.py --experiment sliding --adaptive
    python BarrierContact/run_demo.py --experiment falling --save fall.gif

Experiments
-----------
falling : 10×10 block falls under gravity and bounces on flat ground.
          Parameters match adaptive_stepping_2d/experiments/falling_block.py:
          E=1800, nu=0.2, rho=1.0, h0=5.0, g=10.0, κ=0.1E, d̂=dx.
          Default dt is from Courant condition: 0.75 * dx / sqrt(E/rho).

sliding : 2×2 block slides right with v0=[2,0] on flat ground.
          Parameters match adaptive_stepping_2d/experiments/sliding_block.py:
          E=200000, rho=1000, κ=20000, d̂=0.02, dt=0.01.

Options
-------
--adaptive      Use adaptive time stepping (PI controller on Newton iters).
--dt DT         Fixed time step (default: Courant for falling, 0.01 for sliding).
--T  T_END      End time (default: 6.0 for falling, 2.0 for sliding).
--save PATH     Save animation to a .gif or .mp4 file.
--no-anim       Skip the animation, show only static plots.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Ensure the project root (ContactsDemo/) is importable regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from BarrierContact.simple_simulator import (  # noqa: E402
    run_adaptive,
    run_fixed,
    setup_falling_block,
    setup_sliding_block,
)
from BarrierContact.visualize import (  # noqa: E402
    animate_simulation,
    plot_final_frame,
    plot_newton_iters,
    plot_trajectory,
)


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------


def _courant_dt(E_young: float = 1800.0, rho: float = 1.0, dx: float = 1.0) -> float:
    """Compute dt from Courant condition: 0.75 * dx / sqrt(E / rho)."""
    c0 = np.sqrt(E_young / rho)
    return 0.75 * dx / c0


def run_falling(
    adaptive: bool = False,
    dt: float | None = None,
    T_end: float = 6.0,
    save_path: str | None = None,
    no_anim: bool = False,
) -> dict:
    print("=" * 55)
    print("  Falling Block Experiment")
    print("=" * 55)

    # Parameters matching adaptive_stepping_2d/experiments/falling_block.py
    E_young = 1800.0
    rho = 1.0
    width = 10.0
    nx = 10

    setup = setup_falling_block(
        width=width,
        height=10.0,
        nx=nx,
        ny=10,
        E_young=E_young,
        nu=0.2,
        rho=rho,
        h0=5.0,
        g_mag=10.0,
        # kappa=None → 0.1*E = 180, dhat=None → dx=1.0
    )

    # Default dt from Courant condition (matches reference)
    if dt is None:
        dx = width / nx
        dt = _courant_dt(E_young, rho, dx)
        print(f"  Courant dt = {dt:.6f} s")

    if adaptive:
        print(f"Running adaptive stepping (T={T_end}s) ...")
        traj = run_adaptive(
            setup, dt0=dt, T_end=T_end, tol=1e-4,
            dt_min=1e-5, dt_max=0.1, target_iters=8, verbose=True,
        )
    else:
        print(f"Running fixed stepping dt={dt:.6f} (T={T_end}s) ...")
        traj = run_fixed(setup, dt=dt, T_end=T_end, tol=1e-4, verbose=True)

    x_final = traj["x"][-1]
    print(f"\nDone. {len(traj['t'])-1} steps recorded.")
    print(f"  Final y_min  = {x_final[:, 1].min():.4f} m")
    print(f"  Final y_com  = {x_final[:, 1].mean():.4f} m")

    if not no_anim:
        animate_simulation(
            traj, interval=60,
            title=f"Falling Block ({'adaptive' if adaptive else f'dt={dt:.4f}'})",
            save_path=save_path,
        )
    plot_trajectory(traj, quantity="y_min", title="Block bottom height over time")
    if adaptive:
        plot_newton_iters(traj)
    else:
        plot_final_frame(traj, title="Falling block — final frame")
    return traj


def run_sliding(
    adaptive: bool = False,
    dt: float = 0.01,
    T_end: float = 2.0,
    save_path: str | None = None,
    no_anim: bool = False,
) -> dict:
    print("=" * 55)
    print("  Sliding Block Experiment")
    print("=" * 55)

    # Parameters matching adaptive_stepping_2d/experiments/sliding_block.py
    setup = setup_sliding_block(
        width=2.0,
        height=2.0,
        nx=8,
        ny=8,
        E_young=200_000.0,
        nu=0.3,
        rho=1000.0,
        v0x=2.0,
        g_mag=10.0,
        kappa=20_000.0,
        dhat=0.02,
    )

    if adaptive:
        print(f"Running adaptive stepping (T={T_end}s) ...")
        traj = run_adaptive(
            setup, dt0=dt, T_end=T_end, tol=1e-4,
            dt_min=1e-5, dt_max=0.1, target_iters=8, verbose=True,
        )
    else:
        print(f"Running fixed stepping dt={dt} (T={T_end}s) ...")
        traj = run_fixed(setup, dt=dt, T_end=T_end, tol=1e-4, verbose=True)

    x_final = traj["x"][-1]
    print(f"\nDone. {len(traj['t'])-1} steps recorded.")
    print(f"  Final x_com  = {x_final[:, 0].mean():.4f} m")
    print(f"  Final y_min  = {x_final[:, 1].min():.4f} m")

    if not no_anim:
        animate_simulation(
            traj, interval=60,
            title=f"Sliding Block ({'adaptive' if adaptive else f'dt={dt}'})",
            save_path=save_path,
        )
    plot_trajectory(traj, quantity="x_com", title="Block x centre-of-mass over time")
    if adaptive:
        plot_newton_iters(traj)
    else:
        plot_final_frame(traj, title="Sliding block — final frame")
    return traj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2D FEM Barrier-Contact Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment", "-e",
        choices=["falling", "sliding"],
        default="falling",
        help="Which experiment to run (default: falling)",
    )
    parser.add_argument(
        "--adaptive", "-a",
        action="store_true",
        help="Use adaptive time stepping",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step size (default: Courant falling / 0.01 sliding)",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=None,
        dest="T_end",
        help="End time in seconds (default: 6.0 falling / 2.0 sliding)",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Save animation to PATH (.gif or .mp4)",
    )
    parser.add_argument(
        "--no-anim",
        action="store_true",
        help="Skip animation (faster, just show static plots)",
    )
    args = parser.parse_args()

    if args.experiment == "falling":
        run_falling(
            adaptive=args.adaptive,
            dt=args.dt,  # None → Courant condition applied inside
            T_end=args.T_end if args.T_end is not None else 6.0,
            save_path=args.save,
            no_anim=args.no_anim,
        )
    elif args.experiment == "sliding":
        run_sliding(
            adaptive=args.adaptive,
            dt=args.dt if args.dt is not None else 0.01,
            T_end=args.T_end if args.T_end is not None else 2.0,
            save_path=args.save,
            no_anim=args.no_anim,
        )


if __name__ == "__main__":
    main()
