#!/usr/bin/env python3
"""Falling block experiment — standalone runner.

A 10×10 block falls under gravity onto flat ground at y=0.
Parameters match adaptive_stepping_2d/experiments/falling_block.py (Section 9.1.2).

Run from any directory:
    python BarrierContact/experiments/falling_block.py
    python BarrierContact/experiments/falling_block.py --adaptive
    python BarrierContact/experiments/falling_block.py --T 3.0 --out my.gif
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from BarrierContact.simple_simulator import run_adaptive, run_fixed, setup_falling_block
from BarrierContact.visualize import save_gif


def main(
    T_end: float = 6.0,
    dt: float | None = None,
    adaptive: bool = False,
    out: str = "falling_block.gif",
) -> dict:
    print("=" * 55)
    print("  Falling Block — Section 9.1.2")
    print("=" * 55)

    E_young, rho, width, nx = 1800.0, 1.0, 10.0, 10

    setup = setup_falling_block(
        width=width, height=10.0, nx=nx, ny=10,
        E_young=E_young, nu=0.2, rho=rho,
        h0=5.0, g_mag=10.0,
        # kappa=0.1*E=180, dhat=dx=1.0 (defaults)
    )

    if dt is None:
        dt = 0.75 * (width / nx) / np.sqrt(E_young / rho)
        print(f"  Courant dt = {dt:.6f} s")

    if adaptive:
        print(f"Adaptive stepping  T={T_end}s ...")
        traj = run_adaptive(setup, dt0=dt, T_end=T_end, tol=1e-4,
                            dt_min=1e-5, dt_max=0.1, target_iters=8, verbose=True)
    else:
        print(f"Fixed stepping  dt={dt:.6f}  T={T_end}s ...")
        traj = run_fixed(setup, dt=dt, T_end=T_end, tol=1e-4, verbose=True)

    x_final = traj["x"][-1]
    print(f"\nDone. {len(traj['t'])-1} steps.")
    print(f"  Final y_min = {x_final[:, 1].min():.4f} m")
    print(f"  Final y_com = {x_final[:, 1].mean():.4f} m")

    label = "adaptive" if adaptive else f"dt={dt:.4f}"
    save_gif(traj, out, fps=16, title=f"Falling Block ({label})")
    return traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Falling block FEM experiment")
    parser.add_argument("--T", type=float, default=6.0, dest="T_end",
                        help="End time in seconds (default: 6.0)")
    parser.add_argument("--dt", type=float, default=None,
                        help="Time step (default: Courant condition)")
    parser.add_argument("--adaptive", "-a", action="store_true",
                        help="Use adaptive time stepping")
    parser.add_argument("--out", default="falling_block.gif",
                        help="Output GIF path (default: falling_block.gif)")
    args = parser.parse_args()
    main(T_end=args.T_end, dt=args.dt, adaptive=args.adaptive, out=args.out)
