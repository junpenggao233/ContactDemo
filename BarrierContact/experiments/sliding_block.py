#!/usr/bin/env python3
"""Sliding block experiment — standalone runner.

A 2×2 block slides right with initial velocity v0=[2, 0] on flat ground.
Parameters match adaptive_stepping_2d/experiments/sliding_block.py (Section 9.2).

Run from any directory:
    python BarrierContact/experiments/sliding_block.py
    python BarrierContact/experiments/sliding_block.py --adaptive
    python BarrierContact/experiments/sliding_block.py --T 1.0 --out my.gif
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from BarrierContact.simple_simulator import run_adaptive, run_fixed, setup_sliding_block
from BarrierContact.visualize import save_gif


def main(
    T_end: float = 2.0,
    dt: float = 0.01,
    adaptive: bool = False,
    out: str = "sliding_block.gif",
) -> dict:
    print("=" * 55)
    print("  Sliding Block — Section 9.2")
    print("=" * 55)

    setup = setup_sliding_block(
        width=2.0, height=2.0, nx=8, ny=8,
        E_young=200_000.0, nu=0.3, rho=1000.0,
        v0x=2.0, g_mag=10.0,
        kappa=20_000.0, dhat=0.02,
    )

    if adaptive:
        print(f"Adaptive stepping  T={T_end}s ...")
        traj = run_adaptive(setup, dt0=dt, T_end=T_end, tol=1e-4,
                            dt_min=1e-5, dt_max=0.1, target_iters=8, verbose=True)
    else:
        print(f"Fixed stepping  dt={dt}  T={T_end}s ...")
        traj = run_fixed(setup, dt=dt, T_end=T_end, tol=1e-4, verbose=True)

    x_final = traj["x"][-1]
    print(f"\nDone. {len(traj['t'])-1} steps.")
    print(f"  Final x_com = {x_final[:, 0].mean():.4f} m")
    print(f"  Final y_min = {x_final[:, 1].min():.4f} m")

    label = "adaptive" if adaptive else f"dt={dt}"
    save_gif(traj, out, fps=16, title=f"Sliding Block ({label})")
    return traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sliding block FEM experiment")
    parser.add_argument("--T", type=float, default=2.0, dest="T_end",
                        help="End time in seconds (default: 2.0)")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Time step (default: 0.01)")
    parser.add_argument("--adaptive", "-a", action="store_true",
                        help="Use adaptive time stepping")
    parser.add_argument("--out", default="sliding_block.gif",
                        help="Output GIF path (default: sliding_block.gif)")
    args = parser.parse_args()
    main(T_end=args.T_end, dt=args.dt, adaptive=args.adaptive, out=args.out)
