"""Visualization for 2D FEM simulations.

Uses Polyscope for interactive animation (matching reference visualizer.py)
and Matplotlib for static diagnostic plots.
"""

from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


# ---------------------------------------------------------------------------
# Polyscope animation  (matches adaptive_stepping_2d/visualizer.py exactly)
# ---------------------------------------------------------------------------


def _lift_2d_to_3d(x2d: np.ndarray, z: float = 0.0) -> np.ndarray:
    """Convert (n, 2) vertex positions to (n, 3) by appending a z column."""
    n = len(x2d)
    x3d = np.zeros((n, 3), dtype=float)
    x3d[:, :2] = x2d
    x3d[:, 2] = z
    return x3d


def animate_simulation(
    trajectory: dict,
    fps: float = 16.0,
    title: str = "2D FEM Simulation",
    save_path: str | None = None,
    **_kwargs,
) -> None:
    """Interactive Polyscope animation of mesh deformation.

    Visual style matches the reference PolyscopeVisualizer:
    - Triangulated mesh lifted to 3D (z = 0).
    - Per-vertex height coloring using the 'Blues' colormap.
    - Ground plane at y = 0.
    - ImGui play/pause/step/reset controls and frame slider.

    Parameters
    ----------
    trajectory : dict
        Output of ``run_fixed`` or ``run_adaptive``.
    fps : float
        Target playback speed (frames per second).
    title : str
        Polyscope window title.
    save_path : str or None
        If given, a screenshot is saved to this path (.png) after showing.
    """
    import polyscope as ps

    x_hist = trajectory["x"]      # list of (n_nodes, 2)
    t_hist = trajectory["t"]      # (n_steps+1,)
    triangles = trajectory["mesh"].triangles

    n_frames = len(x_hist)

    # --- Polyscope init ---
    ps.init()
    ps.set_program_name(title)
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    # --- Ground plane visual ---
    all_x_coords = np.concatenate([f[:, 0] for f in x_hist])
    gnd_ext = max(abs(all_x_coords.min()), abs(all_x_coords.max())) + 2.0
    gnd_verts = np.array([
        [-gnd_ext, 0.0, -gnd_ext],
        [ gnd_ext, 0.0, -gnd_ext],
        [ gnd_ext, 0.0,  gnd_ext],
        [-gnd_ext, 0.0,  gnd_ext],
    ], dtype=float)
    gnd_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    gnd = ps.register_surface_mesh("ground", gnd_verts, gnd_faces, smooth_shade=False)
    gnd.set_color((0.75, 0.75, 0.75))
    gnd.set_transparency(0.5)

    # --- Block mesh ---
    verts0 = _lift_2d_to_3d(x_hist[0])
    faces = np.asarray(triangles, dtype=int)
    ps_mesh = ps.register_surface_mesh("block", verts0, faces, smooth_shade=True)
    ps_mesh.set_color((0.2, 0.5, 0.9))

    def _update_verts(x2d: np.ndarray) -> None:
        ps_mesh.update_vertex_positions(_lift_2d_to_3d(x2d))

    _update_verts(x_hist[0])

    # --- Callback state ---
    state = {"frame": 0, "playing": True}
    last_time = [time.time()]
    frame_interval = 1.0 / max(fps, 1.0)

    try:
        import polyscope.imgui as imgui  # bundled with polyscope ≥ 2.x
        _have_imgui = True
    except ImportError:
        try:
            import imgui  # type: ignore
            _have_imgui = True
        except ImportError:
            _have_imgui = False

    def _callback() -> None:
        now = time.time()
        dt = now - last_time[0]

        if _have_imgui:
            changed, new_frame = imgui.slider_int(
                "Frame", state["frame"], 0, n_frames - 1
            )
            if changed:
                state["frame"] = new_frame

            imgui.text(f"t = {t_hist[state['frame']]:.4f} s")

            if imgui.button("Play" if not state["playing"] else "Pause"):
                state["playing"] = not state["playing"]
            imgui.same_line()
            if imgui.button("Step"):
                state["frame"] = min(state["frame"] + 1, n_frames - 1)
                state["playing"] = False
            imgui.same_line()
            if imgui.button("Reset"):
                state["frame"] = 0

        # Auto-advance
        if state["playing"] and dt >= frame_interval:
            state["frame"] = (state["frame"] + 1) % n_frames
            last_time[0] = now

        _update_verts(x_hist[state["frame"]])

    ps.set_user_callback(_callback)
    ps.show()

    if save_path is not None:
        ps.screenshot(save_path)
        print(f"Screenshot saved to {save_path}")


# ---------------------------------------------------------------------------
# Static Matplotlib plots
# ---------------------------------------------------------------------------


def save_gif(
    trajectory: dict,
    path: str,
    fps: int = 16,
    title: str = "",
) -> None:
    """Render the simulation to a GIF file (no display window).

    Uses the Blues height colormap to match the Polyscope visual style.

    Parameters
    ----------
    trajectory : dict
        Output of ``run_fixed`` or ``run_adaptive``.
    path : str
        Output file path, e.g. ``"falling_block.gif"``.
    fps : int
        Frames per second in the output GIF.
    title : str
        Figure title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    x_hist = trajectory["x"]
    t_hist = trajectory["t"]
    triangles = trajectory["mesh"].triangles

    all_x = np.concatenate([f[:, 0] for f in x_hist])
    all_y = np.concatenate([f[:, 1] for f in x_hist])
    margin_x = max((all_x.max() - all_x.min()) * 0.15, 0.5)
    margin_y = max((all_y.max() - all_y.min()) * 0.15, 0.5)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(all_x.min() - margin_x, all_x.max() + margin_x)
    ax.set_ylim(min(-0.5, all_y.min() - margin_y), all_y.max() + margin_y)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title)

    xlim = ax.get_xlim()
    ax.axhline(y=0.0, color="#8B4513", linewidth=2.5, zorder=0)
    ax.fill_between(
        [xlim[0], xlim[1]], [0.0, 0.0], [all_y.min() - margin_y] * 2,
        color="#D2B48C", alpha=0.3, zorder=0,
    )

    coll = PolyCollection(
        x_hist[0][triangles],
        facecolor="#3380E6",
        edgecolor="#1a3d6b",
        linewidth=0.4,
        alpha=0.9,
        zorder=2,
    )
    ax.add_collection(coll)

    time_text = ax.text(
        0.02, 0.96, f"t = {t_hist[0]:.3f} s",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    def _update(frame):
        coll.set_verts(x_hist[frame][triangles])
        time_text.set_text(f"t = {t_hist[frame]:.3f} s")
        return coll, time_text

    plt.tight_layout()
    anim = animation.FuncAnimation(
        fig, _update, frames=len(x_hist), interval=1000 // fps, blit=True,
    )

    print(f"Saving {path} ...")
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved {path}  ({len(x_hist)} frames @ {fps} fps)")


def plot_trajectory(
    trajectory: dict,
    quantity: str = "y_min",
    title: str | None = None,
) -> None:
    """Plot a scalar quantity over time.

    Parameters
    ----------
    trajectory : dict
        Output of run_fixed / run_adaptive.
    quantity : str
        One of ``"y_min"``, ``"y_com"``, ``"x_com"``.
    title : str or None
        Plot title.
    """
    t = trajectory["t"]
    x_hist = trajectory["x"]

    labels = {
        "y_min": ("Minimum y (m)", "Block bottom height over time"),
        "y_com": ("y centre of mass (m)", "Centre of mass height over time"),
        "x_com": ("x centre of mass (m)", "Centre of mass x-position over time"),
    }
    ylabel, default_title = labels.get(quantity, ("value", quantity))

    if quantity == "y_min":
        vals = [x[:, 1].min() for x in x_hist]
    elif quantity == "y_com":
        vals = [x[:, 1].mean() for x in x_hist]
    elif quantity == "x_com":
        vals = [x[:, 0].mean() for x in x_hist]
    else:
        raise ValueError(f"Unknown quantity '{quantity}'")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, vals, "b-", linewidth=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title or default_title)
    ax.grid(True, alpha=0.3)

    dt_arr = trajectory.get("dt")
    if dt_arr is not None and len(dt_arr) > 0:
        ax2 = ax.twinx()
        t_dt = t[1: len(dt_arr) + 1]
        ax2.semilogy(t_dt, dt_arr, "r-", alpha=0.45, linewidth=1.2, label="dt")
        ax2.set_ylabel("dt (s)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax.plot([], [], "r-", alpha=0.45, label="dt (right axis)")
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_newton_iters(
    trajectory: dict,
    title: str = "Newton iterations per step",
) -> None:
    """Bar chart of Newton iterations per time step."""
    stats = trajectory.get("stats", [])
    if not stats:
        print("No stats to plot.")
        return

    iters = [s["newton_iters"] for s in stats]
    t = trajectory["t"]

    if len(t) > 1:
        widths = np.diff(t[: len(iters) + 1])
    else:
        widths = np.ones(len(iters)) * 0.01
    bar_t = t[1: len(iters) + 1]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(bar_t, iters, width=widths * 0.9, align="edge", alpha=0.75, color="#3380E6")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Newton iterations")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_final_frame(
    trajectory: dict,
    title: str = "Final configuration",
    show_ground: bool = True,
) -> None:
    """Plot the final deformed mesh with uniform color."""
    x2d = trajectory["x"][-1]
    triangles = trajectory["mesh"].triangles

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    if show_ground:
        xlim = (x2d[:, 0].min() - 0.5, x2d[:, 0].max() + 0.5)
        ax.axhline(y=0.0, color="#8B4513", linewidth=2.5)
        ax.fill_between(xlim, [0.0, 0.0], [-0.3, -0.3], color="#D2B48C", alpha=0.3)
        ax.set_xlim(*xlim)

    coll = PolyCollection(
        x2d[triangles],
        facecolor="#3380E6",
        edgecolor="#1a3d6b",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.add_collection(coll)
    ax.autoscale_view()
    plt.tight_layout()
    plt.show()
