"""3D penalty contact experiment with Polyscope visualization.

Entry point: python -m PenaltyContact.experiment [--experiment falling|spinning|sliding]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .contact_model import ContactParams
from .simulation import RigidBody3D, CUBE_FACES, simulate, quat_from_axis_angle


# ---------------------------------------------------------------------------
# Preconfigured experiments
# ---------------------------------------------------------------------------

def make_experiment(name: str) -> tuple[RigidBody3D, ContactParams, float]:
    """Return (body, params, T_end) for a named experiment."""
    params = ContactParams()

    if name == "falling":
        # Cube at y=2m, slight tilt, small horizontal velocity
        q = quat_from_axis_angle(np.array([1.0, 0.0, 1.0]), 0.15)
        body = RigidBody3D(
            pos=[0.0, 2.0, 0.0],
            vel=[0.5, 0.0, 0.0],
            quat=q,
            omega=[0.0, 0.0, 0.0],
            mass=1.0,
            size=0.5,
        )
        return body, params, 3.0

    elif name == "spinning":
        # Cube at y=3m, large angular velocity
        q = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.0)
        body = RigidBody3D(
            pos=[0.0, 3.0, 0.0],
            vel=[0.0, 0.0, 0.0],
            quat=q,
            omega=[0.0, 5.0, 2.0],
            mass=1.0,
            size=0.5,
        )
        return body, params, 5.0

    elif name == "sliding":
        # Cube resting on ground with horizontal velocity
        h = 0.5 / 2.0  # half-size
        q = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.0)
        body = RigidBody3D(
            pos=[0.0, h + 0.001, 0.0],  # just above ground
            vel=[3.0, 0.0, 0.0],
            quat=q,
            omega=[0.0, 0.0, 0.0],
            mass=1.0,
            size=0.5,
        )
        return body, params, 3.0

    else:
        raise ValueError(f"Unknown experiment: {name!r}")


# ---------------------------------------------------------------------------
# Polyscope visualization
# ---------------------------------------------------------------------------

def run_visualization(result: dict, fps: float = 30.0) -> None:
    """Interactive Polyscope animation of the simulation result."""
    import polyscope as ps

    verts_hist = result["vertices_history"]
    t_hist = result["t"]
    n_frames = len(verts_hist)

    # --- Polyscope init ---
    ps.init()
    ps.set_program_name("3D Penalty Contact")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")

    # --- Compute scene bounds from full trajectory ---
    all_verts = np.concatenate(verts_hist, axis=0)
    bbox_min = all_verts.min(axis=0)
    bbox_max = all_verts.max(axis=0)
    scene_center = 0.5 * (bbox_min + bbox_max)
    scene_size = np.linalg.norm(bbox_max - bbox_min)

    # --- Ground plane (sized to trajectory, not too large) ---
    ext = max(
        abs(bbox_min[0]), abs(bbox_max[0]),
        abs(bbox_min[2]), abs(bbox_max[2]),
    ) + 1.0
    gnd_verts = np.array([
        [-ext, 0.0, -ext],
        [ ext, 0.0, -ext],
        [ ext, 0.0,  ext],
        [-ext, 0.0,  ext],
    ])
    gnd_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    gnd = ps.register_surface_mesh("ground", gnd_verts, gnd_faces, smooth_shade=False)
    gnd.set_color((0.75, 0.75, 0.75))
    gnd.set_transparency(0.5)

    # --- Cube mesh ---
    cube = ps.register_surface_mesh("cube", verts_hist[0], CUBE_FACES, smooth_shade=False)
    cube.set_color((0.2, 0.5, 0.9))
    cube.set_edge_width(1.0)

    # --- Camera: frame the full trajectory ---
    eye = scene_center + np.array([1.0, 0.8, 1.3]) * scene_size * 0.7
    ps.look_at(eye, scene_center)

    # --- ImGui callback state ---
    state = {"frame": 0, "playing": True}
    last_time = [time.time()]
    frame_interval = 1.0 / max(fps, 1.0)

    try:
        import polyscope.imgui as imgui
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
            changed, new_frame = imgui.SliderInt(
                "Frame", state["frame"], 0, n_frames - 1
            )
            if changed:
                state["frame"] = new_frame

            imgui.TextUnformatted(f"t = {t_hist[state['frame']]:.4f} s")
            imgui.TextUnformatted(f"Frame {state['frame']} / {n_frames - 1}")

            if imgui.Button("Play" if not state["playing"] else "Pause"):
                state["playing"] = not state["playing"]
            imgui.SameLine()
            if imgui.Button("Step"):
                state["frame"] = min(state["frame"] + 1, n_frames - 1)
                state["playing"] = False
            imgui.SameLine()
            if imgui.Button("Reset"):
                state["frame"] = 0

        # Auto-advance
        if state["playing"] and dt >= frame_interval:
            state["frame"] = (state["frame"] + 1) % n_frames
            last_time[0] = now

        cube.update_vertex_positions(verts_hist[state["frame"]])

    ps.set_user_callback(_callback)
    ps.show()


# ---------------------------------------------------------------------------
# GIF export via headless polyscope rendering
# ---------------------------------------------------------------------------

def save_gif(
    result: dict,
    path: str = "PenaltyContact/penalty_contact_3d.gif",
    fps: int = 30,
    resolution: tuple[int, int] = (800, 600),
) -> None:
    """Render simulation frames with polyscope and save as GIF."""
    import polyscope as ps
    from PIL import Image

    verts_hist = result["vertices_history"]
    n_frames = len(verts_hist)

    # --- Headless polyscope init ---
    ps.set_allow_headless_backends(True)
    ps.init()
    ps.set_program_name("3D Penalty Contact")
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_screenshot_extension(".png")
    ps.set_open_imgui_window_for_user_callback(False)

    w, h = resolution
    ps.set_window_size(w, h)

    # --- Ground plane ---
    all_verts = np.concatenate(verts_hist, axis=0)
    ext = max(
        abs(all_verts[:, 0].min()), abs(all_verts[:, 0].max()),
        abs(all_verts[:, 2].min()), abs(all_verts[:, 2].max()),
    ) + 2.0
    gnd_verts = np.array([
        [-ext, 0.0, -ext],
        [ ext, 0.0, -ext],
        [ ext, 0.0,  ext],
        [-ext, 0.0,  ext],
    ])
    gnd_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    gnd = ps.register_surface_mesh("ground", gnd_verts, gnd_faces, smooth_shade=False)
    gnd.set_color((0.75, 0.75, 0.75))
    gnd.set_transparency(0.5)

    # --- Cube ---
    cube = ps.register_surface_mesh("cube", verts_hist[0], CUBE_FACES, smooth_shade=False)
    cube.set_color((0.2, 0.5, 0.9))
    cube.set_edge_width(1.0)

    # --- Camera: frame the full trajectory nicely ---
    cm_trajectory = result["cm_pos"]
    bbox_min = all_verts.min(axis=0)
    bbox_max = all_verts.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    scene_size = np.linalg.norm(bbox_max - bbox_min)
    # Place camera looking from front-right-above, scaled to scene
    eye = center + np.array([1.0, 0.8, 1.3]) * scene_size * 0.7
    ps.look_at(eye, center)

    # --- Render each frame ---
    print(f"Rendering {n_frames} frames to {path} ...")
    images = []
    for i in range(n_frames):
        cube.update_vertex_positions(verts_hist[i])
        ps.frame_tick()
        buf = ps.screenshot_to_buffer(transparent_bg=False)
        img = Image.fromarray(buf)
        images.append(img)

    # Save GIF
    duration_ms = int(1000 / fps)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved {path}  ({n_frames} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3D Penalty Contact Demo")
    parser.add_argument(
        "--experiment", "-e",
        choices=["falling", "spinning", "sliding"],
        default="falling",
        help="Preconfigured experiment (default: falling)",
    )
    parser.add_argument("--dt", type=float, default=5e-4, help="Time step")
    parser.add_argument("--ke", type=float, default=None, help="Elastic stiffness")
    parser.add_argument("--kd", type=float, default=None, help="Damping")
    parser.add_argument("--mu", type=float, default=None, help="Friction coefficient")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS")
    parser.add_argument("--save-every", type=int, default=20, help="Save every N steps")
    parser.add_argument("--save-gif", type=str, default=None,
                        help="Save GIF to this path instead of showing interactive viewer")
    args = parser.parse_args()

    body, params, T_end = make_experiment(args.experiment)

    # Apply CLI overrides
    if args.ke is not None:
        params.ke = args.ke
    if args.kd is not None:
        params.kd = args.kd
    if args.mu is not None:
        params.mu = args.mu

    print(f"Experiment: {args.experiment}")
    print(f"  pos={body.pos}, vel={body.vel}")
    print(f"  omega={body.omega}, mass={body.mass}, size={body.size}")
    print(f"  ke={params.ke}, kd={params.kd}, kf={params.kf}, mu={params.mu}")
    print(f"  dt={args.dt}, T_end={T_end}")
    print("Simulating...")

    result = simulate(
        body,
        params=params,
        dt=args.dt,
        T_end=T_end,
        save_every=args.save_every,
    )

    n_frames = len(result["t"])
    print(f"Done: {n_frames} frames saved")
    print(f"Final CM height: {result['cm_pos'][-1, 1]:.4f} m")

    # Check max penetration
    all_verts = np.concatenate(result["vertices_history"], axis=0)
    min_y = all_verts[:, 1].min()
    print(f"Max penetration: {max(0, -min_y):.5f} m")

    if args.save_gif:
        save_gif(result, path=args.save_gif, fps=int(args.fps))
    else:
        run_visualization(result, fps=args.fps)


if __name__ == "__main__":
    main()
