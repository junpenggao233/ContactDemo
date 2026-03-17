"""Evaluate a trained PPO checkpoint and optionally export USD."""

import argparse
import numpy as np
import torch
from ant_env import AntEnv
from run_ppo import ActorCritic, RunningMeanStd


def add_grid_ground(viewer, size=50.0, spacing=1.0):
    """Add a grid ground plane as line segments to the USD stage."""
    from pxr import UsdGeom, Gf, Sdf

    stage = viewer.stage
    grid_path = "/root/grid_ground"
    xform = UsdGeom.Xform.Define(stage, grid_path)

    # Build grid lines
    points = []
    indices = []
    counts = []
    n = int(size / spacing)
    idx = 0

    for i in range(-n, n + 1):
        coord = i * spacing
        # Line along X
        points.append((-size, coord, 0.0))
        points.append((size, coord, 0.0))
        indices.extend([idx, idx + 1])
        counts.append(2)
        idx += 2
        # Line along Y
        points.append((coord, -size, 0.0))
        points.append((coord, size, 0.0))
        indices.extend([idx, idx + 1])
        counts.append(2)
        idx += 2

    curves = UsdGeom.BasisCurves.Define(stage, f"{grid_path}/lines")
    curves.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
    curves.GetCurveVertexCountsAttr().Set(counts)
    curves.GetTypeAttr().Set("linear")
    curves.GetWidthsAttr().Set([0.01] * len(points))
    curves.GetDisplayColorAttr().Set([Gf.Vec3f(0.4, 0.4, 0.4)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to PPO checkpoint (.pt)")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=1000, help="AntRun episode length")
    parser.add_argument("--usd", type=str, default=None, help="Output USD file path")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    env = AntEnv(num_envs=args.num_envs, device=args.device)
    model = ActorCritic(env.num_obs, env.num_act).to(args.device)
    obs_rms = RunningMeanStd(env.num_obs).to(args.device)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        obs_rms.load_state_dict(ckpt["running_mean_std"])
        print(
            f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
            f"reward {ckpt.get('mean_reward', '?')})"
        )
    else:
        model.load_state_dict(ckpt)
        print("Loaded legacy checkpoint (no obs_rms)")

    model.eval()
    obs_rms.eval()

    viewer = None
    if args.usd:
        from newton.viewer import ViewerUSD

        viewer = ViewerUSD(args.usd, fps=60, up_axis="Z", num_frames=args.num_steps)
        viewer.set_model(env.model)
        add_grid_ground(viewer)

    obs = env.reset()
    ep_rewards = torch.zeros(args.num_envs, device=args.device)
    completed = []

    # Track world position for displacement verification
    import warp as wp
    joint_q = wp.to_torch(env.state_0.joint_q).view(args.num_envs, -1)
    start_pos = joint_q[:, 0:3].clone()

    print(f"Evaluating {args.checkpoint} for {args.num_steps} steps...")

    with torch.no_grad():
        for step in range(args.num_steps):
            proc_obs = obs_rms(obs)
            actions = model.act_inference(proc_obs).clamp(-1, 1)
            obs, rew, done, info = env.step(actions)
            ep_rewards += rew

            if viewer:
                viewer.begin_frame(step * env.frame_dt)
                viewer.log_state(env.state_0)
                viewer.end_frame()

            finished = done.nonzero(as_tuple=False).squeeze(-1)
            if len(finished) > 0:
                completed.extend(ep_rewards[finished].tolist())
                ep_rewards[finished] = 0.0
                # Reset start position for restarted envs
                jq = wp.to_torch(env.state_0.joint_q).view(args.num_envs, -1)
                start_pos[finished] = jq[finished, 0:3].clone()

            if (step + 1) % 100 == 0:
                height = obs[:, 0].mean().item()
                fwd_vel = obs[:, 5].mean().item()
                jq = wp.to_torch(env.state_0.joint_q).view(args.num_envs, -1)
                disp = (jq[:, 0:3] - start_pos).norm(dim=-1).mean().item()
                print(
                    f"  step {step + 1:>5}: height={height:.3f}, fwd_vel={fwd_vel:.3f}, "
                    f"disp={disp:.1f}m, reward={rew.mean().item():.3f}"
                )

    if viewer:
        viewer.close()
        print(f"\nUSD saved to {args.usd}")

    # Final displacement
    jq = wp.to_torch(env.state_0.joint_q).view(args.num_envs, -1)
    final_disp = (jq[:, 0:3] - start_pos).norm(dim=-1)
    print(f"\nFinal displacement: {final_disp.mean().item():.1f}m (per env: {final_disp.tolist()})")

    if completed:
        print(f"Completed episodes: {len(completed)}")
        print(f"Mean episode reward: {sum(completed) / len(completed):.1f}")


if __name__ == "__main__":
    main()
