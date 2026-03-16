"""Differentiable trajectory optimization for Newton Ant via wp.Tape.

Uses Warp's tape-based autodiff to optimize an action sequence directly
through the differentiable Featherstone physics.
"""
import argparse
import os
import numpy as np
import warp as wp
import warp.optim
import newton


@wp.kernel
def apply_actions_kernel(
    actions: wp.array(dtype=float),
    joint_f: wp.array(dtype=float),
    action_scale: float,
    num_act: int,
    act_offset: int,
    total_dofs: int,
    frame_offset: int,
):
    """Write actions for a specific frame into joint_f at correct DOF offsets."""
    env_id, act_id = wp.tid()
    src_idx = frame_offset + env_id * num_act + act_id
    dst_idx = env_id * total_dofs + act_offset + act_id
    joint_f[dst_idx] = actions[src_idx] * action_scale


@wp.kernel
def compute_cost_kernel(
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    num_q: int,
    num_qd: int,
    up_axis: int,
    fwd_axis: int,
    termination_height: float,
    cost: wp.array(dtype=float),
):
    """Compute negative reward as cost for one timestep."""
    env_id = wp.tid()
    q_offset = env_id * num_q
    qd_offset = env_id * num_qd

    height = joint_q[q_offset + up_axis]
    fwd_vel = joint_qd[qd_offset + 3 + fwd_axis]  # lin_vel starts at index 3

    reward = fwd_vel + (height - termination_height)
    wp.atomic_add(cost, env_id, -reward)


def main(args):
    num_envs = args.num_envs
    horizon = args.horizon
    sim_substeps = 16
    frame_dt = 1.0 / 60.0
    sim_dt = frame_dt / sim_substeps

    # Build model (matching rewarped settings)
    ant = newton.ModelBuilder()
    ant.default_shape_cfg.ke = 4.0e3
    ant.default_shape_cfg.kd = 1.0e3
    ant.default_shape_cfg.kf = 3.0e2
    ant.default_shape_cfg.mu = 0.75
    ant.default_shape_cfg.restitution = 0.0
    ant.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1,
    )

    asset_dir = os.path.join(os.path.dirname(__file__), "assets")
    ant.add_mjcf(
        os.path.join(asset_dir, "ant.xml"),
        enable_self_collisions=True,
        armature_scale=5,
    )
    ant.joint_q[7:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
    ant.joint_q[2] = 0.75

    builder = newton.ModelBuilder()
    builder.replicate(ant, num_envs)
    builder.add_ground_plane()
    model = builder.finalize(device="cuda:0", requires_grad=True)

    solver = newton.solvers.SolverFeatherstone(
        model, angular_damping=0.0, update_mass_matrix_interval=sim_substeps,
    )

    # DOF layout
    num_q_per_env = model.joint_q.shape[0] // num_envs
    num_qd_per_env = model.joint_qd.shape[0] // num_envs
    num_act = 8
    act_offset = 6  # revolute joints start at DOF index 6
    total_dofs = num_qd_per_env

    # Action parameters to optimize: one action per frame per env
    actions_flat = wp.zeros(horizon * num_envs * num_act, dtype=float, requires_grad=True)
    cost = wp.zeros(num_envs, dtype=float, requires_grad=True)

    # Allocate states: one per substep per frame + 1
    total_substeps = horizon * sim_substeps
    states = [model.state() for _ in range(total_substeps + 1)]
    controls = [model.control() for _ in range(total_substeps)]

    optimizer = warp.optim.Adam([actions_flat], lr=args.lr)

    print(f"Trajectory optimization | {num_envs} envs | horizon={horizon} | {args.iterations} iters")
    print(f"DOF layout: {num_q_per_env} q, {num_qd_per_env} qd per env")
    print(f"Total states allocated: {total_substeps + 1}")
    print(f"{'Iter':>6} {'MeanCost':>12}")
    print("-" * 20)

    for it in range(args.iterations):
        # Reset to default state
        states[0].joint_q.assign(model.joint_q)
        states[0].joint_qd.assign(model.joint_qd)
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        cost.zero_()

        tape = wp.Tape()
        with tape:
            step_idx = 0
            for t in range(horizon):
                frame_offset = t * num_envs * num_act

                for s in range(sim_substeps):
                    # Apply same action for all substeps within a frame
                    wp.launch(
                        apply_actions_kernel,
                        dim=(num_envs, num_act),
                        inputs=[
                            actions_flat,
                            controls[step_idx].joint_f,
                            200.0,
                            num_act,
                            act_offset,
                            total_dofs,
                            frame_offset,
                        ],
                    )

                    states[step_idx].clear_forces()
                    solver.step(
                        states[step_idx],
                        states[step_idx + 1],
                        controls[step_idx],
                        None,  # skip contacts for differentiable mode
                        sim_dt,
                    )
                    step_idx += 1

                # Compute cost at end of this frame
                wp.launch(
                    compute_cost_kernel,
                    dim=num_envs,
                    inputs=[
                        states[step_idx].joint_q,
                        states[step_idx].joint_qd,
                        num_q_per_env,
                        num_qd_per_env,
                        2,     # up_axis = Z
                        0,     # fwd_axis = X
                        0.27,  # termination_height
                    ],
                    outputs=[cost],
                )

        cost.grad.fill_(1.0)
        tape.backward()

        optimizer.step([actions_flat.grad])
        tape.zero()

        # Clamp actions to [-1, 1]
        wp.synchronize()
        actions_np = actions_flat.numpy()
        actions_np = np.clip(actions_np, -1.0, 1.0)
        actions_flat.assign(actions_np)

        if (it + 1) % 10 == 0 or it == 0:
            mean_cost = cost.numpy().mean()
            print(f"{it + 1:>6} {mean_cost:>12.4f}")

    print("Optimization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    main(args)
