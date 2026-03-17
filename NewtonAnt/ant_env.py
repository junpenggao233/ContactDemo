import os

import newton
import torch
import warp as wp
from torch_utils import normalize, quat_conjugate, quat_mul, quat_rotate


class AntEnv:
    """Gym-like Ant environment using Newton Physics with Featherstone solver.

    Replicates the rewarped DFlex Ant environment settings exactly:
    https://github.com/rewarped/rewarped/blob/main/rewarped/envs/dflex/ant.py
    """

    def __init__(self, num_envs=64, device="cuda:0", requires_grad=False):
        self.num_envs = num_envs
        self.device = device
        self.requires_grad = requires_grad
        self.num_obs = 37
        self.num_act = 8
        self.episode_length = 1000
        self.termination_height = 0.27
        self.action_scale = 200.0
        self.action_penalty = 0.0
        self.joint_vel_obs_scaling = 0.1
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 16  # rewarped: sim_substeps_featherstone = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Degrees of freedom per env
        self.num_joint_q = 15  # 7 (free root) + 8 (hinges)
        self.num_joint_qd = 14  # 6 (free root) + 8 (hinges)

        self._build_model()
        self._init_buffers()

    def _build_model(self):
        ant = newton.ModelBuilder()

        # Contact parameters (matching rewarped exactly)
        ant.default_shape_cfg.ke = 4.0e3  # contact_ke
        ant.default_shape_cfg.kd = 1.0e3  # contact_kd
        ant.default_shape_cfg.kf = 3.0e2  # contact_kf
        ant.default_shape_cfg.mu = 0.75  # contact_mu
        ant.default_shape_cfg.restitution = 0.0  # contact_restitution

        # Joint limit parameters (matching rewarped)
        ant.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )

        # Load the dflex ant.xml (same MJCF as rewarped, density=1000 baked in)
        asset_dir = os.path.join(os.path.dirname(__file__), "assets")
        ant.add_mjcf(
            os.path.join(asset_dir, "ant.xml"),
            enable_self_collisions=True,
            armature_scale=5,  # rewarped: armature_scale=5
        )

        # Set initial joint positions (matching rewarped create_articulation)
        ant.joint_q[7:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
        ant.joint_q[2] = 0.75  # start_height (Z-up in Newton)

        # Replicate across environments
        builder = newton.ModelBuilder()
        builder.replicate(ant, self.num_envs)
        builder.add_ground_plane()

        self.model = builder.finalize(
            device=self.device,
            requires_grad=self.requires_grad,
        )

        # Fix joint limit parameters: Newton's MJCF parser derives limit_ke=2500, limit_kd=100
        # from MuJoCo's default solref=(0.02, 1.0), but rewarped uses limit_ke=1000, limit_kd=10.
        # The 10x higher limit_kd is especially damaging — it cancels control torques near limits.
        joint_limit_ke = wp.to_torch(self.model.joint_limit_ke)
        joint_limit_kd = wp.to_torch(self.model.joint_limit_kd)
        for env_id in range(self.num_envs):
            offset = env_id * self.num_joint_qd
            joint_limit_ke[offset + 6 : offset + 14] = 1.0e3  # rewarped: limit_ke=1000
            joint_limit_kd[offset + 6 : offset + 14] = 1.0e1  # rewarped: limit_kd=10
        self.model.joint_limit_ke.assign(wp.from_torch(joint_limit_ke))
        self.model.joint_limit_kd.assign(wp.from_torch(joint_limit_kd))

        # Featherstone solver (matching rewarped: angular_damping=0.0, update_mass_matrix_every=16)
        self.solver = newton.solvers.SolverFeatherstone(
            self.model,
            angular_damping=0.0,
            update_mass_matrix_interval=self.sim_substeps,
        )

        # Allocate simulation objects
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Forward kinematics for initial state
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    def _init_buffers(self):
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_act), device=self.device)

        # Store default joint state for resets
        joint_q = wp.to_torch(self.state_0.joint_q).clone()
        joint_qd = wp.to_torch(self.state_0.joint_qd).clone()
        self.default_joint_q = joint_q.view(self.num_envs, -1)
        self.default_joint_qd = joint_qd.view(self.num_envs, -1)

        # Heading / up vectors for observation computation
        self.targets = torch.tensor([10000.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )

        # Newton converts MJCF to Z-up (gravity = (0, 0, -9.81))
        self.up_axis = 2  # Z-up
        self.forward_axis = 0  # X-forward

        self.up_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.up_vec[:, self.up_axis] = 1.0
        self.heading_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_vec[:, self.forward_axis] = 1.0

        # Start rotation (identity — ant starts upright in Z-up)
        self.start_rotation = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat(self.num_envs, 1)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        joint_q = wp.to_torch(self.state_0.joint_q).view(self.num_envs, -1)
        joint_qd = wp.to_torch(self.state_0.joint_qd).view(self.num_envs, -1)

        N = len(env_ids)

        # Reset to default + small random perturbation (matching rewarped randomize_init)
        joint_q[env_ids] = self.default_joint_q[env_ids].clone()
        joint_qd[env_ids] = self.default_joint_qd[env_ids].clone()

        # Randomize positions slightly
        joint_q[env_ids, 0:3] += 0.1 * (torch.rand(N, 3, device=self.device) - 0.5) * 2.0
        # Randomize joint angles slightly
        joint_q[env_ids, 7:] += (
            0.2 * (torch.rand(N, self.num_joint_q - 7, device=self.device) - 0.5) * 2.0
        )
        # Small random velocities
        joint_qd[env_ids] = 0.5 * (torch.rand(N, self.num_joint_qd, device=self.device) - 0.5)

        # Write back
        self.state_0.joint_q.assign(wp.from_torch(joint_q.flatten()))
        self.state_0.joint_qd.assign(wp.from_torch(joint_qd.flatten()))

        # Re-run forward kinematics (matching rewarped eval_fk=True)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.actions[env_ids] = 0.0

        self._compute_observations()
        return self.obs_buf

    def step(self, actions):
        actions = actions.view(self.num_envs, -1)
        actions = torch.clamp(actions, -1.0, 1.0)
        self.actions = actions

        # Apply actions via control.joint_f (Featherstone uses joint_f for direct forces)
        # joint_f has 14 DOFs per env: [6 free root, 8 revolute]
        # We write scaled_actions into indices 6:14 (the 8 revolute joints)
        scaled_actions = self.action_scale * actions
        scaled_actions = -scaled_actions  # invert to match rewarped/dflex joint convention
        ctrl_joint_f = wp.to_torch(self.control.joint_f)
        ctrl_joint_f.zero_()
        f_view = ctrl_joint_f.view(self.num_envs, -1)
        f_view[:, 6:14] = scaled_actions
        self.control.joint_f.assign(wp.from_torch(ctrl_joint_f))

        # Physics step — collide inside substep loop (matching rewarped sim_update)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.progress_buf += 1

        self._compute_observations()
        self._compute_reward()

        # NaN protection: auto-reset envs with NaN observations
        nan_envs = torch.isnan(self.obs_buf).any(dim=-1)
        if nan_envs.any():
            self.obs_buf[nan_envs] = 0.0
            self.rew_buf[nan_envs] = 0.0

        # Check termination (matching rewarped compute_reward)
        terminated = nan_envs | (self.obs_buf[:, 0] < self.termination_height)
        truncated = self.progress_buf >= self.episode_length
        done = terminated | truncated

        # Auto-reset terminated envs (capture done BEFORE reset clears it)
        env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        obs_before_reset = self.obs_buf.clone() if len(env_ids) > 0 else None
        if len(env_ids) > 0:
            self.reset(env_ids)

        info = {
            "terminated": terminated,
            "truncated": truncated,
            "obs_before_reset": obs_before_reset,
            "time_outs": truncated & ~terminated,
        }
        return self.obs_buf, self.rew_buf, done, info

    def _compute_observations(self):
        """Compute 37-dim observation."""
        joint_q = wp.to_torch(self.state_0.joint_q).view(self.num_envs, -1)
        joint_qd = wp.to_torch(self.state_0.joint_qd).view(self.num_envs, -1)

        torso_pos = joint_q[:, 0:3]
        torso_rot = joint_q[:, 3:7]
        # Newton Featherstone joint_qd: [lin_spatial(3), ang_vel(3), hinge_vel(8)]
        # Note: Newton stores spatial linear velocity FIRST, then angular (opposite of DFlex)
        lin_vel_spatial = joint_qd[:, 0:3]
        ang_vel = joint_qd[:, 3:6]

        # Convert spatial velocity to world-frame CoM velocity: v_world = v_s + ω × p
        lin_vel = lin_vel_spatial + torch.cross(ang_vel, torso_pos, dim=-1)

        # Target direction (project onto ground plane)
        to_target = self.targets - torso_pos
        to_target[:, self.up_axis] = 0.0
        target_dirs = normalize(to_target)

        # Rotate into torso frame
        torso_quat = quat_mul(torso_rot, self.inv_start_rot)
        up_vec = quat_rotate(torso_quat, self.up_vec)
        heading_vec = quat_rotate(torso_quat, self.heading_vec)

        self.obs_buf = torch.cat(
            [
                torso_pos[:, self.up_axis : self.up_axis + 1],  # 0: height
                torso_rot,  # 1:5: quaternion
                lin_vel,  # 5:8: linear velocity
                ang_vel,  # 8:11: angular velocity
                joint_q[:, 7:],  # 11:19: joint positions
                self.joint_vel_obs_scaling * joint_qd[:, 6:],  # 19:27: joint velocities
                up_vec[:, self.up_axis : self.up_axis + 1],  # 27: up component
                (heading_vec * target_dirs).sum(dim=-1, keepdim=True),  # 28: heading
                self.actions,  # 29:37: previous actions
            ],
            dim=-1,
        )
        # Sanitize: replace NaN/Inf to prevent corruption of running statistics
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=1e4, neginf=-1e4)
        self.obs_buf = torch.clamp(self.obs_buf, -1e4, 1e4)

    def _compute_reward(self):
        """Compute reward (matching rewarped/DiffRL exactly)."""
        up_reward = 0.1 * self.obs_buf[:, 27]
        heading_reward = self.obs_buf[:, 28]
        height_reward = self.obs_buf[:, 0] - self.termination_height
        progress_reward = self.obs_buf[:, 5]  # x-velocity

        self.rew_buf = (
            progress_reward + up_reward + heading_reward + height_reward
            + torch.sum(self.actions**2, dim=-1) * self.action_penalty
        )
