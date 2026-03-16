"""2D FEM energy classes for triangulated meshes.

Provides:
- Barrier utility functions (shared by flat-ground and point-edge contact)
- Helper functions for lumped mass and contact area
- Energy2D ABC matching the 1D Energy interface
- Concrete energy classes: InertiaEnergy2D, GravityEnergy2D, NeoHookeanEnergy,
  BarrierEnergy2D, PointEdgeBarrierEnergy

All energy classes operate on a flattened DOF vector ``x_flat`` of shape ``(2*n_nodes,)``
where ``x_flat[2*i]``, ``x_flat[2*i+1]`` are the ``(x, y)`` coordinates of node ``i``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sparse
from numpy import ndarray
from scipy.sparse import spmatrix

from BarrierContact.mesh import Mesh

# ---------------------------------------------------------------------------
# Barrier utility functions
# ---------------------------------------------------------------------------


def barrier_val(d: float, dhat: float) -> float:
    """IPC log-barrier value: ``b(d) = -(d/dhat - 1)^2 * ln(d/dhat)`` for ``0 < d < dhat``."""
    if d >= dhat or d <= 0:
        return 0.0
    s = d / dhat
    return -((s - 1.0) ** 2) * np.log(s)


def barrier_grad(d: float, dhat: float) -> float:
    """First derivative ``db/dd``."""
    if d >= dhat or d <= 0:
        return 0.0
    s = d / dhat
    return -(2.0 * (s - 1.0) * np.log(s) + (s - 1.0) ** 2 / s) / dhat


def barrier_hess(d: float, dhat: float) -> float:
    """Second derivative ``d^2b/dd^2``."""
    if d >= dhat or d <= 0:
        return 0.0
    s = d / dhat
    # d2b/dd2 = g'(s)/dhat^2 where g(s) = -2(s-1)*ln(s) - (s-1)^2/s
    # g'(s) = -2*ln(s) - 2(s-1)/s - (s-1)(s+1)/s^2
    return (
        -2.0 * np.log(s) - 2.0 * (s - 1.0) / s - (s - 1.0) * (s + 1.0) / s**2
    ) / dhat**2


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def compute_lumped_mass(mesh: Mesh, rho: float) -> ndarray:
    """Compute lumped mass per node from mesh triangles.

    ``m_i = rho * (sum of areas of triangles incident to node i) / 3``.

    Parameters
    ----------
    mesh : Mesh
        The triangular mesh.
    rho : float
        Material density.

    Returns
    -------
    ndarray, shape (n_nodes,)
        Lumped mass per node.
    """
    m = np.zeros(mesh.n_nodes)
    for tri in mesh.triangles:
        i0, i1, i2 = tri
        e1 = mesh.vertices[i1] - mesh.vertices[i0]
        e2 = mesh.vertices[i2] - mesh.vertices[i0]
        area = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
        for node in tri:
            m[node] += rho * area / 3.0
    return m


def compute_contact_area(mesh: Mesh) -> ndarray:
    """Compute contact area (quadrature weight) per node from boundary edges.

    For boundary nodes: half the sum of incident boundary edge lengths.
    For interior nodes: 0.

    Parameters
    ----------
    mesh : Mesh
        The triangular mesh.

    Returns
    -------
    ndarray, shape (n_nodes,)
        Contact area per node.
    """
    ca = np.zeros(mesh.n_nodes)
    for edge in mesh.boundary_edges:
        i0, i1 = edge
        length = np.linalg.norm(mesh.vertices[i1] - mesh.vertices[i0])
        ca[i0] += length / 2.0
        ca[i1] += length / 2.0
    return ca


# ---------------------------------------------------------------------------
# Energy2D ABC
# ---------------------------------------------------------------------------


class Energy2D(ABC):
    """Abstract base class for 2D energy terms.

    All energies operate on a flattened DOF vector ``x_flat`` of shape ``(2*n_nodes,)``.
    """

    @abstractmethod
    def val(self, x_flat: ndarray) -> float:
        """Compute energy value."""
        ...

    @abstractmethod
    def grad(self, x_flat: ndarray) -> ndarray:
        """Compute energy gradient."""
        ...

    @abstractmethod
    def hess(self, x_flat: ndarray) -> spmatrix:
        """Compute energy Hessian (sparse)."""
        ...


# ---------------------------------------------------------------------------
# InertiaEnergy2D
# ---------------------------------------------------------------------------


class InertiaEnergy2D(Energy2D):
    """Inertia energy: ``E = (1/2) sum_i m_i ||x_i - x_tilde_i||^2``.

    The mutable ``x_tilde`` attribute must be set by the integrator before each step.

    Parameters
    ----------
    m : ndarray, shape (n_nodes,)
        Lumped mass per node.
    """

    def __init__(self, m: ndarray):
        self.m = m
        self.x_tilde: ndarray | None = None

    def val(self, x_flat: ndarray) -> float:
        diff = (x_flat - self.x_tilde).reshape(-1, 2)
        return 0.5 * np.sum(self.m[:, None] * diff**2)

    def grad(self, x_flat: ndarray) -> ndarray:
        diff = (x_flat - self.x_tilde).reshape(-1, 2)
        g = self.m[:, None] * diff
        return g.ravel()

    def hess(self, x_flat: ndarray) -> spmatrix:
        diag = np.repeat(self.m, 2)
        return sparse.diags(diag, format="csr")


# ---------------------------------------------------------------------------
# GravityEnergy2D
# ---------------------------------------------------------------------------


class GravityEnergy2D(Energy2D):
    """Gravitational potential energy: ``PE = -sum_i m_i * g_vec . x_i``.

    The gradient is constant and the Hessian is zero.

    Parameters
    ----------
    m : ndarray, shape (n_nodes,)
        Lumped mass per node.
    g_vec : ndarray, shape (2,)
        Gravitational acceleration vector, e.g. ``[0, -9.81]``.
    """

    def __init__(self, m: ndarray, g_vec: ndarray):
        self.m = m
        self.g_vec = np.asarray(g_vec)
        self.n_nodes = len(m)
        # Precompute constant gradient
        self._grad = np.zeros(2 * self.n_nodes)
        for i in range(self.n_nodes):
            self._grad[2 * i] = -m[i] * g_vec[0]
            self._grad[2 * i + 1] = -m[i] * g_vec[1]

    def val(self, x_flat: ndarray) -> float:
        x = x_flat.reshape(-1, 2)
        return np.sum(-self.m[:, None] * x * self.g_vec[None, :])

    def grad(self, x_flat: ndarray) -> ndarray:
        return self._grad.copy()

    def hess(self, x_flat: ndarray) -> spmatrix:
        n_dof = 2 * self.n_nodes
        return sparse.csr_matrix((n_dof, n_dof))


# ---------------------------------------------------------------------------
# NeoHookeanEnergy
# ---------------------------------------------------------------------------


class NeoHookeanEnergy(Energy2D):
    """2D Neo-Hookean hyperelastic energy on a triangular mesh.

    Energy density::

        Psi(F) = mu/2 * (I1 - 2 - 2*ln(J)) + lambda/2 * (ln(J))^2

    where ``I1 = tr(F^T F)``, ``J = det(F)``.

    Parameters
    ----------
    mesh : Mesh
        The triangular mesh (reference configuration).
    E_young : float
        Young's modulus.
    nu_poisson : float
        Poisson's ratio.
    ----------
    references:https://www.cs.cmu.edu/~15763-s26/lec/8-stress-and-derivatives.pdf
    """

    def __init__(self, mesh: Mesh, E_young: float, nu_poisson: float):
        self.mesh = mesh
        self.n_nodes = mesh.n_nodes
        self.n_tri = mesh.n_triangles

        # Lame parameters
        self.mu = E_young / (2.0 * (1.0 + nu_poisson))
        self.lam = (
            E_young
            * nu_poisson
            / ((1.0 + nu_poisson) * (1.0 - 2.0 * nu_poisson))
        )

        # Precompute reference inverse and rest area
        self.Dm_inv = np.zeros((self.n_tri, 2, 2))
        self.rest_area = np.zeros(self.n_tri)
        for e_idx in range(self.n_tri):
            i0, i1, i2 = mesh.triangles[e_idx]
            Dm = np.column_stack(
                [
                    mesh.vertices[i1] - mesh.vertices[i0],
                    mesh.vertices[i2] - mesh.vertices[i0],
                ]
            )
            self.rest_area[e_idx] = 0.5 * abs(np.linalg.det(Dm))
            self.Dm_inv[e_idx] = np.linalg.inv(Dm)

    def _deformation_gradient(self, x_flat: ndarray, e_idx: int) -> ndarray:
        """Compute ``F = Ds @ Dm_inv`` for triangle *e_idx*."""
        tri = self.mesh.triangles[e_idx]
        x = x_flat.reshape(-1, 2)
        Ds = np.column_stack(
            [
                x[tri[1]] - x[tri[0]],
                x[tri[2]] - x[tri[0]],
            ]
        )
        return Ds @ self.Dm_inv[e_idx]

    def _psi(self, F: ndarray) -> float:
        """Neo-Hookean energy density."""
        if not np.all(np.isfinite(F)):
            return 1e18
        I1 = np.sum(F * F)
        J = np.linalg.det(F)
        if J <= 0:
            return 1e18
        lnJ = np.log(J)
        return self.mu / 2.0 * (I1 - 2.0 - 2.0 * lnJ) + self.lam / 2.0 * lnJ**2

    def _dpsi_dF(self, F: ndarray) -> ndarray:
        """First Piola-Kirchhoff stress via SVD: ``P = U * diag(dPsi/dsigma) * VT``.

        Uses the SVD-based formulation for robustness near degenerate elements.
        """
        if not np.all(np.isfinite(F)):
            return np.zeros((2, 2))
        U, sigma, VT = self._polar_svd(F)
        sigma = np.maximum(sigma, 1e-8)
        mu, lam = self.mu, self.lam
        ln_sigma_prod = np.log(sigma[0] * sigma[1])
        dPsi_ds0 = (
            mu * (sigma[0] - 1.0 / sigma[0]) + lam / sigma[0] * ln_sigma_prod
        )
        dPsi_ds1 = (
            mu * (sigma[1] - 1.0 / sigma[1]) + lam / sigma[1] * ln_sigma_prod
        )
        return U @ np.diag([dPsi_ds0, dPsi_ds1]) @ VT

    @staticmethod
    def _polar_svd(F: ndarray) -> tuple[ndarray, ndarray, ndarray]:
        """Polar SVD with sign correction for proper rotation."""
        if not np.all(np.isfinite(F)):
            return np.eye(2), np.array([1.0, 1.0]), np.eye(2)
        try:
            U, s, VT = np.linalg.svd(F)
        except np.linalg.LinAlgError:
            return np.eye(2), np.array([1.0, 1.0]), np.eye(2)
        if np.linalg.det(U) < 0:
            U[:, 1] = -U[:, 1]
            s[1] = -s[1]
        if np.linalg.det(VT) < 0:
            VT[1, :] = -VT[1, :]
            s[1] = -s[1]
        return U, s, VT

    @staticmethod
    def _make_psd(M: ndarray) -> ndarray:
        """Project a symmetric matrix to positive semi-definite."""
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 0.0)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _d2psi_dF2(self, F: ndarray) -> ndarray:
        """SVD-based Hessian of Psi w.r.t. vectorized F (4x4 matrix in 2D).

        Uses the eigenanalysis approach from the phys-sim-book tutorial:
        decompose into diagonal (stretch) and off-diagonal (twist/flip)
        blocks in principal stretch space, project each to PSD, then
        rotate back.

        ``vec(F)`` ordering: ``[F00, F10, F01, F11]``.
        """
        U, sigma, VT = self._polar_svd(F)
        mu, lam = self.mu, self.lam

        # Clamp singular values away from zero
        sigma = np.maximum(sigma, 1e-8)

        # dPsi/dsigma
        ln_sigma_prod = np.log(sigma[0] * sigma[1])
        dPsi_ds0 = (
            mu * (sigma[0] - 1.0 / sigma[0]) + lam / sigma[0] * ln_sigma_prod
        )
        dPsi_ds1 = (
            mu * (sigma[1] - 1.0 / sigma[1]) + lam / sigma[1] * ln_sigma_prod
        )

        # d2Psi/dsigma2 (2x2 diagonal block)
        inv2_0 = 1.0 / (sigma[0] * sigma[0])
        inv2_1 = 1.0 / (sigma[1] * sigma[1])
        H00 = mu * (1.0 + inv2_0) - lam * inv2_0 * (ln_sigma_prod - 1.0)
        H11 = mu * (1.0 + inv2_1) - lam * inv2_1 * (ln_sigma_prod - 1.0)
        H01 = lam / (sigma[0] * sigma[1])
        Psi_ss = self._make_psd(np.array([[H00, H01], [H01, H11]]))

        # Twist/flip block (2x2 off-diagonal)
        sigma_prod = sigma[0] * sigma[1]
        B_left = (mu + (mu - lam * np.log(sigma_prod)) / sigma_prod) / 2.0
        B_right = (dPsi_ds0 + dPsi_ds1) / (2.0 * max(sigma[0] + sigma[1], 1e-6))
        B = self._make_psd(
            np.array(
                [
                    [B_left + B_right, B_left - B_right],
                    [B_left - B_right, B_left + B_right],
                ]
            )
        )

        # Assemble 4x4 M matrix in principal space
        M = np.zeros((4, 4))
        M[0, 0] = Psi_ss[0, 0]
        M[0, 3] = Psi_ss[0, 1]
        M[1, 1] = B[0, 0]
        M[1, 2] = B[0, 1]
        M[2, 1] = B[1, 0]
        M[2, 2] = B[1, 1]
        M[3, 0] = Psi_ss[1, 0]
        M[3, 3] = Psi_ss[1, 1]

        # Rotate back to F-space: dP/dF[ij,rs] = sum_{abcd} U[i,a]*VT[b,j]*M[ab,cd]*U[r,c]*VT[d,s]
        dP_dF = np.zeros((4, 4))
        for j in range(2):
            for i in range(2):
                ij = j * 2 + i
                for s in range(2):
                    for r in range(2):
                        rs = s * 2 + r
                        dP_dF[ij, rs] = (
                            M[0, 0] * U[i, 0] * VT[0, j] * U[r, 0] * VT[0, s]
                            + M[0, 3] * U[i, 0] * VT[0, j] * U[r, 1] * VT[1, s]
                            + M[1, 1] * U[i, 1] * VT[0, j] * U[r, 1] * VT[0, s]
                            + M[1, 2] * U[i, 1] * VT[0, j] * U[r, 0] * VT[1, s]
                            + M[2, 1] * U[i, 0] * VT[1, j] * U[r, 1] * VT[0, s]
                            + M[2, 2] * U[i, 0] * VT[1, j] * U[r, 0] * VT[1, s]
                            + M[3, 0] * U[i, 1] * VT[1, j] * U[r, 0] * VT[0, s]
                            + M[3, 3] * U[i, 1] * VT[1, j] * U[r, 1] * VT[1, s]
                        )
        return dP_dF

    @staticmethod
    def _smallest_positive_real_root_quad(
        a: float, b: float, c: float
    ) -> float:
        """Find smallest positive real root of ``a*t^2 + b*t + c = 0``.

        Returns -1 if no positive real root exists.
        """
        tol = 1e-6
        if abs(a) <= tol:
            if abs(b) <= tol:
                return -1.0
            t = -c / b
            return t if t > 0 else -1.0

        disc = b * b - 4.0 * a * c
        if disc < 0:
            return -1.0

        sqrt_disc = np.sqrt(disc)
        t = (-b - sqrt_disc) / (2.0 * a)
        if t < 0:
            t = (-b + sqrt_disc) / (2.0 * a)
        return t

    def init_step_size(self, x_flat: ndarray, p: ndarray) -> float:
        """Compute maximum step size to prevent element inversions.

        For each triangle, solves for the step size that brings volume to
        10% of current (``c = 0.9``), matching the phys-sim-book reference.
        """
        alpha = 1.0
        p2 = p.reshape(-1, 2)
        x2 = x_flat.reshape(-1, 2)

        for e_idx in range(self.n_tri):
            tri = self.mesh.triangles[e_idx]
            i0, i1, i2 = tri

            x21 = x2[i1] - x2[i0]
            x31 = x2[i2] - x2[i0]
            p21 = p2[i1] - p2[i0]
            p31 = p2[i2] - p2[i0]

            detT = x21[0] * x31[1] - x21[1] * x31[0]
            if abs(detT) < 1e-30:
                continue

            a_coeff = (p21[0] * p31[1] - p21[1] * p31[0]) / detT
            b_coeff = (
                x21[0] * p31[1]
                - x21[1] * p31[0]
                + p21[0] * x31[1]
                - p21[1] * x31[0]
            ) / detT
            # c = 0.9: solve for when volume drops to 10% of current
            c_coeff = 0.9

            critical = self._smallest_positive_real_root_quad(
                a_coeff, b_coeff, c_coeff
            )
            if critical > 0:
                alpha = min(alpha, critical)

        return alpha

    def val(self, x_flat: ndarray) -> float:
        total = 0.0
        for e_idx in range(self.n_tri):
            F = self._deformation_gradient(x_flat, e_idx)
            total += self.rest_area[e_idx] * self._psi(F)
        return total

    def grad(self, x_flat: ndarray) -> ndarray:
        g = np.zeros(2 * self.n_nodes)
        for e_idx in range(self.n_tri):
            tri = self.mesh.triangles[e_idx]
            F = self._deformation_gradient(x_flat, e_idx)
            P = self._dpsi_dF(F)

            Dm_inv_T = self.Dm_inv[e_idx].T
            A = self.rest_area[e_idx]
            H = A * P @ Dm_inv_T

            i0, i1, i2 = tri
            g[2 * i1 : 2 * i1 + 2] += H[:, 0]
            g[2 * i2 : 2 * i2 + 2] += H[:, 1]
            g[2 * i0 : 2 * i0 + 2] -= H[:, 0] + H[:, 1]

        return g

    def hess(self, x_flat: ndarray) -> spmatrix:
        II, JJ, VV = [], [], []

        for e_idx in range(self.n_tri):
            tri = self.mesh.triangles[e_idx]
            F = self._deformation_gradient(x_flat, e_idx)
            A = self.rest_area[e_idx]

            # 4x4 material Hessian in vec(F) space (already PSD-projected)
            dPdF = self._d2psi_dF2(F)

            # Build B matrix (4 x 6): maps node displacements to vec(F)
            Dinv = self.Dm_inv[e_idx]
            B = np.zeros((4, 6))
            for i in range(2):  # spatial dim
                for j in range(2):  # ref dim
                    f_idx = j * 2 + i  # flat(i,j)
                    B[f_idx, i] = -(Dinv[0, j] + Dinv[1, j])
                    B[f_idx, 2 + i] = Dinv[0, j]
                    B[f_idx, 4 + i] = Dinv[1, j]

            # Element Hessian (6x6)
            H_e = A * B.T @ dPdF @ B

            # Assemble into global
            nodes = [tri[0], tri[1], tri[2]]
            for nI in range(3):
                for nJ in range(3):
                    for r in range(2):
                        for c in range(2):
                            row = nodes[nI] * 2 + r
                            col = nodes[nJ] * 2 + c
                            II.append(row)
                            JJ.append(col)
                            VV.append(H_e[nI * 2 + r, nJ * 2 + c])

        n_dof = 2 * self.n_nodes
        H = sparse.coo_matrix(
            (np.array(VV), (np.array(II, dtype=int), np.array(JJ, dtype=int))),
            shape=(n_dof, n_dof),
        ).tocsr()
        return H


# ---------------------------------------------------------------------------
# BarrierEnergy2D (flat ground)
# ---------------------------------------------------------------------------


class BarrierEnergy2D(Energy2D):
    """IPC barrier energy for flat ground contact (y-plane).

    For each node ``i``: ``d_i = y_i - y_ground``.
    ``E = sum_i contact_area_i * kappa * b(d_i, dhat)``.

    Parameters
    ----------
    y_ground : float
        Ground plane y-coordinate.
    kappa : float
        Barrier stiffness.
    dhat : float
        Barrier activation distance.
    contact_area : ndarray, shape (n_nodes,)
        Contact quadrature weights per node.
    ----------
    This energy is not exactly the same as the one taught in course, but comes from https://arxiv.org/pdf/2307.15908
    """

    def __init__(
        self,
        y_ground: float,
        kappa: float,
        dhat: float,
        contact_area: ndarray,
    ):
        self.y_ground = y_ground
        self.kappa = kappa
        self.dhat = dhat
        self.contact_area = contact_area
        self.n_nodes = len(contact_area)

    def val(self, x_flat: ndarray) -> float:
        x = x_flat.reshape(-1, 2)
        total = 0.0
        for i in range(self.n_nodes):
            d = x[i, 1] - self.y_ground
            if 0 < d < self.dhat:
                total += (
                    self.contact_area[i]
                    * self.kappa
                    * barrier_val(d, self.dhat)
                )
        return total

    def grad(self, x_flat: ndarray) -> ndarray:
        g = np.zeros(2 * self.n_nodes)
        x = x_flat.reshape(-1, 2)
        for i in range(self.n_nodes):
            d = x[i, 1] - self.y_ground
            if 0 < d < self.dhat:
                g[2 * i + 1] += (
                    self.contact_area[i]
                    * self.kappa
                    * barrier_grad(d, self.dhat)
                )
        return g

    def hess(self, x_flat: ndarray) -> spmatrix:
        II, JJ, VV = [], [], []
        x = x_flat.reshape(-1, 2)
        for i in range(self.n_nodes):
            d = x[i, 1] - self.y_ground
            if 0 < d < self.dhat:
                h_val = (
                    self.contact_area[i]
                    * self.kappa
                    * barrier_hess(d, self.dhat)
                )
                h_val = max(h_val, 0.0)  # PSD projection
                idx = 2 * i + 1
                II.append(idx)
                JJ.append(idx)
                VV.append(h_val)
        n_dof = 2 * self.n_nodes
        if VV:
            return sparse.coo_matrix(
                (np.array(VV), (np.array(II), np.array(JJ))),
                shape=(n_dof, n_dof),
            ).tocsr()
        return sparse.csr_matrix((n_dof, n_dof))

    def init_step_size(self, x_flat: ndarray, p: ndarray) -> float:
        """Filtered line search step size for barrier feasibility."""
        alpha = 1.0
        x = x_flat.reshape(-1, 2)
        p2 = p.reshape(-1, 2)
        for i in range(self.n_nodes):
            if p2[i, 1] < 0:
                gap = x[i, 1] - self.y_ground
                if gap > 0:
                    a = 0.9 * gap / (-p2[i, 1])
                    alpha = min(alpha, a)
        return alpha


# ---------------------------------------------------------------------------
# Point-edge distance primitive
# ---------------------------------------------------------------------------


def point_edge_unsigned_distance(
    p: ndarray, e0: ndarray, e1: ndarray
) -> tuple[float, float, ndarray, int]:
    """Compute unsigned distance from point ``p`` to segment ``[e0, e1]``.

    Returns
    -------
    d : float
        Unsigned distance.
    t : float
        Clamped parameter on edge.
    closest : ndarray (2,)
        Closest point on the segment.
    case : int
        0 = vertex e0, 1 = interior, 2 = vertex e1.
    """
    edge = e1 - e0
    edge_len_sq = np.dot(edge, edge)
    if edge_len_sq < 1e-30:
        return float(np.linalg.norm(p - e0)), 0.0, e0.copy(), 0

    t_raw = np.dot(p - e0, edge) / edge_len_sq

    if t_raw <= 0:
        return float(np.linalg.norm(p - e0)), 0.0, e0.copy(), 0
    elif t_raw >= 1:
        return float(np.linalg.norm(p - e1)), 1.0, e1.copy(), 2
    else:
        closest = e0 + t_raw * edge
        return float(np.linalg.norm(p - closest)), t_raw, closest, 1


# ---------------------------------------------------------------------------
# PointEdgeBarrierEnergy
# ---------------------------------------------------------------------------


class PointEdgeBarrierEnergy(Energy2D):
    """IPC barrier energy: block boundary nodes vs fixed ground edges.

    Only derivatives w.r.t. block DOFs are computed (ground is fixed).

    Parameters
    ----------
    bottom_node_indices : ndarray, int
        Indices of block bottom boundary nodes.
    ground_edges : ndarray, shape (n_ge, 2, 2)
        Ground edge endpoints. ``ground_edges[j] = [[x0,y0], [x1,y1]]``.
    n_block_nodes : int
        Total number of block nodes.
    kappa : float
        Barrier stiffness.
    dhat : float
        Barrier activation distance.
    contact_area : ndarray
        Contact quadrature weights for bottom nodes.
    """

    def __init__(
        self,
        bottom_node_indices: ndarray,
        ground_edges: ndarray,
        n_block_nodes: int,
        kappa: float,
        dhat: float,
        contact_area: ndarray,
    ):
        self.bottom_nodes = np.asarray(bottom_node_indices, dtype=int)
        self.ground_edges = np.asarray(ground_edges, dtype=float)
        self.n_block_nodes = n_block_nodes
        self.kappa = kappa
        self.dhat = dhat
        self.contact_area = np.asarray(contact_area, dtype=float)
        self.n_bottom = len(self.bottom_nodes)
        self.n_ground_edges = len(self.ground_edges)
        self.y_ground = 0.0  # compatibility attribute

    def _active_pairs(self, x_flat: ndarray):
        """Yield ``(bi, gi, node_idx, d, closest, case)`` for ``d < dhat``."""
        x = x_flat.reshape(-1, 2)
        for bi, node_idx in enumerate(self.bottom_nodes):
            p = x[node_idx]
            for gi in range(self.n_ground_edges):
                e0 = self.ground_edges[gi, 0]
                e1 = self.ground_edges[gi, 1]
                d, _t, closest, case = point_edge_unsigned_distance(p, e0, e1)
                if 0 < d < self.dhat:
                    yield bi, gi, node_idx, d, closest, case

    def val(self, x_flat: ndarray) -> float:
        total = 0.0
        for bi, _gi, _ni, d, _cl, _ca in self._active_pairs(x_flat):
            total += (
                self.contact_area[bi] * self.kappa * barrier_val(d, self.dhat)
            )
        return total

    def grad(self, x_flat: ndarray) -> ndarray:
        g = np.zeros(2 * self.n_block_nodes)
        x = x_flat.reshape(-1, 2)

        for bi, _gi, node_idx, d, closest, _case in self._active_pairs(x_flat):
            p = x[node_idx]
            dd_dp = (p - closest) / d  # unit direction from closest to p

            w = self.contact_area[bi]
            db = barrier_grad(d, self.dhat)

            g[2 * node_idx] += w * self.kappa * db * dd_dp[0]
            g[2 * node_idx + 1] += w * self.kappa * db * dd_dp[1]

        return g

    def hess(self, x_flat: ndarray) -> spmatrix:
        II, JJ, VV = [], [], []
        x = x_flat.reshape(-1, 2)

        for bi, _gi, node_idx, d, closest, case in self._active_pairs(x_flat):
            p = x[node_idx]
            dd_dp = (p - closest) / d

            # d^2 d / dp^2
            d2d = (
                np.zeros((2, 2))
                if case == 1
                else (np.eye(2) - np.outer(dd_dp, dd_dp)) / d
            )

            w = self.contact_area[bi]
            db = barrier_grad(d, self.dhat)
            d2b = barrier_hess(d, self.dhat)

            H_local = w * self.kappa * (d2b * np.outer(dd_dp, dd_dp) + db * d2d)

            # PSD projection
            eigvals, eigvecs = np.linalg.eigh(H_local)
            eigvals = np.maximum(eigvals, 0.0)
            H_local = eigvecs @ np.diag(eigvals) @ eigvecs.T

            for r in range(2):
                for c in range(2):
                    II.append(2 * node_idx + r)
                    JJ.append(2 * node_idx + c)
                    VV.append(H_local[r, c])

        n_dof = 2 * self.n_block_nodes
        if VV:
            return sparse.coo_matrix(
                (
                    np.array(VV),
                    (np.array(II, dtype=int), np.array(JJ, dtype=int)),
                ),
                shape=(n_dof, n_dof),
            ).tocsr()
        return sparse.csr_matrix((n_dof, n_dof))

    def init_step_size(self, x_flat: ndarray, p: ndarray) -> float:
        """Filtered step size: keep block nodes from penetrating ground."""
        alpha = 1.0
        x = x_flat.reshape(-1, 2)
        p2 = p.reshape(-1, 2)

        for _bi, node_idx in enumerate(self.bottom_nodes):
            pos = x[node_idx]
            dp = p2[node_idx]

            for gi in range(self.n_ground_edges):
                e0 = self.ground_edges[gi, 0]
                e1 = self.ground_edges[gi, 1]
                d, _t, closest, _case = point_edge_unsigned_distance(
                    pos, e0, e1
                )
                if d <= 0:
                    continue
                dd_dp = (pos - closest) / d
                rate = np.dot(dd_dp, dp)
                if rate < 0:
                    alpha = min(alpha, 0.9 * d / (-rate))

        return max(alpha, 1e-10)


# ---------------------------------------------------------------------------
# SelfContactEnergy (point-edge barriers between deformable boundary nodes)
# ---------------------------------------------------------------------------


def _find_all_boundary_edges(triangles: ndarray) -> ndarray:
    """Find all boundary edges (edges appearing in only one triangle).

    Parameters
    ----------
    triangles : ndarray, shape (n_tri, 3)
        Triangle connectivity.

    Returns
    -------
    ndarray, shape (n_be, 2), dtype int
        Boundary edges.
    """
    edge_count: dict[tuple[int, int], list[int]] = {}
    for tri in triangles:
        for k in range(3):
            e = (int(tri[k]), int(tri[(k + 1) % 3]))
            canonical = (min(e), max(e))
            edge_count.setdefault(canonical, []).append(1)

    return np.array(
        [list(e) for e, counts in edge_count.items() if len(counts) == 1],
        dtype=int,
    )


class SelfContactEnergy(Energy2D):
    """IPC barrier energy for self/inter-body contact on deformable meshes.

    Iterates all ``(boundary_point, boundary_edge)`` pairs, skipping incident
    pairs (point is an endpoint of the edge). Both the point and edge positions
    come from the DOF vector, so derivatives w.r.t. all involved DOFs are computed.

    Parameters
    ----------
    boundary_points : ndarray, shape (n_bp,), dtype int
        Node indices of boundary points.
    boundary_edges : ndarray, shape (n_be, 2), dtype int
        Node index pairs for boundary edges.
    n_total_nodes : int
        Total number of nodes in the combined mesh.
    kappa : float
        Barrier stiffness.
    dhat : float
        Barrier activation distance.
    contact_area : ndarray, shape (n_bp,)
        Contact quadrature weights for boundary points.
    """

    def __init__(
        self,
        boundary_points: ndarray,
        boundary_edges: ndarray,
        n_total_nodes: int,
        kappa: float,
        dhat: float,
        contact_area: ndarray,
    ):
        self.boundary_points = np.asarray(boundary_points, dtype=int)
        self.boundary_edges = np.asarray(boundary_edges, dtype=int)
        self.n_total_nodes = n_total_nodes
        self.kappa = kappa
        self.dhat = dhat
        self.contact_area = np.asarray(contact_area, dtype=float)
        self.n_bp = len(self.boundary_points)
        self.n_be = len(self.boundary_edges)

        # Build set of edges for fast incident check
        self._edge_set: set[tuple[int, int]] = set()
        for edge in self.boundary_edges:
            self._edge_set.add((int(edge[0]), int(edge[1])))
            self._edge_set.add((int(edge[1]), int(edge[0])))

        # For init_step_size compatibility
        self.y_ground = 0.0

    def _is_incident(self, point_idx: int, e0_idx: int, e1_idx: int) -> bool:
        """Check if point is an endpoint of the edge."""
        return point_idx in (e0_idx, e1_idx)

    def _active_pairs(self, x_flat: ndarray):
        """Yield active (point, edge) pairs with distance < dhat.

        Yields
        ------
        bi : int
            Index into boundary_points.
        ei : int
            Index into boundary_edges.
        point_idx, e0_idx, e1_idx : int
            Node indices.
        d : float
            Unsigned distance.
        t_param : float
            Edge parameter.
        closest : ndarray
            Closest point on edge.
        case : int
            Distance type.
        """
        x = x_flat.reshape(-1, 2)
        for bi, point_idx in enumerate(self.boundary_points):
            p = x[point_idx]
            for ei in range(self.n_be):
                e0_idx = self.boundary_edges[ei, 0]
                e1_idx = self.boundary_edges[ei, 1]

                if self._is_incident(point_idx, e0_idx, e1_idx):
                    continue

                e0 = x[e0_idx]
                e1 = x[e1_idx]
                d, t_param, closest, case = point_edge_unsigned_distance(
                    p, e0, e1
                )
                if 0 < d < self.dhat:
                    yield (
                        bi,
                        ei,
                        point_idx,
                        e0_idx,
                        e1_idx,
                        d,
                        t_param,
                        closest,
                        case,
                    )

    def val(self, x_flat: ndarray) -> float:
        total = 0.0
        for bi, _ei, _pi, _e0i, _e1i, d, _t, _cl, _ca in self._active_pairs(
            x_flat
        ):
            total += (
                self.contact_area[bi] * self.kappa * barrier_val(d, self.dhat)
            )
        return total

    def grad(self, x_flat: ndarray) -> ndarray:
        g = np.zeros(2 * self.n_total_nodes)
        x = x_flat.reshape(-1, 2)

        for (
            bi,
            _ei,
            pi,
            e0i,
            e1i,
            d,
            t_param,
            closest,
            case,
        ) in self._active_pairs(x_flat):
            p = x[pi]
            n_dir = (p - closest) / d  # unit normal from edge to point

            w = self.contact_area[bi]
            db = barrier_grad(d, self.dhat)
            scalar = w * self.kappa * db

            # dd/dp = n_dir (derivative of distance w.r.t. point position)
            g[2 * pi] += scalar * n_dir[0]
            g[2 * pi + 1] += scalar * n_dir[1]

            # dd/de0, dd/de1 depend on the case
            if case == 0:
                # Closest to e0: d = ||p - e0||, dd/de0 = -n_dir
                g[2 * e0i] += scalar * (-n_dir[0])
                g[2 * e0i + 1] += scalar * (-n_dir[1])
            elif case == 2:
                # Closest to e1: d = ||p - e1||, dd/de1 = -n_dir
                g[2 * e1i] += scalar * (-n_dir[0])
                g[2 * e1i + 1] += scalar * (-n_dir[1])
            else:
                # Interior: closest = e0 + t*(e1-e0), dd/de0 = -(1-t)*n_dir
                g[2 * e0i] += scalar * (-(1 - t_param) * n_dir[0])
                g[2 * e0i + 1] += scalar * (-(1 - t_param) * n_dir[1])
                g[2 * e1i] += scalar * (-t_param * n_dir[0])
                g[2 * e1i + 1] += scalar * (-t_param * n_dir[1])

        return g

    def hess(self, x_flat: ndarray) -> spmatrix:
        """Approximate Hessian using only the point DOFs (diagonal blocks).

        For robustness, we use a simplified Hessian that only populates
        the 2x2 block for the point node. This is a common approximation
        that still gives Newton convergence (albeit slower) while avoiding
        the complexity of full 6x6 point-edge Hessian blocks.
        """
        II, JJ, VV = [], [], []
        x = x_flat.reshape(-1, 2)

        for bi, _ei, pi, _e0i, _e1i, d, _t, closest, case in self._active_pairs(
            x_flat
        ):
            p = x[pi]
            dd_dp = (p - closest) / d

            # d^2d/dp^2: zero for interior case, (I - nn^T)/d for vertex case
            d2d = (
                np.zeros((2, 2))
                if case == 1
                else (np.eye(2) - np.outer(dd_dp, dd_dp)) / d
            )

            w = self.contact_area[bi]
            db = barrier_grad(d, self.dhat)
            d2b = barrier_hess(d, self.dhat)

            H_local = w * self.kappa * (d2b * np.outer(dd_dp, dd_dp) + db * d2d)

            # PSD projection
            eigvals, eigvecs = np.linalg.eigh(H_local)
            eigvals = np.maximum(eigvals, 0.0)
            H_local = eigvecs @ np.diag(eigvals) @ eigvecs.T

            for r in range(2):
                for c in range(2):
                    II.append(2 * pi + r)
                    JJ.append(2 * pi + c)
                    VV.append(H_local[r, c])

        n_dof = 2 * self.n_total_nodes
        if VV:
            return sparse.coo_matrix(
                (
                    np.array(VV),
                    (np.array(II, dtype=int), np.array(JJ, dtype=int)),
                ),
                shape=(n_dof, n_dof),
            ).tocsr()
        return sparse.csr_matrix((n_dof, n_dof))

    def init_step_size(self, x_flat: ndarray, p: ndarray) -> float:
        """Filtered step size: prevent boundary points from crossing edges."""
        alpha = 1.0
        x = x_flat.reshape(-1, 2)
        p2 = p.reshape(-1, 2)

        for (
            _bi,
            _ei,
            pi,
            e0i,
            e1i,
            d,
            _t,
            _closest,
            _case,
        ) in self._active_pairs(x_flat):
            if d <= 0:
                continue
            pos_p = x[pi]
            closest_on_edge = _closest if _closest is not None else x[e0i]

            dd_dp = (pos_p - closest_on_edge) / d
            # Relative velocity along normal (point moves, edge moves)
            dp_p = p2[pi]
            dp_e0 = p2[e0i]
            dp_e1 = p2[e1i]

            if _case == 0:
                dp_edge = dp_e0
            elif _case == 2:
                dp_edge = dp_e1
            else:
                dp_edge = (1 - _t) * dp_e0 + _t * dp_e1

            rate = np.dot(dd_dp, dp_p - dp_edge)
            if rate < 0:
                alpha = min(alpha, 0.9 * d / (-rate))

        return max(alpha, 1e-10)

