"""Time integrator classes for 2D FEM simulation.

Provides:
- ``TimeIntegrator2D`` ABC defining the integrator interface.
- ``ImplicitEuler2D`` implementing backward Euler with Newton + filtered line search.

The Incremental Potential (IP) is::

    IP(x) = (1/2) sum_i m_i ||x_i - x_tilde_i||^2 + h^2 * sum_k E_k(x)

where the sum over k includes elastic, gravity, and (optionally) contact energies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.linalg as LA
from numpy import ndarray
from scipy.sparse import spmatrix
from scipy.sparse.linalg import spsolve

from .energies import Energy2D, InertiaEnergy2D


class TimeIntegrator2D(ABC):
    """Abstract base class for 2D time integrators."""

    @abstractmethod
    def step(
        self,
        x: ndarray,
        v: ndarray,
        dt: float,
        tol: float,
        max_iter: int = 100,
    ) -> tuple[ndarray, ndarray, dict]:
        """Advance one time step.

        Parameters
        ----------
        x : ndarray, shape (2*n_nodes,)
            Current positions (flattened).
        v : ndarray, shape (2*n_nodes,)
            Current velocities (flattened).
        dt : float
            Time step size.
        tol : float
            Newton convergence tolerance.
        max_iter : int
            Maximum Newton iterations.

        Returns
        -------
        x_new : ndarray
            Updated positions.
        v_new : ndarray
            Updated velocities.
        stats : dict
            Solver statistics (newton_iters, final_residual, backtrack_count,
            min_filter_alpha).
        """
        ...


class ImplicitEuler2D(TimeIntegrator2D):
    """Implicit Euler (backward Euler) integrator for 2D FEM.

    Each time step minimizes the Incremental Potential via Newton's method
    with a filtered line search to maintain barrier feasibility.

    Parameters
    ----------
    inertia : InertiaEnergy2D
        Inertia energy term (stores mass and mutable x_tilde).
    potentials : list[Energy2D]
        Potential energy terms (elastic, gravity, contact, etc.).
    """

    def __init__(self, inertia: InertiaEnergy2D, potentials: list[Energy2D]) -> None:
        self.inertia = inertia
        self.potentials = potentials

    # --- IP evaluation ---

    def ip_val(self, x: ndarray, h: float) -> float:
        """Incremental Potential value."""
        E = self.inertia.val(x)
        for pot in self.potentials:
            E += h * h * pot.val(x)
        return E

    def ip_grad(self, x: ndarray, h: float) -> ndarray:
        """Gradient of IP w.r.t. x."""
        g = self.inertia.grad(x)
        for pot in self.potentials:
            g = g + h * h * pot.grad(x)
        return g

    def ip_hess(self, x: ndarray, h: float) -> spmatrix:
        """Hessian of IP w.r.t. x (sparse)."""
        H = self.inertia.hess(x)
        for pot in self.potentials:
            H = H + h * h * pot.hess(x)
        return H

    # --- Filtered line search helpers ---

    def _init_step_size(self, x: ndarray, p: ndarray) -> float:
        """Compute maximum step size to maintain barrier feasibility.

        Calls ``init_step_size`` on any potential that implements it
        (e.g. BarrierEnergy2D, PointEdgeBarrierEnergy) and returns the minimum.
        """
        alpha = 1.0
        for pot in self.potentials:
            if hasattr(pot, "init_step_size"):
                alpha = min(alpha, pot.init_step_size(x, p))
        return alpha

    # --- Newton solver ---

    def _newton_solve(
        self,
        x_init: ndarray,
        h: float,
        tol: float,
        max_iter: int = 100,
    ) -> tuple[ndarray, dict]:
        """Solve IP minimization via Newton with filtered line search.

        Parameters
        ----------
        x_init : ndarray
            Initial guess (usually x_tilde, possibly projected).
        h : float
            Time step size.
        tol : float
            Convergence tolerance: ``||p||_inf / h < tol``.
        max_iter : int
            Maximum Newton iterations.

        Returns
        -------
        x_new : ndarray
            Converged solution.
        stats : dict
            newton_iters, final_residual, backtrack_count, min_filter_alpha.
        """
        x_new = x_init.copy()
        E_last = self.ip_val(x_new, h)
        newton_iters = 0
        total_backtracks = 0
        min_filter_alpha = 1.0

        for _it in range(max_iter):
            g = self.ip_grad(x_new, h)
            H = self.ip_hess(x_new, h)

            p = spsolve(H, -g)

            # If spsolve produced NaN (singular H), regularize and retry
            if not np.all(np.isfinite(p)):
                import scipy.sparse as sp
                reg = sp.diags(np.ones(H.shape[0]) * 1e-6 * H.diagonal().max(),
                               format="csr")
                p = spsolve(H + reg, -g)
                if not np.all(np.isfinite(p)):
                    break  # give up on this step

            residual = LA.norm(p, np.inf) / h

            if residual < tol:
                break

            # Filtered line search
            alpha = self._init_step_size(x_new, p)
            min_filter_alpha = min(min_filter_alpha, alpha)

            backtracks = 0
            E_trial = self.ip_val(x_new + alpha * p, h)
            while E_trial > E_last and backtracks < 64:
                alpha /= 2.0
                E_trial = self.ip_val(x_new + alpha * p, h)
                backtracks += 1
            total_backtracks += backtracks

            # Only accept step if energy decreased
            if E_trial <= E_last:
                x_new = x_new + alpha * p
                E_last = E_trial
            else:
                break  # line search failed, stop iterating

            newton_iters += 1

        stats = {
            "newton_iters": newton_iters,
            "final_residual": (
                LA.norm(self.ip_grad(x_new, h), np.inf) / h if newton_iters > 0 else 0.0
            ),
            "backtrack_count": total_backtracks,
            "min_filter_alpha": min_filter_alpha,
        }
        return x_new, stats

    # --- Public interface ---

    def step(
        self,
        x: ndarray,
        v: ndarray,
        dt: float,
        tol: float,
        max_iter: int = 100,
    ) -> tuple[ndarray, ndarray, dict]:
        """Advance one implicit Euler time step.

        Computes ``x_tilde = x + v * dt``, sets up the inertia term, runs
        the Newton solver, and updates the velocity.

        Parameters
        ----------
        x : ndarray, shape (2*n_nodes,)
            Current positions (flattened).
        v : ndarray, shape (2*n_nodes,)
            Current velocities (flattened).
        dt : float
            Time step size.
        tol : float
            Newton convergence tolerance.
        max_iter : int
            Maximum Newton iterations.

        Returns
        -------
        x_new, v_new, stats
        """
        x_tilde = x + v * dt
        self.inertia.x_tilde = x_tilde
        # Start Newton from current position x (which is feasible),
        # not from x_tilde (which may penetrate barriers).
        x_new, stats = self._newton_solve(x.copy(), dt, tol, max_iter)
        v_new = (x_new - x) / dt
        # Expose predicted and converged positions for LTE estimators
        stats["x_tilde"] = x_tilde
        stats["x_converged"] = x_new
        return x_new, v_new, stats
