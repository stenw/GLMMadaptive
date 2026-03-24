"""
Adaptive Gauss-Hermite quadrature utilities.

Mirrors the R functions ``gauher()``, ``GHfun()``, and ``find_modes()`` from
``R/Functions.R`` of the original GLMMadaptive package.

The core idea is Pinheiro & Bates (1995): approximate
    integral p(y_i | b_i) p(b_i | theta) db_i
using quadrature nodes centred at the posterior mode b̂_i and scaled by the
posterior curvature, making the approximation highly accurate near convergence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.linalg import cholesky, LinAlgError
from typing import Callable, Tuple


# ---------------------------------------------------------------------------
# Gauss-Hermite nodes and weights  (port of R's gauher())
# ---------------------------------------------------------------------------

def gauher(n: int) -> Tuple[NDArray, NDArray]:
    """
    Compute nodes and weights for Gauss-Hermite quadrature of order *n*.

    Mirrors ``gauher()`` in ``R/Functions.R``.  The weight convention is
    chosen so that::

        integral_{-inf}^{inf} f(x) exp(-x^2) dx  ≈  sum_k w_k * f(x_k)

    For the standard GLMM adaptive quadrature we further adjust for the
    exp(x^2) absorbed in the nodes (see :func:`gh_adaptive`).

    Parameters
    ----------
    n : int
        Number of quadrature points (``nAGQ`` in the R code).

    Returns
    -------
    nodes : ndarray of shape (n,)
        Sorted quadrature nodes.
    weights : ndarray of shape (n,)
        Corresponding positive quadrature weights.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer")

    m = (n + 1) // 2
    nodes = np.zeros(n)
    weights = np.zeros(n)

    # Initial guess approximations from Abramowitz & Stegun (1972)
    for i in range(1, m + 1):
        if i == 1:
            z = np.sqrt(2.0 * n + 1) - 1.85575 * (2.0 * n + 1) ** (-1.0 / 6.0)
        elif i == 2:
            z = z - 1.14 * n ** 0.426 / z
        elif i == 3:
            z = 1.86 * z - 0.86 * nodes[0]
        elif i == 4:
            z = 1.91 * z - 0.91 * nodes[1]
        else:
            z = 2.0 * z - nodes[i - 3]

        # Newton-Raphson refinement
        for _ in range(10):
            p1 = 1.0 / np.pi ** 0.25
            p2 = 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = z * np.sqrt(2.0 / j) * p2 - np.sqrt((j - 1.0) / j) * p3
            # p1 is now the Hermite polynomial H_n(z)
            # Derivative: pp = sqrt(2n) * p2
            pp = np.sqrt(2.0 * n) * p2
            z_old = z
            z = z_old - p1 / pp
            if abs(z - z_old) <= 3e-14:
                break

        nodes[i - 1] = z
        nodes[n - i] = -z
        weights[i - 1] = 2.0 / (pp * pp)
        weights[n - i] = weights[i - 1]

    return np.sort(nodes), weights[np.argsort(nodes)]


# ---------------------------------------------------------------------------
# Find posterior mode per group  (port of R's find_modes())
# ---------------------------------------------------------------------------

def find_posterior_mode(
    log_post_fun: Callable[[NDArray], float],
    b_init: NDArray,
) -> Tuple[NDArray, NDArray]:
    """
    Find the mode of ``log_post_fun(b)`` and return (mode, -Hessian).

    Mirrors ``find_modes()`` in ``R/Functions.R``.

    Uses L-BFGS-B (gradient-free via finite differences) to maximise the
    log-posterior, then computes the Hessian numerically.

    Parameters
    ----------
    log_post_fun : callable
        Function that takes a 1-D array *b* and returns the log-posterior
        (scalar).  Should be defined as *negative* log-posterior since we
        minimise.
    b_init : ndarray
        Starting point for optimisation (shape ``(q,)``).

    Returns
    -------
    mode : ndarray of shape (q,)
        Posterior mode b̂.
    neg_hess : ndarray of shape (q, q)
        Negative Hessian (= information matrix) at the mode, projected to
        be positive definite via :func:`~glmmadaptive.utils.linalg.nearPD`.
    """
    from glmmadaptive.utils.numdiff import cd_hess
    from glmmadaptive.utils.linalg import nearPD

    q = len(b_init)
    result = minimize(
        log_post_fun,
        b_init,
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 200},
    )
    mode = result.x

    # Hessian of *negative* log-posterior (so it should be PD at mode)
    H = cd_hess(log_post_fun, mode)
    H = nearPD(0.5 * (H + H.T))
    return mode, H


# ---------------------------------------------------------------------------
# Adaptive GH quadrature setup  (port of R's GHfun())
# ---------------------------------------------------------------------------

class GHQuadrature:
    """
    Adaptive Gauss-Hermite quadrature for one fitting iteration.

    For each group i this class stores the transformed nodes B[i] (shape
    ``(n_gh^q, q)``), log-weights log_w[i] (shape ``(n_gh^q,)``), and
    cached linear predictors.

    Mirrors the ``GHfun()`` return value in ``R/Functions.R``.
    """

    def __init__(
        self,
        nodes_std: NDArray,
        weights_std: NDArray,
        modes: NDArray,
        neg_hessians: list[NDArray],
        n_re: int,
    ):
        self.nodes_std = nodes_std      # (nAGQ,)
        self.weights_std = weights_std  # (nAGQ,)
        self.modes = modes              # (n_groups, n_re)
        self.neg_hessians = neg_hessians  # list of (n_re, n_re)
        self.n_re = n_re
        self._build_grid()

    def _build_grid(self):
        """
        Pre-compute per-group nodes, log-weights, and log|L| corrections.

        Transformation (following Pinheiro & Bates 1995):
            b_k^(i) = sqrt(2) * L_i^{-T} x_k + b̂_i
        where L_i is the Cholesky factor of the neg-Hessian (information).

        Log-weight correction absorbs the Jacobian of the transformation and
        the absorbed exp(x^2) factor from the standard GH rule.
        """
        nAGQ = len(self.nodes_std)
        n_groups = len(self.modes)
        q = self.n_re

        # Build multi-dim grid by tensor product (for q > 1)
        # For q==1 the grid is just the 1-D nodes
        if q == 1:
            x_grid = self.nodes_std[:, None]  # (nAGQ, 1)
            log_w_base = np.log(self.weights_std) + self.nodes_std ** 2
        else:
            grids = np.meshgrid(*([self.nodes_std] * q), indexing="ij")
            x_grid = np.stack([g.ravel() for g in grids], axis=1)  # (nAGQ^q, q)
            # base log-weight: sum of log-weights + ||x||^2 (absorbed exp)
            lw_parts = np.meshgrid(
                *([np.log(self.weights_std) + self.nodes_std ** 2] * q),
                indexing="ij",
            )
            log_w_base = np.sum(
                np.stack([p.ravel() for p in lw_parts], axis=1), axis=1
            )

        n_pts = x_grid.shape[0]

        self.b_nodes: list[NDArray] = []        # per group, (n_pts, q)
        self.log_weights: list[NDArray] = []    # per group, (n_pts,)

        from scipy.linalg import solve_triangular

        for i in range(n_groups):
            H_i = self.neg_hessians[i]
            try:
                L_i = cholesky(H_i, lower=True)
                # Transform: b_k = sqrt(2) * L_i^{-T} x_k + b̂_i
                # Solve the upper-triangular system L_i^T z = x_k  →  z = L_i^{-T} x_k
                # x_grid.T has shape (q, n_pts); result has shape (q, n_pts)
                b_k = (
                    np.sqrt(2.0) * solve_triangular(L_i.T, x_grid.T, lower=False).T
                    + self.modes[i]
                )  # (n_pts, q)

                # Log-weight correction (Jacobian of the transformation):
                #   |db/dx| = (sqrt(2))^q * |det(L_i)|^{-1}
                # In log-space:
                #   (q/2)*log(2) - sum(log(diag(L_i)))
                log_det_correction = (
                    (q / 2.0) * np.log(2.0) - np.sum(np.log(np.diag(L_i)))
                )
            except (LinAlgError, np.linalg.LinAlgError):
                # Fallback: centre at mode, unit covariance
                b_k = np.sqrt(2.0) * x_grid + self.modes[i]
                log_det_correction = (q / 2.0) * np.log(2.0)

            self.b_nodes.append(b_k)
            self.log_weights.append(log_w_base + log_det_correction)


def gh_adaptive(
    n_agh: int,
    n_re: int,
    modes: NDArray,
    neg_hessians: list[NDArray],
) -> GHQuadrature:
    """
    Build an adaptive GH quadrature object.

    Parameters
    ----------
    n_agh : int
        Number of quadrature points per dimension.
    n_re : int
        Number of random effects per group.
    modes : ndarray of shape (n_groups, n_re)
        Posterior modes from the previous EM iteration.
    neg_hessians : list of ndarray of shape (n_re, n_re)
        Negative Hessian at each group's posterior mode.

    Returns
    -------
    GHQuadrature
    """
    nodes, weights = gauher(n_agh)
    return GHQuadrature(nodes, weights, modes, neg_hessians, n_re)
