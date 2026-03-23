"""
Log-likelihood and score functions for the GLMM fitting engine.

Mirrors ``R/Fit_Funs.R`` from the original GLMMadaptive package.

The key function is :func:`loglik_mixed` which computes the marginal
log-likelihood using adaptive Gauss-Hermite quadrature::

    log p(y) ≈ Σ_i log[ Σ_k w_k^(i) p(y_i | b_k^(i)) p(b_k^(i) | D) ]

Numerical stability is ensured by performing all summations in log-space
(log-sum-exp), mirroring the use of ``matrixStats::rowLogSumExps`` in R.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.linalg import solve

from glmmadaptive.families.base import BaseFamily
from glmmadaptive.utils.linalg import log_dmvnorm, nearPD, cov_to_chol, chol_to_cov
from glmmadaptive.utils.numdiff import cd_grad


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pack_params(betas, D, phis, diagonal_D: bool) -> NDArray:
    """Flatten all parameters into a single 1-D vector for optimisers."""
    parts = [betas]
    parts.append(cov_to_chol(D, diagonal=diagonal_D))
    if phis is not None and len(phis) > 0:
        parts.append(phis)
    return np.concatenate(parts)


def _unpack_params(
    theta: NDArray,
    n_betas: int,
    n_re: int,
    n_phis: int,
    diagonal_D: bool,
) -> tuple[NDArray, NDArray, Optional[NDArray]]:
    """Unpack 1-D parameter vector into (betas, D, phis)."""
    idx = 0
    betas = theta[idx : idx + n_betas]
    idx += n_betas
    n_D = n_re if diagonal_D else n_re * (n_re + 1) // 2
    D = chol_to_cov(theta[idx : idx + n_D], q=n_re, diagonal=diagonal_D)
    idx += n_D
    phis = theta[idx : idx + n_phis] if n_phis > 0 else None
    return betas, D, phis


# ---------------------------------------------------------------------------
# Marginal log-likelihood
# ---------------------------------------------------------------------------

def loglik_mixed(
    betas: NDArray,
    D: NDArray,
    phis: Optional[NDArray],
    family: BaseFamily,
    X_list: list[NDArray],
    Z_list: list[NDArray],
    y_list: list[NDArray],
    gh: "GHQuadrature",  # noqa: F821
    *,
    sign: float = 1.0,
) -> float:
    """
    Marginal log-likelihood via adaptive Gauss-Hermite quadrature.

    Mirrors ``logLik_mixed()`` in ``R/Fit_Funs.R``.

    For group i with n_i observations and q random effects the contribution is::

        log p(y_i | θ) ≈ log Σ_k [ w_k^(i) * prod_j p(y_ij | η_ijk) * p(b_k^(i) | D) ]

    where b_k^(i) are the adaptive quadrature nodes centred at the posterior
    mode b̂_i (stored in *gh*).

    Parameters
    ----------
    betas : ndarray of shape (p,)
    D : ndarray of shape (q, q)
        Random-effects covariance matrix.
    phis : ndarray or None
        Extra dispersion parameters (log-scale).
    family : BaseFamily
    X_list : list of (n_i, p) arrays
        Fixed-effects design matrices (one per group).
    Z_list : list of (n_i, q) arrays
        Random-effects design matrices (one per group).
    y_list : list of (n_i,) arrays
        Responses (one per group).
    gh : GHQuadrature
        Pre-built adaptive quadrature object.
    sign : float
        Multiply result by *sign* (use -1.0 to get the negative log-likelihood
        for minimisers).

    Returns
    -------
    float
    """
    n_groups = len(X_list)
    _, log_det_D = np.linalg.slogdet(D)
    D_inv = np.linalg.inv(D)

    total_ll = 0.0

    for i in range(n_groups):
        X_i = X_list[i]
        Z_i = Z_list[i]
        y_i = y_list[i]
        b_nodes = gh.b_nodes[i]      # (n_pts, q)
        log_w = gh.log_weights[i]    # (n_pts,)

        n_pts = b_nodes.shape[0]
        log_integ = np.empty(n_pts)

        for k in range(n_pts):
            b_k = b_nodes[k]
            eta_k = X_i @ betas + Z_i @ b_k
            # log p(y_i | b_k)
            log_py_b = np.sum(family.log_dens(y_i, eta_k, phis=phis))
            # log p(b_k | D)
            log_pb = log_dmvnorm(b_k, cov_inv=D_inv, log_det_cov=log_det_D)
            log_integ[k] = log_py_b + log_pb + log_w[k]

        # log Σ_k exp(log_integ[k])
        total_ll += logsumexp(log_integ)

    return sign * total_ll


def loglik_mixed_vec(theta, n_betas, n_re, n_phis, diagonal_D,
                     family, X_list, Z_list, y_list, gh):
    """Wrapper for optimisers that take a single parameter vector."""
    betas, D, phis = _unpack_params(theta, n_betas, n_re, n_phis, diagonal_D)
    return loglik_mixed(
        betas, D, phis, family, X_list, Z_list, y_list, gh, sign=-1.0
    )


# ---------------------------------------------------------------------------
# Score w.r.t. betas
# ---------------------------------------------------------------------------

def score_betas(
    betas: NDArray,
    D: NDArray,
    phis: Optional[NDArray],
    family: BaseFamily,
    X_list: list[NDArray],
    Z_list: list[NDArray],
    y_list: list[NDArray],
    gh: "GHQuadrature",  # noqa: F821
) -> NDArray:
    """
    Score vector ∂ log L / ∂ betas.

    Mirrors ``score_betas()`` in ``R/Fit_Funs.R``.

    Uses analytic ``family.score_eta`` if available, otherwise falls back to
    central-difference numerical differentiation.
    """
    n_groups = len(X_list)
    p = len(betas)
    _, log_det_D = np.linalg.slogdet(D)
    D_inv = np.linalg.inv(D)

    score = np.zeros(p)

    for i in range(n_groups):
        X_i = X_list[i]
        Z_i = Z_list[i]
        y_i = y_list[i]
        b_nodes = gh.b_nodes[i]
        log_w = gh.log_weights[i]
        n_pts = b_nodes.shape[0]

        # Compute log unnormalised weights for each quadrature point
        log_unnorm = np.empty(n_pts)
        eta_pts = []
        for k in range(n_pts):
            b_k = b_nodes[k]
            eta_k = X_i @ betas + Z_i @ b_k
            eta_pts.append(eta_k)
            log_py_b = np.sum(family.log_dens(y_i, eta_k, phis=phis))
            log_pb = log_dmvnorm(b_k, cov_inv=D_inv, log_det_cov=log_det_D)
            log_unnorm[k] = log_py_b + log_pb + log_w[k]

        # Normalised posterior weights
        log_Z_i = logsumexp(log_unnorm)
        post_w = np.exp(log_unnorm - log_Z_i)  # (n_pts,)

        # Gradient contribution from group i
        for k in range(n_pts):
            eta_k = eta_pts[k]
            s_eta = family.score_eta(y_i, eta_k, phis=phis)
            if s_eta is None:
                # Numerical fallback
                def f_eta(b):
                    return np.sum(family.log_dens(y_i, X_i @ betas + Z_i @ gh.b_nodes[i][k], phis=phis))
                s_eta = cd_grad(
                    lambda bb: np.sum(family.log_dens(y_i, X_i @ bb + Z_i @ b_nodes[k], phis=phis)),
                    betas,
                )
                score += post_w[k] * s_eta
            else:
                # s_eta has shape (n_i,): ∂log p / ∂η_j, apply chain rule ∂η/∂β = X_i
                score += post_w[k] * (X_i.T @ s_eta)

    return score


# ---------------------------------------------------------------------------
# Score w.r.t. D (covariance matrix parameters)
# ---------------------------------------------------------------------------

def score_D(
    betas: NDArray,
    D: NDArray,
    phis: Optional[NDArray],
    family: BaseFamily,
    X_list: list[NDArray],
    Z_list: list[NDArray],
    y_list: list[NDArray],
    gh: "GHQuadrature",  # noqa: F821
    diagonal_D: bool = False,
) -> NDArray:
    """
    Score w.r.t. the unconstrained Cholesky parameters of D.

    Uses the closed-form EM update direction: for the M-step of D the
    expectation E[b_i b_i' | y_i] is needed, which is the posterior second
    moment at the current quadrature.
    """
    n_groups = len(X_list)
    q = D.shape[0]
    _, log_det_D = np.linalg.slogdet(D)
    D_inv = np.linalg.inv(D)

    # Accumulate expected second moments
    E_bb = np.zeros((q, q))

    for i in range(n_groups):
        X_i = X_list[i]
        Z_i = Z_list[i]
        y_i = y_list[i]
        b_nodes = gh.b_nodes[i]
        log_w = gh.log_weights[i]
        n_pts = b_nodes.shape[0]

        log_unnorm = np.empty(n_pts)
        for k in range(n_pts):
            b_k = b_nodes[k]
            eta_k = X_i @ betas + Z_i @ b_k
            log_py_b = np.sum(family.log_dens(y_i, eta_k, phis=phis))
            log_pb = log_dmvnorm(b_k, cov_inv=D_inv, log_det_cov=log_det_D)
            log_unnorm[k] = log_py_b + log_pb + log_w[k]

        post_w = np.exp(log_unnorm - logsumexp(log_unnorm))

        for k in range(n_pts):
            b_k = b_nodes[k]
            E_bb += post_w[k] * np.outer(b_k, b_k)

    # EM update: D_new = E_bb / n_groups
    return E_bb / n_groups


# ---------------------------------------------------------------------------
# Score w.r.t. phis (extra dispersion parameters)
# ---------------------------------------------------------------------------

def score_phis(
    betas: NDArray,
    D: NDArray,
    phis: NDArray,
    family: BaseFamily,
    X_list: list[NDArray],
    Z_list: list[NDArray],
    y_list: list[NDArray],
    gh: "GHQuadrature",  # noqa: F821
) -> NDArray:
    """
    Score ∂ log L / ∂ phis via analytic score or finite differences.

    Mirrors ``score_phis()`` in ``R/Fit_Funs.R``.
    """
    n_phis = len(phis)
    n_groups = len(X_list)
    _, log_det_D = np.linalg.slogdet(D)
    D_inv = np.linalg.inv(D)

    score = np.zeros(n_phis)

    for i in range(n_groups):
        X_i = X_list[i]
        Z_i = Z_list[i]
        y_i = y_list[i]
        b_nodes = gh.b_nodes[i]
        log_w = gh.log_weights[i]
        n_pts = b_nodes.shape[0]

        log_unnorm = np.empty(n_pts)
        eta_pts = []
        for k in range(n_pts):
            b_k = b_nodes[k]
            eta_k = X_i @ betas + Z_i @ b_k
            eta_pts.append(eta_k)
            log_py_b = np.sum(family.log_dens(y_i, eta_k, phis=phis))
            log_pb = log_dmvnorm(b_k, cov_inv=D_inv, log_det_cov=log_det_D)
            log_unnorm[k] = log_py_b + log_pb + log_w[k]

        post_w = np.exp(log_unnorm - logsumexp(log_unnorm))

        for k in range(n_pts):
            eta_k = eta_pts[k]
            s_p = family.score_phis(y_i, eta_k, phis=phis)
            if s_p is None:
                # Numerical fallback
                def f_phis(ph):
                    return np.sum(family.log_dens(y_i, eta_k, phis=ph))
                s_p = cd_grad(f_phis, phis)
            score += post_w[k] * s_p

    return score
