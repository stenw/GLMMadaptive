"""
Log-likelihood and score functions for the GLMM fitting engine.

Mirrors ``R/Fit_Funs.R`` from the original GLMMadaptive package.

The key function is :func:`loglik_mixed` which computes the marginal
log-likelihood using adaptive Gauss-Hermite quadrature::

    log p(y) ≈ Σ_i log[ Σ_k w_k^(i) p(y_i | b_k^(i)) p(b_k^(i) | D) ]

Numerical stability is ensured by performing all summations in log-space
(log-sum-exp), mirroring the use of ``matrixStats::rowLogSumExps`` in R.

Zero-inflated models
--------------------
When ``X_zi_list``, ``Z_zi_list`` and ``gammas`` are provided the
fitting engine computes the combined random-effects vector
``b = [b_count, b_zi]`` of dimension ``q_count + q_zi``.  The linear
predictors are split accordingly inside each function.
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
# Parameter packing / unpacking
# ---------------------------------------------------------------------------

def _pack_params(
    betas: NDArray,
    D: NDArray,
    phis: Optional[NDArray],
    diagonal_D: bool,
    gammas: Optional[NDArray] = None,
) -> NDArray:
    """Flatten all parameters into a single 1-D vector for optimisers."""
    parts = [betas, cov_to_chol(D, diagonal=diagonal_D)]
    if phis is not None and len(phis) > 0:
        parts.append(phis)
    if gammas is not None and len(gammas) > 0:
        parts.append(gammas)
    return np.concatenate(parts)


def _unpack_params(
    theta: NDArray,
    n_betas: int,
    n_re: int,
    n_phis: int,
    diagonal_D: bool,
    n_gammas: int = 0,
) -> tuple[NDArray, NDArray, Optional[NDArray], Optional[NDArray]]:
    """Unpack 1-D parameter vector into (betas, D, phis, gammas)."""
    idx = 0
    betas = theta[idx : idx + n_betas]
    idx += n_betas
    n_D = n_re if diagonal_D else n_re * (n_re + 1) // 2
    D = chol_to_cov(theta[idx : idx + n_D], q=n_re, diagonal=diagonal_D)
    idx += n_D
    phis = theta[idx : idx + n_phis] if n_phis > 0 else None
    idx += n_phis
    gammas = theta[idx : idx + n_gammas] if n_gammas > 0 else None
    return betas, D, phis, gammas


# ---------------------------------------------------------------------------
# Internal: build eta_zi for one group
# ---------------------------------------------------------------------------

def _make_eta_zi(
    b_k: NDArray,
    ncz: int,
    X_zi_i: Optional[NDArray],
    Z_zi_i: Optional[NDArray],
    gammas: Optional[NDArray],
) -> Optional[NDArray]:
    """Compute η_zi for one quadrature node b_k."""
    if X_zi_i is None or gammas is None:
        return None
    eta_zi = X_zi_i @ gammas
    if Z_zi_i is not None and Z_zi_i.shape[1] > 0:
        eta_zi = eta_zi + Z_zi_i @ b_k[ncz:]
    return eta_zi


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
    X_zi_list: Optional[list[NDArray]] = None,
    Z_zi_list: Optional[list[Optional[NDArray]]] = None,
    gammas: Optional[NDArray] = None,
    sign: float = 1.0,
) -> float:
    """
    Marginal log-likelihood via adaptive Gauss-Hermite quadrature.

    Mirrors ``logLik_mixed()`` in ``R/Fit_Funs.R``.

    Parameters
    ----------
    betas, D, phis : parameters for the count part.
    X_zi_list, Z_zi_list, gammas : ZI parameters (optional).
    sign : multiply result by sign (-1.0 for minimisers).
    """
    n_groups = len(X_list)
    _, log_det_D = np.linalg.slogdet(D)
    D_inv = np.linalg.inv(D)

    total_ll = 0.0

    for i in range(n_groups):
        X_i = X_list[i]
        Z_i = Z_list[i]
        y_i = y_list[i]
        ncz = Z_i.shape[1]
        X_zi_i = X_zi_list[i] if X_zi_list is not None else None
        Z_zi_i = Z_zi_list[i] if Z_zi_list is not None else None

        b_nodes = gh.b_nodes[i]      # (n_pts, q)
        log_w = gh.log_weights[i]    # (n_pts,)
        n_pts = b_nodes.shape[0]

        log_integ = np.empty(n_pts)
        for k in range(n_pts):
            b_k = b_nodes[k]
            eta_k = X_i @ betas + Z_i @ b_k[:ncz]
            eta_zi_k = _make_eta_zi(b_k, ncz, X_zi_i, Z_zi_i, gammas)
            log_py_b = np.sum(family.log_dens(y_i, eta_k, phis=phis, eta_zi=eta_zi_k))
            log_pb = log_dmvnorm(b_k, cov_inv=D_inv, log_det_cov=log_det_D)
            log_integ[k] = log_py_b + log_pb + log_w[k]

        total_ll += logsumexp(log_integ)

    return sign * total_ll


def loglik_mixed_vec(
    theta: NDArray,
    n_betas: int,
    n_re: int,
    n_phis: int,
    diagonal_D: bool,
    family: BaseFamily,
    X_list: list[NDArray],
    Z_list: list[NDArray],
    y_list: list[NDArray],
    gh: "GHQuadrature",  # noqa: F821
    n_gammas: int = 0,
    X_zi_list: Optional[list[NDArray]] = None,
    Z_zi_list: Optional[list[Optional[NDArray]]] = None,
) -> float:
    """Wrapper for optimisers that take a single parameter vector."""
    betas, D, phis, gammas = _unpack_params(
        theta, n_betas, n_re, n_phis, diagonal_D, n_gammas
    )
    return loglik_mixed(
        betas, D, phis, family, X_list, Z_list, y_list, gh,
        X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=gammas,
        sign=-1.0,
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
    X_zi_list: Optional[list[NDArray]] = None,
    Z_zi_list: Optional[list[Optional[NDArray]]] = None,
    gammas: Optional[NDArray] = None,
) -> NDArray:
    """Score vector ∂ log L / ∂ betas."""
    n_groups = len(X_list)
    p = len(betas)
    _, log_det_D = np.linalg.slogdet(D)
    D_inv = np.linalg.inv(D)

    score = np.zeros(p)

    for i in range(n_groups):
        X_i = X_list[i]
        Z_i = Z_list[i]
        y_i = y_list[i]
        ncz = Z_i.shape[1]
        X_zi_i = X_zi_list[i] if X_zi_list is not None else None
        Z_zi_i = Z_zi_list[i] if Z_zi_list is not None else None

        b_nodes = gh.b_nodes[i]
        log_w = gh.log_weights[i]
        n_pts = b_nodes.shape[0]

        log_unnorm = np.empty(n_pts)
        eta_pts = []
        eta_zi_pts = []

        for k in range(n_pts):
            b_k = b_nodes[k]
            eta_k = X_i @ betas + Z_i @ b_k[:ncz]
            eta_zi_k = _make_eta_zi(b_k, ncz, X_zi_i, Z_zi_i, gammas)
            eta_pts.append(eta_k)
            eta_zi_pts.append(eta_zi_k)
            log_py_b = np.sum(family.log_dens(y_i, eta_k, phis=phis, eta_zi=eta_zi_k))
            log_pb = log_dmvnorm(b_k, cov_inv=D_inv, log_det_cov=log_det_D)
            log_unnorm[k] = log_py_b + log_pb + log_w[k]

        post_w = np.exp(log_unnorm - logsumexp(log_unnorm))

        for k in range(n_pts):
            eta_k = eta_pts[k]
            eta_zi_k = eta_zi_pts[k]
            s_eta = family.score_eta(y_i, eta_k, phis=phis, eta_zi=eta_zi_k)
            if s_eta is None:
                _betas = betas  # captured
                _b_k = b_nodes[k]
                s_eta = cd_grad(
                    lambda bb: np.sum(
                        family.log_dens(y_i, X_i @ bb + Z_i @ _b_k[:ncz],
                                        phis=phis, eta_zi=eta_zi_k)
                    ),
                    betas,
                )
                score += post_w[k] * s_eta
            else:
                score += post_w[k] * (X_i.T @ s_eta)

    return score


# ---------------------------------------------------------------------------
# Score w.r.t. D
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
    X_zi_list: Optional[list[NDArray]] = None,
    Z_zi_list: Optional[list[Optional[NDArray]]] = None,
    gammas: Optional[NDArray] = None,
) -> NDArray:
    """EM update direction for D (expected second moment of random effects)."""
    n_groups = len(X_list)
    q = D.shape[0]
    _, log_det_D = np.linalg.slogdet(D)
    D_inv = np.linalg.inv(D)

    E_bb = np.zeros((q, q))

    for i in range(n_groups):
        X_i = X_list[i]
        Z_i = Z_list[i]
        y_i = y_list[i]
        ncz = Z_i.shape[1]
        X_zi_i = X_zi_list[i] if X_zi_list is not None else None
        Z_zi_i = Z_zi_list[i] if Z_zi_list is not None else None

        b_nodes = gh.b_nodes[i]
        log_w = gh.log_weights[i]
        n_pts = b_nodes.shape[0]

        log_unnorm = np.empty(n_pts)
        for k in range(n_pts):
            b_k = b_nodes[k]
            eta_k = X_i @ betas + Z_i @ b_k[:ncz]
            eta_zi_k = _make_eta_zi(b_k, ncz, X_zi_i, Z_zi_i, gammas)
            log_py_b = np.sum(family.log_dens(y_i, eta_k, phis=phis, eta_zi=eta_zi_k))
            log_pb = log_dmvnorm(b_k, cov_inv=D_inv, log_det_cov=log_det_D)
            log_unnorm[k] = log_py_b + log_pb + log_w[k]

        post_w = np.exp(log_unnorm - logsumexp(log_unnorm))
        for k in range(n_pts):
            b_k = b_nodes[k]
            E_bb += post_w[k] * np.outer(b_k, b_k)

    return E_bb / n_groups


# ---------------------------------------------------------------------------
# Score w.r.t. phis
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
    X_zi_list: Optional[list[NDArray]] = None,
    Z_zi_list: Optional[list[Optional[NDArray]]] = None,
    gammas: Optional[NDArray] = None,
) -> NDArray:
    """Score ∂ log L / ∂ phis via analytic score or finite differences."""
    n_phis = len(phis)
    n_groups = len(X_list)
    _, log_det_D = np.linalg.slogdet(D)
    D_inv = np.linalg.inv(D)

    score = np.zeros(n_phis)

    for i in range(n_groups):
        X_i = X_list[i]
        Z_i = Z_list[i]
        y_i = y_list[i]
        ncz = Z_i.shape[1]
        X_zi_i = X_zi_list[i] if X_zi_list is not None else None
        Z_zi_i = Z_zi_list[i] if Z_zi_list is not None else None

        b_nodes = gh.b_nodes[i]
        log_w = gh.log_weights[i]
        n_pts = b_nodes.shape[0]

        log_unnorm = np.empty(n_pts)
        eta_pts = []
        eta_zi_pts = []

        for k in range(n_pts):
            b_k = b_nodes[k]
            eta_k = X_i @ betas + Z_i @ b_k[:ncz]
            eta_zi_k = _make_eta_zi(b_k, ncz, X_zi_i, Z_zi_i, gammas)
            eta_pts.append(eta_k)
            eta_zi_pts.append(eta_zi_k)
            log_py_b = np.sum(family.log_dens(y_i, eta_k, phis=phis, eta_zi=eta_zi_k))
            log_pb = log_dmvnorm(b_k, cov_inv=D_inv, log_det_cov=log_det_D)
            log_unnorm[k] = log_py_b + log_pb + log_w[k]

        post_w = np.exp(log_unnorm - logsumexp(log_unnorm))

        for k in range(n_pts):
            eta_k = eta_pts[k]
            eta_zi_k = eta_zi_pts[k]
            s_p = family.score_phis(y_i, eta_k, phis=phis, eta_zi=eta_zi_k)
            if s_p is None:
                _eta_k = eta_k
                _eta_zi_k = eta_zi_k
                s_p = cd_grad(
                    lambda ph: np.sum(
                        family.log_dens(y_i, _eta_k, phis=ph, eta_zi=_eta_zi_k)
                    ),
                    phis,
                )
            score += post_w[k] * s_p

    return score


# ---------------------------------------------------------------------------
# Score w.r.t. gammas  (ZI fixed effects)
# ---------------------------------------------------------------------------

def score_gammas(
    betas: NDArray,
    D: NDArray,
    phis: Optional[NDArray],
    gammas: NDArray,
    family: BaseFamily,
    X_list: list[NDArray],
    Z_list: list[NDArray],
    y_list: list[NDArray],
    gh: "GHQuadrature",  # noqa: F821
    X_zi_list: list[NDArray],
    Z_zi_list: Optional[list[Optional[NDArray]]] = None,
) -> NDArray:
    """
    Score ∂ log L / ∂ gammas (ZI fixed-effects coefficients).

    Uses analytic ``family.score_eta_zi`` if available, otherwise falls back
    to central-difference numerical differentiation.
    """
    n_groups = len(X_list)
    n_gammas = len(gammas)
    _, log_det_D = np.linalg.slogdet(D)
    D_inv = np.linalg.inv(D)

    score = np.zeros(n_gammas)

    for i in range(n_groups):
        X_i = X_list[i]
        Z_i = Z_list[i]
        y_i = y_list[i]
        ncz = Z_i.shape[1]
        X_zi_i = X_zi_list[i]
        Z_zi_i = Z_zi_list[i] if Z_zi_list is not None else None

        b_nodes = gh.b_nodes[i]
        log_w = gh.log_weights[i]
        n_pts = b_nodes.shape[0]

        log_unnorm = np.empty(n_pts)
        eta_pts = []
        eta_zi_pts = []

        for k in range(n_pts):
            b_k = b_nodes[k]
            eta_k = X_i @ betas + Z_i @ b_k[:ncz]
            eta_zi_k = _make_eta_zi(b_k, ncz, X_zi_i, Z_zi_i, gammas)
            eta_pts.append(eta_k)
            eta_zi_pts.append(eta_zi_k)
            log_py_b = np.sum(family.log_dens(y_i, eta_k, phis=phis, eta_zi=eta_zi_k))
            log_pb = log_dmvnorm(b_k, cov_inv=D_inv, log_det_cov=log_det_D)
            log_unnorm[k] = log_py_b + log_pb + log_w[k]

        post_w = np.exp(log_unnorm - logsumexp(log_unnorm))

        for k in range(n_pts):
            eta_k = eta_pts[k]
            eta_zi_k = eta_zi_pts[k]
            s_zi = family.score_eta_zi(y_i, eta_k, phis=phis, eta_zi=eta_zi_k)
            if s_zi is None:
                _eta_k = eta_k
                _b_k = b_nodes[k]
                def _f_gammas(g, _b=_b_k, _e=_eta_k):
                    _eta_zi = _make_eta_zi(_b, ncz, X_zi_i, Z_zi_i, g)
                    return np.sum(family.log_dens(y_i, _e, phis=phis, eta_zi=_eta_zi))
                s_zi_grad = cd_grad(_f_gammas, gammas)
                score += post_w[k] * s_zi_grad
            else:
                score += post_w[k] * (X_zi_i.T @ s_zi)

    return score


# ---------------------------------------------------------------------------
# Per-group score contributions  (used for sandwich estimator)
# ---------------------------------------------------------------------------

def score_contributions(
    betas: NDArray,
    D: NDArray,
    phis: Optional[NDArray],
    gammas: Optional[NDArray],
    family: BaseFamily,
    X_list: list[NDArray],
    Z_list: list[NDArray],
    y_list: list[NDArray],
    gh: "GHQuadrature",  # noqa: F821
    diagonal_D: bool = False,
    X_zi_list: Optional[list[NDArray]] = None,
    Z_zi_list: Optional[list[Optional[NDArray]]] = None,
) -> NDArray:
    """
    Per-group score contribution matrix for the sandwich estimator.

    Returns an ``(n_groups, n_params)`` matrix ``S`` where row ``i`` is the
    score contribution of group ``i`` w.r.t. the full parameter vector
    ``θ = [betas | D_chol | phis | gammas]``.

    The sandwich meat is then ``S.T @ S``.

    Mirrors ``score_mixed(i_contributions=TRUE)`` in ``R/Fit_Funs.R``.
    """
    from glmmadaptive.utils.linalg import cov_to_chol
    from glmmadaptive.utils.numdiff import cd_grad

    n_groups = len(X_list)
    n_betas = len(betas)
    n_re = D.shape[0]
    n_D = n_re if diagonal_D else n_re * (n_re + 1) // 2
    n_phis = len(phis) if phis is not None else 0
    n_gammas = len(gammas) if gammas is not None else 0
    n_params = n_betas + n_D + n_phis + n_gammas

    S = np.zeros((n_groups, n_params))

    _, log_det_D = np.linalg.slogdet(D)
    D_inv = np.linalg.inv(D)

    # Pre-compute Cholesky representation of D for gradient
    D_chol = cov_to_chol(D, diagonal=diagonal_D)  # (n_D,)

    for i in range(n_groups):
        X_i = X_list[i]
        Z_i = Z_list[i]
        y_i = y_list[i]
        ncz = Z_i.shape[1]
        X_zi_i = X_zi_list[i] if X_zi_list is not None else None
        Z_zi_i = Z_zi_list[i] if Z_zi_list is not None else None

        b_nodes = gh.b_nodes[i]
        log_w = gh.log_weights[i]
        n_pts = b_nodes.shape[0]

        log_unnorm = np.empty(n_pts)
        eta_pts = []
        eta_zi_pts = []

        for k in range(n_pts):
            b_k = b_nodes[k]
            eta_k = X_i @ betas + Z_i @ b_k[:ncz]
            eta_zi_k = _make_eta_zi(b_k, ncz, X_zi_i, Z_zi_i, gammas)
            eta_pts.append(eta_k)
            eta_zi_pts.append(eta_zi_k)
            log_py_b = np.sum(family.log_dens(y_i, eta_k, phis=phis, eta_zi=eta_zi_k))
            from glmmadaptive.utils.linalg import log_dmvnorm
            log_pb = log_dmvnorm(b_k, cov_inv=D_inv, log_det_cov=log_det_D)
            log_unnorm[k] = log_py_b + log_pb + log_w[k]

        post_w = np.exp(log_unnorm - logsumexp(log_unnorm))

        s_i = np.zeros(n_params)

        # --- score w.r.t. betas ---
        for k in range(n_pts):
            eta_k = eta_pts[k]
            eta_zi_k = eta_zi_pts[k]
            s_eta = family.score_eta(y_i, eta_k, phis=phis, eta_zi=eta_zi_k)
            if s_eta is None:
                _b_k = b_nodes[k]
                s_eta_full = cd_grad(
                    lambda bb: np.sum(
                        family.log_dens(y_i, X_i @ bb + Z_i @ _b_k[:ncz],
                                        phis=phis, eta_zi=eta_zi_k)
                    ),
                    betas,
                )
                s_i[:n_betas] += post_w[k] * s_eta_full
            else:
                s_i[:n_betas] += post_w[k] * (X_i.T @ s_eta)

        # --- score w.r.t. D (via Cholesky parameterisation, numerical diff) ---
        def _ll_group_D(d_chol):
            from glmmadaptive.utils.linalg import chol_to_cov, log_dmvnorm as _ldmvn
            D_ = chol_to_cov(d_chol, q=n_re, diagonal=diagonal_D)
            _, log_det_ = np.linalg.slogdet(D_)
            D_inv_ = np.linalg.inv(D_)
            log_u = np.empty(n_pts)
            for k_ in range(n_pts):
                b_k_ = b_nodes[k_]
                eta_k_ = eta_pts[k_]
                eta_zi_k_ = eta_zi_pts[k_]
                log_py_ = np.sum(family.log_dens(y_i, eta_k_, phis=phis, eta_zi=eta_zi_k_))
                log_pb_ = _ldmvn(b_k_, cov_inv=D_inv_, log_det_cov=log_det_)
                log_u[k_] = log_py_ + log_pb_ + log_w[k_]
            return logsumexp(log_u)

        s_D = cd_grad(_ll_group_D, D_chol)
        s_i[n_betas:n_betas + n_D] = s_D

        # --- score w.r.t. phis ---
        if n_phis > 0:
            for k in range(n_pts):
                eta_k = eta_pts[k]
                eta_zi_k = eta_zi_pts[k]
                s_p = family.score_phis(y_i, eta_k, phis=phis, eta_zi=eta_zi_k)
                if s_p is None:
                    _eta_k = eta_k
                    _eta_zi_k = eta_zi_k
                    s_p = cd_grad(
                        lambda ph: np.sum(
                            family.log_dens(y_i, _eta_k, phis=ph, eta_zi=_eta_zi_k)
                        ),
                        phis,
                    )
                s_i[n_betas + n_D:n_betas + n_D + n_phis] += post_w[k] * s_p

        # --- score w.r.t. gammas ---
        if n_gammas > 0:
            for k in range(n_pts):
                eta_k = eta_pts[k]
                eta_zi_k = eta_zi_pts[k]
                _b_k = b_nodes[k]
                s_zi = family.score_eta_zi(y_i, eta_k, phis=phis, eta_zi=eta_zi_k)
                if s_zi is None:
                    def _f_g(g, _b=_b_k, _e=eta_k):
                        _ezi = _make_eta_zi(_b, ncz, X_zi_i, Z_zi_i, g)
                        return np.sum(family.log_dens(y_i, _e, phis=phis, eta_zi=_ezi))
                    s_zi_grad = cd_grad(_f_g, gammas)
                    s_i[n_betas + n_D + n_phis:] += post_w[k] * s_zi_grad
                else:
                    s_i[n_betas + n_D + n_phis:] += post_w[k] * (X_zi_i.T @ s_zi)

        S[i] = s_i

    return S
