"""
Core GLMM fitting engine: hybrid EM + quasi-Newton algorithm.

Mirrors ``R/mixed_fit.R`` from the original GLMMadaptive package.

The algorithm has two phases:

1. **EM phase** (``iter_em`` iterations): each iteration performs an E-step
   (update adaptive GH quadrature at the current posterior modes) and an M-step
   (update parameters via Newton-Raphson sub-steps or closed-form updates).

2. **Quasi-Newton phase** (``iter_qn_outer`` outer iterations): if the EM does
   not converge, switches to direct marginal log-likelihood maximisation using
   ``scipy.optimize.minimize`` (BFGS) or statsmodels' ``GenericLikelihoodModel``
   optimizer wrappers.  The quadrature is updated at the start of each outer
   iteration.

Reference: Pinheiro & Bates (1995) DOI 10.1080/10618600.1995.10474663.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from glmmadaptive.families.base import BaseFamily
from glmmadaptive.utils.linalg import nearPD, cov_to_chol, chol_to_cov, log_dmvnorm
from glmmadaptive.utils.quadrature import gh_adaptive, find_posterior_mode
from glmmadaptive.utils.numdiff import cd_hess, cd_grad
from glmmadaptive.core.fit_funs import (
    loglik_mixed,
    loglik_mixed_vec,
    score_betas,
    score_D,
    score_phis,
    _pack_params,
    _unpack_params,
)


# ---------------------------------------------------------------------------
# Default control settings  (mirrors mixed_model.R defaults)
# ---------------------------------------------------------------------------

DEFAULT_CONTROL = {
    "iter_em": 30,              # EM iterations
    "iter_qn_outer": 15,        # quasi-Newton outer iterations
    "iter_qn": 10,              # max optimizer iterations per call
    "iter_qn_incr": 10,         # increment iter_qn each outer pass
    "optimizer": "BFGS",        # "BFGS", "L-BFGS-B", or "Nelder-Mead"
    "tol1": 1e-4,               # param change tolerance (EM)
    "tol2": 1e-5,               # scaled param change tolerance (EM)
    "tol3": 1e-8,               # optimizer relative tolerance
    "n_agh": None,              # None → auto (11 if n_re <= 2 else 7)
    "update_gh_every": 10,      # update GH nodes every N EM iters
    "max_coef": 12.0,           # clip |beta| at this value
    "max_phis": np.exp(10.0),   # clip phis at this value
    "verbose": False,
    "diagonal_D": False,        # force diagonal covariance matrix
}


# ---------------------------------------------------------------------------
# Compute per-group posterior modes for quadrature initialisation
# ---------------------------------------------------------------------------

def _compute_posterior_modes(
    betas, D, phis, family,
    X_list, Z_list, y_list,
    modes_init: Optional[NDArray] = None,
):
    """Return modes (n_groups, q) and neg-Hessians for all groups."""
    n_groups = len(X_list)
    q = D.shape[0]
    D_inv = np.linalg.inv(D)
    _, log_det_D = np.linalg.slogdet(D)

    modes = np.zeros((n_groups, q)) if modes_init is None else modes_init.copy()
    neg_hessians = []

    for i in range(n_groups):
        X_i = X_list[i]
        Z_i = Z_list[i]
        y_i = y_list[i]

        def neg_log_post(b):
            eta = X_i @ betas + Z_i @ b
            ll = np.sum(family.log_dens(y_i, eta, phis=phis))
            lp = log_dmvnorm(b, cov_inv=D_inv, log_det_cov=log_det_D)
            return -(ll + lp)

        mode_i, H_i = find_posterior_mode(neg_log_post, modes[i])
        modes[i] = mode_i
        neg_hessians.append(nearPD(H_i))

    return modes, neg_hessians


# ---------------------------------------------------------------------------
# EM M-step: update betas (Newton-Raphson)
# ---------------------------------------------------------------------------

def _update_betas(betas, D, phis, family, X_list, Z_list, y_list, gh,
                  max_coef: float):
    """One Newton-Raphson step for betas (mirrors M-step in mixed_fit.R)."""
    p = len(betas)

    def neg_score(bb):
        return -score_betas(bb, D, phis, family, X_list, Z_list, y_list, gh)

    # Approximate Hessian via finite differences of score
    from glmmadaptive.utils.numdiff import fd_hess
    H_approx = fd_hess(
        lambda bb: -loglik_mixed(bb, D, phis, family, X_list, Z_list, y_list, gh),
        betas,
    )
    H_approx = nearPD(H_approx)
    grad = neg_score(betas)

    try:
        delta = np.linalg.solve(H_approx, grad)
    except np.linalg.LinAlgError:
        delta = grad

    betas_new = betas - delta
    betas_new = np.clip(betas_new, -max_coef, max_coef)
    return betas_new


# ---------------------------------------------------------------------------
# EM M-step: update phis (Newton-Raphson)
# ---------------------------------------------------------------------------

def _update_phis(betas, D, phis, family, X_list, Z_list, y_list, gh,
                 max_phis: float):
    """
    Update phis via bounded L-BFGS-B optimisation (a few steps).

    A raw Newton step is unreliable for dispersion parameters because the
    marginal likelihood is often flat in that direction — the Hessian is tiny
    and the Newton step overshoots badly.  L-BFGS-B with bounds is robust
    and avoids the need for a line search.
    """
    if phis is None or len(phis) == 0:
        return phis

    log_max = np.log(max_phis)

    def neg_ll_phis(ph):
        ph = np.asarray(ph)
        return -loglik_mixed(betas, D, ph, family, X_list, Z_list, y_list, gh)

    bounds = [(-log_max, log_max)] * len(phis)
    result = minimize(
        neg_ll_phis, phis, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 10, "ftol": 1e-9, "gtol": 1e-6},
    )
    return result.x


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def mixed_fit(
    betas_init: NDArray,
    D_init: NDArray,
    phis_init: Optional[NDArray],
    family: BaseFamily,
    X_list: list[NDArray],
    Z_list: list[NDArray],
    y_list: list[NDArray],
    control: dict | None = None,
) -> dict:
    """
    Fit a GLMM via the hybrid EM + quasi-Newton algorithm.

    Mirrors ``mixed_fit()`` in ``R/mixed_fit.R``.

    Parameters
    ----------
    betas_init : ndarray of shape (p,)
    D_init : ndarray of shape (q, q)
    phis_init : ndarray or None
    family : BaseFamily
    X_list : list of (n_i, p) arrays
    Z_list : list of (n_i, q) arrays
    y_list : list of (n_i,) arrays
    control : dict
        Overrides for :data:`DEFAULT_CONTROL`.

    Returns
    -------
    dict
        ``betas``, ``D``, ``phis``, ``logLik``, ``Hessian``,
        ``post_modes``, ``converged``, ``n_iter``.
    """
    ctrl = {**DEFAULT_CONTROL, **(control or {})}
    verbose = ctrl["verbose"]
    diagonal_D = ctrl["diagonal_D"]

    betas = betas_init.copy()
    D = D_init.copy()
    phis = phis_init.copy() if phis_init is not None else None

    n_re = D.shape[0]
    n_agh = ctrl["n_agh"] or (11 if n_re <= 2 else 7)
    n_groups = len(X_list)

    # Initialise posterior modes at zeros
    modes = np.zeros((n_groups, n_re))
    converged = False
    n_iter = 0

    ll_prev = -np.inf

    # ------------------------------------------------------------------
    # Phase 1: EM
    # ------------------------------------------------------------------
    for em_iter in range(ctrl["iter_em"]):
        n_iter += 1

        # Update quadrature every update_gh_every iterations
        if em_iter % ctrl["update_gh_every"] == 0:
            modes, neg_hess = _compute_posterior_modes(
                betas, D, phis, family, X_list, Z_list, y_list, modes_init=modes
            )

        gh = gh_adaptive(n_agh, n_re, modes, neg_hess)

        # M-step: D (closed-form EM update)
        D_new_raw = score_D(betas, D, phis, family, X_list, Z_list, y_list, gh,
                            diagonal_D=diagonal_D)
        if diagonal_D:
            D = np.diag(np.diag(D_new_raw))
        else:
            D = nearPD(D_new_raw)

        # M-step: betas (Newton-Raphson)
        betas_new = _update_betas(betas, D, phis, family, X_list, Z_list, y_list,
                                  gh, max_coef=ctrl["max_coef"])

        # M-step: phis (Newton-Raphson, if applicable)
        phis_new = _update_phis(betas, D, phis, family, X_list, Z_list, y_list,
                                gh, max_phis=ctrl["max_phis"])

        # Log-likelihood
        ll = loglik_mixed(betas_new, D, phis_new, family,
                          X_list, Z_list, y_list, gh)

        # Convergence check
        delta_params = np.max(np.abs(betas_new - betas))
        if phis is not None and phis_new is not None:
            delta_params = max(delta_params, np.max(np.abs(phis_new - phis)))
        delta_ll = abs(ll - ll_prev) / (abs(ll_prev) + 1.0 + ctrl["tol3"])

        if verbose:
            print(f"EM iter {em_iter+1}: logLik={ll:.6f}  Δparam={delta_params:.2e}")

        betas = betas_new
        phis = phis_new
        ll_prev = ll

        if delta_params < ctrl["tol1"] and delta_ll < ctrl["tol2"]:
            converged = True
            if verbose:
                print(f"EM converged at iteration {em_iter+1}")
            break

    # ------------------------------------------------------------------
    # Phase 2: Quasi-Newton (if EM didn't fully converge)
    # ------------------------------------------------------------------
    if not converged:
        n_phis = len(phis) if phis is not None else 0
        n_betas = len(betas)

        def neg_ll(theta):
            try:
                b, D_, p_ = _unpack_params(theta, n_betas, n_re, n_phis, diagonal_D)
                # Refresh modes every outer iteration is handled outside
                return loglik_mixed(b, D_, p_, family, X_list, Z_list, y_list,
                                    gh, sign=-1.0)
            except Exception:
                return 1e10

        theta0 = _pack_params(betas, D, phis, diagonal_D)
        iter_qn = ctrl["iter_qn"]

        for outer in range(ctrl["iter_qn_outer"]):
            # Refresh quadrature
            modes, neg_hess = _compute_posterior_modes(
                betas, D, phis, family, X_list, Z_list, y_list, modes_init=modes
            )
            gh = gh_adaptive(n_agh, n_re, modes, neg_hess)

            result = minimize(
                neg_ll,
                theta0,
                method=ctrl["optimizer"],
                options={"maxiter": iter_qn, "gtol": ctrl["tol3"]},
            )
            theta0 = result.x
            betas, D, phis = _unpack_params(theta0, n_betas, n_re, n_phis, diagonal_D)
            D = nearPD(D)

            ll = -result.fun
            delta_ll = abs(ll - ll_prev) / (abs(ll_prev) + 1.0 + ctrl["tol3"])
            ll_prev = ll
            iter_qn += ctrl["iter_qn_incr"]

            if verbose:
                print(f"qN outer {outer+1}: logLik={ll:.6f}  converged={result.success}")

            if result.success or delta_ll < ctrl["tol3"]:
                converged = True
                break

    # ------------------------------------------------------------------
    # Post-convergence: compute Hessian and per-group logLik contributions
    # ------------------------------------------------------------------
    # Final quadrature
    modes, neg_hess = _compute_posterior_modes(
        betas, D, phis, family, X_list, Z_list, y_list, modes_init=modes
    )
    gh_final = gh_adaptive(n_agh, n_re, modes, neg_hess)

    ll_final = loglik_mixed(betas, D, phis, family, X_list, Z_list, y_list, gh_final)

    # Hessian of negative log-likelihood w.r.t. betas only (for SE)
    n_phis = len(phis) if phis is not None else 0
    n_betas = len(betas)
    theta_final = _pack_params(betas, D, phis, diagonal_D)

    def neg_ll_final(theta):
        b, D_, p_ = _unpack_params(theta, n_betas, n_re, n_phis, diagonal_D)
        return -loglik_mixed(b, D_, p_, family, X_list, Z_list, y_list, gh_final)

    try:
        H = cd_hess(neg_ll_final, theta_final)
        H = nearPD(H)
    except Exception:
        H = np.eye(len(theta_final)) * 1e6  # fallback: large variance

    return {
        "betas": betas,
        "D": D,
        "phis": phis,
        "logLik": ll_final,
        "Hessian": H,
        "post_modes": modes,
        "post_neg_hessians": neg_hess,
        "converged": converged,
        "n_iter": n_iter,
        "control": ctrl,
    }
