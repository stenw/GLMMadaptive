"""
Core GLMM fitting engine: hybrid EM + quasi-Newton algorithm.

Mirrors ``R/mixed_fit.R`` from the original GLMMadaptive package.

The algorithm has two phases:

1. **EM phase** (``iter_em`` iterations): E-step (update adaptive GH
   quadrature) followed by M-step (closed-form D update, Newton-Raphson
   for betas/gammas, L-BFGS-B for phis).

2. **Quasi-Newton phase**: direct joint maximisation of the marginal
   log-likelihood via ``scipy.optimize.minimize``.

Zero-inflated models
--------------------
When ``X_zi_list`` / ``Z_zi_list`` / ``gammas_init`` are supplied the
combined random-effects vector ``b = [b_count, b_zi]`` is used throughout,
the D matrix has the combined dimension, and the M-step also updates
``gammas`` (ZI fixed effects).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from glmmadaptive.families.base import BaseFamily
from glmmadaptive.utils.linalg import nearPD, cov_to_chol, chol_to_cov, log_dmvnorm
from glmmadaptive.utils.quadrature import gh_adaptive, find_posterior_mode
from glmmadaptive.utils.numdiff import cd_hess, cd_grad, fd_hess
from glmmadaptive.core.fit_funs import (
    loglik_mixed,
    loglik_mixed_vec,
    score_betas,
    score_D,
    score_phis,
    score_gammas,
    _pack_params,
    _unpack_params,
)


# ---------------------------------------------------------------------------
# Default control settings
# ---------------------------------------------------------------------------

DEFAULT_CONTROL = {
    "iter_em": 30,
    "iter_qn_outer": 15,
    "iter_qn": 10,
    "iter_qn_incr": 10,
    "optimizer": "BFGS",
    "tol1": 1e-4,
    "tol2": 1e-5,
    "tol3": 1e-8,
    "n_agh": None,
    "update_gh_every": 10,
    "max_coef": 12.0,
    "max_phis": np.exp(10.0),
    "verbose": False,
    "diagonal_D": False,
}


# ---------------------------------------------------------------------------
# Posterior-mode computation (E-step)
# ---------------------------------------------------------------------------

def _compute_posterior_modes(
    betas: NDArray,
    D: NDArray,
    phis: Optional[NDArray],
    family: BaseFamily,
    X_list: list[NDArray],
    Z_list: list[NDArray],
    y_list: list[NDArray],
    X_zi_list: Optional[list[NDArray]] = None,
    Z_zi_list: Optional[list[Optional[NDArray]]] = None,
    gammas: Optional[NDArray] = None,
    modes_init: Optional[NDArray] = None,
) -> tuple[NDArray, list[NDArray]]:
    """Return posterior modes (n_groups, q) and neg-Hessians for all groups."""
    n_groups = len(X_list)
    q = D.shape[0]  # combined dim = q_count + q_zi
    D_inv = np.linalg.inv(D)
    _, log_det_D = np.linalg.slogdet(D)

    modes = np.zeros((n_groups, q)) if modes_init is None else modes_init.copy()
    neg_hessians = []

    for i in range(n_groups):
        X_i = X_list[i]
        Z_i = Z_list[i]
        y_i = y_list[i]
        ncz = Z_i.shape[1]
        X_zi_i = X_zi_list[i] if X_zi_list is not None else None
        Z_zi_i = Z_zi_list[i] if Z_zi_list is not None else None

        def neg_log_post(b, _Xi=X_i, _Zi=Z_i, _yi=y_i,
                         _Xzi=X_zi_i, _Zzi=Z_zi_i):
            eta = _Xi @ betas + _Zi @ b[:ncz]
            eta_zi_here = None
            if _Xzi is not None and gammas is not None:
                eta_zi_here = _Xzi @ gammas
                if _Zzi is not None and _Zzi.shape[1] > 0:
                    eta_zi_here = eta_zi_here + _Zzi @ b[ncz:]
            ll = np.sum(family.log_dens(_yi, eta, phis=phis, eta_zi=eta_zi_here))
            lp = log_dmvnorm(b, cov_inv=D_inv, log_det_cov=log_det_D)
            return -(ll + lp)

        mode_i, H_i = find_posterior_mode(neg_log_post, modes[i])
        modes[i] = mode_i
        neg_hessians.append(nearPD(H_i))

    return modes, neg_hessians


# ---------------------------------------------------------------------------
# M-step helpers
# ---------------------------------------------------------------------------

def _update_betas(
    betas, D, phis, family, X_list, Z_list, y_list, gh,
    max_coef: float,
    X_zi_list=None, Z_zi_list=None, gammas=None,
) -> NDArray:
    """One Newton-Raphson step for betas."""

    def neg_ll(bb):
        return -loglik_mixed(
            bb, D, phis, family, X_list, Z_list, y_list, gh,
            X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=gammas,
        )

    H = nearPD(fd_hess(neg_ll, betas))
    grad = -score_betas(
        betas, D, phis, family, X_list, Z_list, y_list, gh,
        X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=gammas,
    )
    try:
        delta = np.linalg.solve(H, grad)
    except np.linalg.LinAlgError:
        delta = grad
    return np.clip(betas - delta, -max_coef, max_coef)


def _update_phis(
    betas, D, phis, family, X_list, Z_list, y_list, gh,
    max_phis: float,
    X_zi_list=None, Z_zi_list=None, gammas=None,
) -> Optional[NDArray]:
    """Update phis via bounded L-BFGS-B."""
    if phis is None or len(phis) == 0:
        return phis
    log_max = np.log(max_phis)

    def neg_ll_phis(ph):
        return -loglik_mixed(
            betas, D, ph, family, X_list, Z_list, y_list, gh,
            X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=gammas,
        )

    bounds = [(-log_max, log_max)] * len(phis)
    result = minimize(
        neg_ll_phis, phis, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 10, "ftol": 1e-9, "gtol": 1e-6},
    )
    return result.x


def _update_gammas(
    betas, D, phis, gammas, family, X_list, Z_list, y_list, gh,
    max_coef: float,
    X_zi_list=None, Z_zi_list=None,
) -> NDArray:
    """One Newton-Raphson step for ZI fixed effects (gammas)."""

    def neg_ll(g):
        return -loglik_mixed(
            betas, D, phis, family, X_list, Z_list, y_list, gh,
            X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=g,
        )

    H = nearPD(fd_hess(neg_ll, gammas))
    grad = -score_gammas(
        betas, D, phis, gammas, family, X_list, Z_list, y_list, gh,
        X_zi_list, Z_zi_list,
    )
    try:
        delta = np.linalg.solve(H, grad)
    except np.linalg.LinAlgError:
        delta = grad
    return np.clip(gammas - delta, -max_coef, max_coef)


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
    X_zi_list: Optional[list[NDArray]] = None,
    Z_zi_list: Optional[list[Optional[NDArray]]] = None,
    gammas_init: Optional[NDArray] = None,
) -> dict:
    """
    Fit a GLMM via the hybrid EM + quasi-Newton algorithm.

    Mirrors ``mixed_fit()`` in ``R/mixed_fit.R``.

    Parameters
    ----------
    betas_init : ndarray of shape (p,)
    D_init : ndarray of shape (q, q)  — combined dim for ZI models
    phis_init : ndarray or None
    family : BaseFamily
    X_list, Z_list, y_list : per-group data lists
    X_zi_list, Z_zi_list, gammas_init : ZI parameters (optional)
    control : dict of control parameters

    Returns
    -------
    dict with keys: betas, D, phis, gammas, logLik, Hessian,
    post_modes, post_neg_hessians, converged, n_iter, control.
    """
    ctrl = {**DEFAULT_CONTROL, **(control or {})}
    verbose = ctrl["verbose"]
    diagonal_D = ctrl["diagonal_D"]

    has_zi = (X_zi_list is not None and gammas_init is not None)

    betas = betas_init.copy()
    D = D_init.copy()
    phis = phis_init.copy() if phis_init is not None else None
    gammas = gammas_init.copy() if gammas_init is not None else None

    n_re = D.shape[0]
    n_agh = ctrl["n_agh"] or (11 if n_re <= 2 else 7)
    n_groups = len(X_list)

    modes = np.zeros((n_groups, n_re))
    converged = False
    n_iter = 0
    ll_prev = -np.inf

    # ------------------------------------------------------------------
    # Phase 1: EM
    # ------------------------------------------------------------------
    for em_iter in range(ctrl["iter_em"]):
        n_iter += 1

        if em_iter % ctrl["update_gh_every"] == 0:
            modes, neg_hess = _compute_posterior_modes(
                betas, D, phis, family, X_list, Z_list, y_list,
                X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=gammas,
                modes_init=modes,
            )

        gh = gh_adaptive(n_agh, n_re, modes, neg_hess)

        # M-step: D (closed-form EM update)
        D_new_raw = score_D(
            betas, D, phis, family, X_list, Z_list, y_list, gh,
            diagonal_D=diagonal_D,
            X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=gammas,
        )
        D = np.diag(np.diag(D_new_raw)) if diagonal_D else nearPD(D_new_raw)

        # M-step: betas
        betas_new = _update_betas(
            betas, D, phis, family, X_list, Z_list, y_list, gh,
            max_coef=ctrl["max_coef"],
            X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=gammas,
        )

        # M-step: phis
        phis_new = _update_phis(
            betas, D, phis, family, X_list, Z_list, y_list, gh,
            max_phis=ctrl["max_phis"],
            X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=gammas,
        )

        # M-step: gammas (ZI fixed effects)
        gammas_new = gammas
        if has_zi and gammas is not None:
            gammas_new = _update_gammas(
                betas, D, phis, gammas, family, X_list, Z_list, y_list, gh,
                max_coef=ctrl["max_coef"],
                X_zi_list=X_zi_list, Z_zi_list=Z_zi_list,
            )

        # Log-likelihood
        ll = loglik_mixed(
            betas_new, D, phis_new, family, X_list, Z_list, y_list, gh,
            X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=gammas_new,
        )

        # Convergence check
        delta_params = np.max(np.abs(betas_new - betas))
        if phis is not None and phis_new is not None:
            delta_params = max(delta_params, np.max(np.abs(phis_new - phis)))
        if has_zi and gammas is not None and gammas_new is not None:
            delta_params = max(delta_params, np.max(np.abs(gammas_new - gammas)))
        delta_ll = abs(ll - ll_prev) / (abs(ll_prev) + 1.0 + ctrl["tol3"])

        if verbose:
            print(f"EM iter {em_iter+1}: logLik={ll:.6f}  Δparam={delta_params:.2e}")

        betas = betas_new
        phis = phis_new
        gammas = gammas_new
        ll_prev = ll

        if delta_params < ctrl["tol1"] and delta_ll < ctrl["tol2"]:
            converged = True
            if verbose:
                print(f"EM converged at iteration {em_iter+1}")
            break

    # ------------------------------------------------------------------
    # Phase 2: Quasi-Newton
    # ------------------------------------------------------------------
    if not converged:
        n_phis = len(phis) if phis is not None else 0
        n_betas = len(betas)
        n_gammas = len(gammas) if gammas is not None else 0

        def neg_ll(theta):
            try:
                b, D_, p_, g_ = _unpack_params(
                    theta, n_betas, n_re, n_phis, diagonal_D, n_gammas
                )
                return loglik_mixed(
                    b, D_, p_, family, X_list, Z_list, y_list, gh,
                    X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=g_,
                    sign=-1.0,
                )
            except Exception:
                return 1e10

        theta0 = _pack_params(betas, D, phis, diagonal_D, gammas)
        iter_qn = ctrl["iter_qn"]

        for outer in range(ctrl["iter_qn_outer"]):
            modes, neg_hess = _compute_posterior_modes(
                betas, D, phis, family, X_list, Z_list, y_list,
                X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=gammas,
                modes_init=modes,
            )
            gh = gh_adaptive(n_agh, n_re, modes, neg_hess)

            result = minimize(
                neg_ll, theta0,
                method=ctrl["optimizer"],
                options={"maxiter": iter_qn, "gtol": ctrl["tol3"]},
            )
            theta0 = result.x
            betas, D, phis, gammas = _unpack_params(
                theta0, n_betas, n_re, n_phis, diagonal_D, n_gammas
            )
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
    # Post-convergence: Hessian
    # ------------------------------------------------------------------
    modes, neg_hess = _compute_posterior_modes(
        betas, D, phis, family, X_list, Z_list, y_list,
        X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=gammas,
        modes_init=modes,
    )
    gh_final = gh_adaptive(n_agh, n_re, modes, neg_hess)
    ll_final = loglik_mixed(
        betas, D, phis, family, X_list, Z_list, y_list, gh_final,
        X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=gammas,
    )

    n_phis = len(phis) if phis is not None else 0
    n_betas = len(betas)
    n_gammas_val = len(gammas) if gammas is not None else 0
    theta_final = _pack_params(betas, D, phis, diagonal_D, gammas)

    def neg_ll_final(theta):
        b, D_, p_, g_ = _unpack_params(
            theta, n_betas, n_re, n_phis, diagonal_D, n_gammas_val
        )
        return -loglik_mixed(
            b, D_, p_, family, X_list, Z_list, y_list, gh_final,
            X_zi_list=X_zi_list, Z_zi_list=Z_zi_list, gammas=g_,
        )

    try:
        H = cd_hess(neg_ll_final, theta_final)
        H = nearPD(H)
    except Exception:
        H = np.eye(len(theta_final)) * 1e6

    return {
        "betas": betas,
        "D": D,
        "phis": phis,
        "gammas": gammas,
        "logLik": ll_final,
        "Hessian": H,
        "post_modes": modes,
        "post_neg_hessians": neg_hess,
        "converged": converged,
        "n_iter": n_iter,
        "control": ctrl,
        "n_betas": n_betas,
        "n_re": n_re,
        "n_phis": n_phis,
        "n_gammas": n_gammas_val,
    }
