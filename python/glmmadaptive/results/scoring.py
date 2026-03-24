"""
Proper scoring rules for dynamic predictions from fitted GLMMs.

Implements the logarithmic, quadratic (Brier), and spherical scoring rules
for count-data mixed models.

Mirrors ``scoring_rules()`` in the R GLMMadaptive package.

Theory
------
For a discrete predictive distribution P = {P(Y=k) : k=0,1,...} and an
observed outcome y, the three proper scoring rules are:

* **Logarithmic**: log P(Y = y)  ∈ (-∞, 0];  closer to 0 = better.
* **Quadratic**:   2 P(Y = y) - Σ_k P(Y=k)²
* **Spherical**:   P(Y = y) / sqrt(Σ_k P(Y=k)²)  ∈ [0, 1];  closer to 1 = better.

The predictive PMF is obtained by Monte Carlo integration over the posterior
of the random effects:

    P(Y = k | y_j(t)) ≈ (1/M) Σ_m P(Y = k | b_j^(m), θ̂)

where b_j^(m) ~ N(b̂_j, Σ_j) is a draw from the Laplace approximation to the
random-effects posterior given the observations up to time t.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import expit


def _pmf_poisson(k_arr: NDArray, mu: float) -> NDArray:
    """Poisson PMF for all k in k_arr."""
    from scipy.stats import poisson
    return poisson.pmf(k_arr, mu=max(mu, 1e-15))


def _pmf_nb(k_arr: NDArray, mu: float, theta: float) -> NDArray:
    """Negative Binomial PMF (NB2) for all k in k_arr."""
    from scipy.stats import nbinom
    # NB2: P(Y=k | μ, θ) with Var = μ + μ²/θ
    # scipy parameterisation: n=theta, p=theta/(theta+mu)
    p = theta / (theta + max(mu, 1e-15))
    return nbinom.pmf(k_arr, n=theta, p=p)


def _pmf_zi_poisson(k_arr: NDArray, mu: float, pi: float) -> NDArray:
    """ZI Poisson PMF for all k in k_arr."""
    base = _pmf_poisson(k_arr, mu)
    pmf = (1.0 - pi) * base
    pmf[0] += pi  # structural zeros add to k=0
    return pmf


def _pmf_zi_nb(k_arr: NDArray, mu: float, theta: float, pi: float) -> NDArray:
    """ZI Negative Binomial PMF for all k in k_arr."""
    base = _pmf_nb(k_arr, mu, theta)
    pmf = (1.0 - pi) * base
    pmf[0] += pi
    return pmf


def scoring_rules(
    results: "MixModResults",  # noqa: F821
    newdata: pd.DataFrame,
    newdata2: Optional[pd.DataFrame] = None,
    max_count: int = 1000,
    return_newdata: bool = True,
    n_mc: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute proper scoring rules for dynamic predictions.

    For each observation in *newdata2* (or *newdata* when *newdata2* is
    ``None``), derives the full predictive PMF by Monte Carlo averaging over
    the posterior of the random effects estimated from *newdata*, then evaluates
    the logarithmic, quadratic and spherical scoring rules against the observed
    outcome.

    Mirrors ``scoring_rules()`` in the R GLMMadaptive package.

    Parameters
    ----------
    results : MixModResults
        A fitted model object.
    newdata : DataFrame
        Observations used to estimate each subject's random effects
        (the "information period").
    newdata2 : DataFrame, optional
        Future observations for which scores are computed.  If ``None``,
        scores are computed for *newdata* itself.
    max_count : int
        Upper truncation point for the count support (k = 0, …, max_count).
    return_newdata : bool
        If True, return a copy of *newdata2* (or *newdata*) with the scoring
        columns appended.  If False, return only the scoring columns.
    n_mc : int
        Monte Carlo draws from the random-effects posterior.
    seed : int
        Random seed.

    Returns
    -------
    DataFrame with columns ``logarithmic``, ``quadratic``, ``spherical``
    (appended to *newdata2* when *return_newdata=True*).
    """
    import patsy

    rng = np.random.default_rng(seed)
    model = results.model
    id_col = model.id_name
    ncz = model._n_re_count

    score_df = newdata2 if newdata2 is not None else newdata

    # ---- Extract response variable name from formula -----------------------
    resp_name = model.fixed_formula.split("~")[0].strip()

    # ---- Determine family type for PMF computation -------------------------
    fam = model.family
    family_name = fam.family

    # For NB families, extract theta
    def _get_theta():
        if hasattr(fam, "theta_fixed") and fam.theta_fixed is not None:
            return float(fam.theta_fixed)
        if results.phis is not None and len(results.phis) > 0:
            return float(np.exp(results.phis[0]))
        return 1.0

    # ---- 1. Estimate posterior for each subject in newdata -----------------
    subject_ids = newdata[id_col].unique()
    subject_info: dict = {}
    for sid in subject_ids:
        subj_mask = newdata[id_col] == sid
        mode, cov_b, _, _, _, _, _ = results._get_subject_posterior(
            newdata[subj_mask].reset_index(drop=True)
        )
        subject_info[sid] = {"mode": mode, "cov_b": cov_b}

    # ---- 2. Build design matrices for score_df ----------------------------
    _, X_all = patsy.dmatrices(model.fixed_formula, score_df, return_type="matrix")
    X_all = np.asarray(X_all)
    Z_all = np.asarray(
        patsy.dmatrix(f"~ {model.re_rhs}", score_df, return_type="matrix")
    )

    X_zi_all, Z_zi_all = None, None
    if model._has_zi and model.zi_fixed is not None:
        rhs_zi = model.zi_fixed.strip().lstrip("~").strip()
        X_zi_all = np.asarray(
            patsy.dmatrix(f"~ {rhs_zi}", score_df, return_type="matrix")
        )
        if model._zi_re_rhs is not None:
            Z_zi_all = np.asarray(
                patsy.dmatrix(
                    f"~ {model._zi_re_rhs}", score_df, return_type="matrix"
                )
            )

    ids_arr = score_df[id_col].values
    y_obs = score_df[resp_name].values
    N = len(score_df)
    k_arr = np.arange(0, max_count + 1, dtype=np.float64)

    # ---- 3. Monte Carlo averaging of predictive PMF ------------------------
    # pmf_mc[m, j, k] = P(Y=k | b_j^(m)) for each draw m and observation j
    # For memory efficiency, we accumulate the sum over m iteratively.

    pmf_sum = np.zeros((N, max_count + 1))

    for m in range(n_mc):
        eta = X_all @ results.params
        eta_zi = (
            X_zi_all @ results.gammas
            if (X_zi_all is not None and results.gammas is not None)
            else None
        )

        for sid in subject_ids:
            mask = ids_arr == sid
            if not np.any(mask):
                continue
            info = subject_info[sid]
            b_j = rng.multivariate_normal(info["mode"], info["cov_b"])
            eta[mask] += Z_all[mask] @ b_j[:ncz]
            if eta_zi is not None and Z_zi_all is not None and Z_zi_all.shape[1] > 0:
                eta_zi[mask] += Z_zi_all[mask] @ b_j[ncz:]

        # Compute per-observation PMF for this MC draw
        mu_arr = model.family.linkinv(eta)
        pi_arr = expit(eta_zi) if eta_zi is not None else None

        for j in range(N):
            mu_j = float(mu_arr[j])
            if pi_arr is not None:
                pi_j = float(pi_arr[j])
            else:
                pi_j = 0.0

            if family_name == "poisson":
                pmf_j = _pmf_poisson(k_arr, mu_j)
            elif family_name == "negative_binomial":
                pmf_j = _pmf_nb(k_arr, mu_j, _get_theta())
            elif family_name == "zi_poisson":
                pmf_j = _pmf_zi_poisson(k_arr, mu_j, pi_j)
            elif family_name == "zi_negative_binomial":
                pmf_j = _pmf_zi_nb(k_arr, mu_j, _get_theta(), pi_j)
            else:
                # Fallback: Poisson
                pmf_j = _pmf_poisson(k_arr, mu_j)

            pmf_sum[j] += pmf_j

    pmf_avg = pmf_sum / n_mc  # (N, max_count+1)

    # ---- 4. Compute scoring rules ------------------------------------------
    log_scores = np.zeros(N)
    quad_scores = np.zeros(N)
    sph_scores = np.zeros(N)

    for j in range(N):
        pmf_j = pmf_avg[j]
        # Normalise (should already sum to ~1, but numerical safety)
        pmf_j = pmf_j / max(pmf_j.sum(), 1e-15)

        y_j = int(y_obs[j])
        p_y = pmf_j[y_j] if y_j <= max_count else 1e-15
        p_y = max(p_y, 1e-300)

        sum_sq = np.sum(pmf_j ** 2)
        sqrt_sum_sq = np.sqrt(max(sum_sq, 1e-300))

        log_scores[j] = np.log(p_y)
        quad_scores[j] = 2.0 * p_y - sum_sq
        sph_scores[j] = p_y / sqrt_sum_sq

    # ---- 5. Return results -------------------------------------------------
    if return_newdata:
        out = score_df.copy()
        out["logarithmic"] = log_scores
        out["quadratic"] = quad_scores
        out["spherical"] = sph_scores
        return out
    else:
        return pd.DataFrame(
            {"logarithmic": log_scores, "quadratic": quad_scores, "spherical": sph_scores}
        )
