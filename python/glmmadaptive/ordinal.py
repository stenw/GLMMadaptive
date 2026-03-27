"""
Helpers for fitting continuation ratio (CR) mixed models for ordinal outcomes.

The continuation ratio approach re-expresses an ordinal mixed model as a
standard binomial GLMM on expanded pseudo-observation data.  No new family
class is needed — fit with ``MixedModel(..., family=Binomial())``.

References
----------
Tutz, G. (1991). Sequential item response models with an ordered response.
    *British Journal of Mathematical and Statistical Psychology*, 44, 87–101.

Ported from ``R/Functions.R:cr_setup`` (lines 956-996) and
``R/Functions.R:cr_marg_probs`` (lines 1015-1032).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, log_expit


def cr_setup(y, direction: str = "forward") -> dict:
    """
    Transform an ordinal response into binary pseudo-observations for CR model fitting.

    Each original observation with ordinal value ``k`` is expanded into multiple
    binary pseudo-observations, one per CR cohort comparison.  The expanded data
    can then be passed to :class:`~glmmadaptive.MixedModel` with
    ``family=Binomial()``.

    Mirrors ``cr_setup()`` from ``R/Functions.R:956``.

    Parameters
    ----------
    y : array-like
        Ordinal response vector with at least 3 distinct levels.  Can be
        integer codes, strings, or a :class:`pandas.Categorical`.  Levels are
        determined by ``numpy.unique`` (for numeric/string arrays) or by
        ``categories`` (for Categoricals).
    direction : {"forward", "backward"}, default "forward"
        * ``"forward"`` — each cohort asks "given Y ≥ k, did Y stop at k?"
        * ``"backward"`` — each cohort asks "given Y ≤ k, is Y exactly k?"

    Returns
    -------
    dict with keys:

    y : numpy.ndarray of int (0/1)
        Binary outcomes for every pseudo-observation.
    cohort : pandas.Categorical
        Cohort label for every pseudo-observation.  Levels ordered from
        ``"all"`` to the highest-order comparison.
    subs : numpy.ndarray of int
        0-based index into the original ``y`` array for each pseudo-observation
        (use to replicate covariate rows in the expanded dataset).
    reps : numpy.ndarray of int
        Number of pseudo-observations generated from each original observation;
        ``len(subs) == reps.sum()``.

    Raises
    ------
    ValueError
        If ``y`` has fewer than 3 distinct levels (use logistic regression instead).
    """
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")

    # Resolve levels and convert y to 0-based integer codes.
    if isinstance(y, pd.Categorical):
        levels = list(y.categories)
        y_codes = np.asarray(y.codes, dtype=int)
    else:
        y_arr = np.asarray(y)
        levels = list(np.unique(y_arr[~pd.isnull(y_arr)]))
        lev_map = {lv: i for i, lv in enumerate(levels)}
        y_codes = np.array([lev_map[v] for v in y_arr], dtype=int)

    K = len(levels)
    ncoefs = K - 1
    if ncoefs < 2:
        raise ValueError(
            "y has fewer than 3 levels; use a mixed-effects logistic regression instead."
        )

    str_levels = [str(lv) for lv in levels]
    n = len(y_codes)

    if direction == "forward":
        # reps[i] = min(y[i] + 1, ncoefs)
        reps = np.where(y_codes < ncoefs - 1, y_codes + 1, ncoefs)

        # For obs with value k: cuts = [0, 1, ..., min(k, ncoefs-1)]
        cuts_list: list[int] = []
        for k in y_codes:
            max_cut = min(k, ncoefs - 1)
            cuts_list.extend(range(max_cut + 1))
        cuts = np.array(cuts_list, dtype=int)

        subs = np.repeat(np.arange(n), reps)
        y_rep = np.repeat(y_codes, reps)
        Y = (y_rep == cuts).astype(int)

        labels = ["all"] + [f"y>={str_levels[j]}" for j in range(1, ncoefs)]

    else:  # backward
        # reps[i]: ncoefs if k <= 1, else ncoefs - k + 1
        reps = np.where(y_codes <= 1, ncoefs, ncoefs - y_codes + 1)

        # For obs with value k: cuts = [0, 1, ..., ncoefs - max(k-1,1)]
        cuts_list = []
        for k in y_codes:
            n_cuts = ncoefs if k <= 1 else ncoefs - k + 1
            cuts_list.extend(range(n_cuts))
        cuts = np.array(cuts_list, dtype=int)

        subs = np.repeat(np.arange(n), reps)
        y_rep = np.repeat(y_codes, reps)
        Y = (y_rep == (ncoefs - cuts)).astype(int)

        # labels: ["all", "y<=levels[-2]", "y<=levels[-3]", ..., "y<=levels[1]"]
        labels = ["all"] + [f"y<={str_levels[j]}" for j in range(ncoefs - 1, 0, -1)]

    cohort = pd.Categorical.from_codes(cuts, categories=labels)
    return {"y": Y, "cohort": cohort, "subs": subs, "reps": reps}


def cr_marg_probs(eta, direction: str = "forward") -> np.ndarray:
    """
    Compute marginal category probabilities P(Y = k) from CR linear predictors.

    Converts the conditional CR probabilities to marginal category probabilities
    using the chain rule (computed in log-space for numerical stability).

    * Forward:
      P(Y = j) = σ(η_j) · ∏_{k<j} [1 − σ(η_k)]
    * Backward:
      P(Y = j) = σ(η_j) · ∏_{k>j} [1 − σ(η_k)]

    The last (forward) or first (backward) category probability is obtained as
    1 − sum of the remaining probabilities.

    Mirrors ``cr_marg_probs()`` from ``R/Functions.R:1015``.

    Parameters
    ----------
    eta : array-like, shape (n, K-1)
        Linear predictor matrix, one column per CR cohort comparison.  Must have
        at least 2 columns (K ≥ 3 categories).
    direction : {"forward", "backward"}, default "forward"
        Must match the ``direction`` used in :func:`cr_setup`.

    Returns
    -------
    numpy.ndarray of shape (n, K)
        Marginal category probabilities; each row sums to 1.

    Raises
    ------
    ValueError
        If ``eta`` is 1-D or has fewer than 2 columns.
    """
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")

    eta = np.asarray(eta, dtype=float)
    if eta.ndim != 2 or eta.shape[1] < 2:
        raise ValueError("eta must be a 2-D array with at least 2 columns (K >= 3 categories).")

    n, ncoefs = eta.shape

    if direction == "forward":
        # log(1 − σ(η)) for all but the last cohort column
        log1m_p = log_expit(-eta[:, :-1])          # (n, ncoefs-1)
        cumsum_log1m_p = np.cumsum(log1m_p, axis=1)  # (n, ncoefs-1)

        # Prepend column of 0 so the first category has no preceding factor
        cumsum_prefix = np.hstack([np.zeros((n, 1)), cumsum_log1m_p])  # (n, ncoefs)

        log_p = log_expit(eta) + cumsum_prefix      # (n, ncoefs)
        probs = np.exp(log_p)                        # (n, ncoefs)

        last = 1.0 - probs.sum(axis=1, keepdims=True)
        return np.hstack([probs, last])              # (n, K)

    else:  # backward
        # log(1 − σ(η)) for columns in reverse order (last to second)
        eta_rev = eta[:, -1:0:-1]                    # (n, ncoefs-1), reversed
        log1m_p = log_expit(-eta_rev)                # (n, ncoefs-1)
        cumsum_log1m_p = np.cumsum(log1m_p, axis=1)  # (n, ncoefs-1)

        # Reverse columns and append 0 for the last cohort (no succeeding factor)
        cumsum_rev = cumsum_log1m_p[:, ::-1]         # (n, ncoefs-1), un-reversed
        cumsum_suffix = np.hstack([cumsum_rev, np.zeros((n, 1))])  # (n, ncoefs)

        log_p = log_expit(eta) + cumsum_suffix       # (n, ncoefs)
        probs = np.exp(log_p)                        # (n, ncoefs)

        first = 1.0 - probs.sum(axis=1, keepdims=True)
        return np.hstack([first, probs])             # (n, K)
