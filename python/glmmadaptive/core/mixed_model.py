"""
Main user-facing class: MixedModel.

Follows statsmodels conventions:
* Model is constructed from a formula string and a DataFrame.
* Calling ``model.fit()`` returns a :class:`~glmmadaptive.results.MixModResults`
  object that holds all post-estimation information.

Mirrors ``mixed_model()`` in ``R/mixed_model.R``.
"""

from __future__ import annotations

import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import logit
from statsmodels.formula.api import glm as sm_glm
import statsmodels.api as sm

from glmmadaptive.families.base import BaseFamily
from glmmadaptive.families.standard import Binomial, Poisson
from glmmadaptive.core.mixed_fit import mixed_fit, DEFAULT_CONTROL
from glmmadaptive.utils.linalg import nearPD


# ---------------------------------------------------------------------------
# Formula parsing helpers  (R's lme4-style "~ x | id" syntax)
# ---------------------------------------------------------------------------

def _parse_random_formula(
    formula: str,
    data: pd.DataFrame,
) -> tuple[str, str, str]:
    """
    Parse a random-effects formula ``"~ x1 + x2 | id"`` into components.

    Returns
    -------
    re_formula : str
        Right-hand side of random effects (e.g. ``"x1 + x2"``).
    id_name : str
        Name of the grouping variable.
    diagonal : bool
        True when ``||`` separator is used (diagonal D).
    """
    formula = formula.strip()
    if formula.startswith("~"):
        formula = formula[1:].strip()

    diagonal = "||" in formula
    sep = "||" if diagonal else "|"
    parts = formula.split(sep, 1)
    if len(parts) != 2:
        raise ValueError(
            f"Random formula must contain '|' or '||', got: '{formula}'"
        )
    re_rhs = parts[0].strip()
    id_name = parts[1].strip()
    if id_name not in data.columns:
        raise ValueError(
            f"Grouping variable '{id_name}' not found in data"
        )
    return re_rhs, id_name, diagonal


def _build_design_matrices(
    fixed_formula: str,
    re_rhs: str,
    id_name: str,
    data: pd.DataFrame,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Construct X (fixed), Z (random), y, and group index arrays.

    Parameters
    ----------
    fixed_formula : str
        Full formula string including response, e.g. ``"y ~ x1 + x2"``.
    re_rhs : str
        Random-effects RHS (without grouping), e.g. ``"1"`` or ``"time"``.
    id_name : str
        Grouping variable name.
    data : DataFrame

    Returns
    -------
    y : ndarray of shape (N,)
    X : ndarray of shape (N, p) — fixed-effects design matrix
    Z : ndarray of shape (N, q) — random-effects design matrix
    groups : ndarray of shape (N,) int — group indices 0..n_groups-1
    group_labels : ndarray — original group labels
    """
    import patsy

    # Fixed effects
    y, X = patsy.dmatrices(fixed_formula, data, return_type="matrix")
    y = np.asarray(y).ravel()
    X = np.asarray(X)

    # Random effects design matrix (intercept-only or with terms)
    re_formula_full = f"~ {re_rhs}"
    Z = np.asarray(patsy.dmatrix(re_formula_full, data, return_type="matrix"))

    # Group indices
    groups_raw = data[id_name].values
    group_labels, groups = np.unique(groups_raw, return_inverse=True)

    return y, X, Z, groups, group_labels


def _split_by_group(
    y: NDArray,
    X: NDArray,
    Z: NDArray,
    groups: NDArray,
    n_groups: int,
) -> tuple[list[NDArray], list[NDArray], list[NDArray]]:
    """Split arrays into per-group lists (in group order)."""
    y_list = []
    X_list = []
    Z_list = []
    for i in range(n_groups):
        mask = groups == i
        y_list.append(y[mask])
        X_list.append(X[mask])
        Z_list.append(Z[mask])
    return y_list, X_list, Z_list


# ---------------------------------------------------------------------------
# Initial value computation  (mirrors mixed_model.R initial value section)
# ---------------------------------------------------------------------------

def _initial_betas(y: NDArray, X: NDArray, family: BaseFamily) -> NDArray:
    """
    Fit a marginal GLM and scale coefficients by sqrt(1.346) as in the R code.

    The factor 1.346 accounts for the random-effects variance inflating the
    variance relative to a marginal model.
    """
    try:
        sm_family = _to_statsmodels_family(family)
        glm_mod = sm.GLM(y, X, family=sm_family).fit(disp=False)
        betas = glm_mod.params * np.sqrt(1.346)
    except Exception:
        # Fallback: least-squares
        betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return np.asarray(betas)


def _to_statsmodels_family(family: BaseFamily):
    """Map GLMMadaptive family → statsmodels family for initialisation only."""
    from statsmodels.genmod import families as smf

    if isinstance(family, Binomial):
        link_map = {
            "logit": smf.links.Logit(),
            "probit": smf.links.Probit(),
            "cloglog": smf.links.CLogLog(),
        }
        return smf.Binomial(link=link_map.get(family.link, smf.links.Logit()))
    if isinstance(family, Poisson):
        return smf.Poisson()
    # Default: Gaussian for initialisation
    return smf.Gaussian()


def _initial_phis(family: BaseFamily) -> Optional[NDArray]:
    """Return zero-valued phis of the right length."""
    if family.n_phis == 0:
        return None
    return np.zeros(family.n_phis)


# ---------------------------------------------------------------------------
# MixedModel class
# ---------------------------------------------------------------------------

class MixedModel:
    """
    Generalized Linear Mixed Model fitted via adaptive Gauss-Hermite quadrature.

    Follows statsmodels conventions: the model is specified at construction time
    and fitted by calling :meth:`fit`.

    Parameters
    ----------
    fixed : str
        Patsy formula for the fixed effects including the response variable,
        e.g. ``"y ~ time + treatment"``.
    random : str
        Random-effects formula in lme4 notation, e.g. ``"~ 1 | id"`` or
        ``"~ time | id"``.  Use ``||`` to constrain *D* to be diagonal.
    data : DataFrame
        Data frame containing all variables referenced in the formulas.
    family : BaseFamily
        Response distribution.  Defaults to :class:`~glmmadaptive.families.Binomial`.
    initial_values : dict, optional
        Override starting values.  Keys: ``"betas"``, ``"D"``, ``"phis"``.
    control : dict, optional
        Override optimisation control parameters (see :data:`DEFAULT_CONTROL`).

    Examples
    --------
    >>> model = MixedModel(
    ...     fixed="cbind(y, n-y) ~ time * treatment",
    ...     random="~ 1 | id",
    ...     data=df,
    ...     family=Binomial(),
    ... )
    >>> res = model.fit(verbose=True)
    >>> print(res.summary())
    """

    def __init__(
        self,
        fixed: str,
        random: str,
        data: pd.DataFrame,
        family: BaseFamily | None = None,
        initial_values: dict | None = None,
        control: dict | None = None,
    ):
        if family is None:
            family = Binomial()

        if family.has_zi:
            raise NotImplementedError(
                "Zero-inflated / hurdle families are not yet implemented in the "
                "Python port. Use the R package for these models."
            )

        # Convert tibbles/etc. to plain DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        self.fixed_formula = fixed
        self.random_formula = random
        self.data = data.copy()
        self.family = family
        self.initial_values = initial_values or {}
        self.control = {**DEFAULT_CONTROL, **(control or {})}

        # Parse random formula
        re_rhs, id_name, diagonal_D = _parse_random_formula(random, data)
        self.re_rhs = re_rhs
        self.id_name = id_name
        self.control["diagonal_D"] = self.control.get("diagonal_D", False) or diagonal_D

        # Build design matrices
        (
            self._y,
            self._X,
            self._Z,
            self._groups,
            self._group_labels,
        ) = _build_design_matrices(fixed, re_rhs, id_name, data)

        self._n_groups = len(self._group_labels)
        self._n_betas = self._X.shape[1]
        self._n_re = self._Z.shape[1]

        # Split into per-group lists
        self._y_list, self._X_list, self._Z_list = _split_by_group(
            self._y, self._X, self._Z, self._groups, self._n_groups
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, verbose: bool = False, **kwargs) -> "MixModResults":  # noqa: F821
        """
        Fit the model and return a :class:`~glmmadaptive.results.MixModResults`
        object.

        Parameters
        ----------
        verbose : bool
            Print iteration details.
        **kwargs
            Additional control overrides (e.g. ``iter_em=50``).

        Returns
        -------
        MixModResults
        """
        from glmmadaptive.results.mixmod_results import MixModResults

        ctrl = {**self.control, "verbose": verbose, **kwargs}

        # Initial values
        betas0 = np.asarray(
            self.initial_values.get("betas",
                _initial_betas(self._y, self._X, self.family))
        )
        D0 = np.asarray(
            self.initial_values.get("D",
                np.eye(self._n_re))
        )
        phis0 = self.initial_values.get("phis",
            _initial_phis(self.family))
        if phis0 is not None:
            phis0 = np.asarray(phis0)

        fit_result = mixed_fit(
            betas_init=betas0,
            D_init=D0,
            phis_init=phis0,
            family=self.family,
            X_list=self._X_list,
            Z_list=self._Z_list,
            y_list=self._y_list,
            control=ctrl,
        )

        return MixModResults(model=self, fit_result=fit_result)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nobs(self) -> int:
        """Total number of observations."""
        return len(self._y)

    @property
    def n_groups(self) -> int:
        """Number of groups."""
        return self._n_groups

    @property
    def endog(self) -> NDArray:
        return self._y

    @property
    def exog(self) -> NDArray:
        return self._X

    def __repr__(self) -> str:
        return (
            f"MixedModel(\n"
            f"  fixed='{self.fixed_formula}',\n"
            f"  random='{self.random_formula}',\n"
            f"  family={self.family!r},\n"
            f"  n={self.nobs}, groups={self.n_groups}\n"
            f")"
        )
