"""
Main user-facing class: MixedModel.

Follows statsmodels conventions:
* Model is constructed from a formula string and a DataFrame.
* Calling ``model.fit()`` returns a :class:`~glmmadaptive.results.MixModResults`
  object that holds all post-estimation information.

Mirrors ``mixed_model()`` in ``R/mixed_model.R``.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
import statsmodels.api as sm

from glmmadaptive.families.base import BaseFamily
from glmmadaptive.families.standard import Binomial, Poisson
from glmmadaptive.core.mixed_fit import mixed_fit, DEFAULT_CONTROL
from glmmadaptive.utils.linalg import nearPD


# ---------------------------------------------------------------------------
# Formula parsing helpers
# ---------------------------------------------------------------------------

def _parse_random_formula(formula: str, data: pd.DataFrame) -> tuple[str, str, bool]:
    """
    Parse ``"~ x1 + x2 | id"`` → (re_rhs, id_name, diagonal).
    """
    formula = formula.strip()
    if formula.startswith("~"):
        formula = formula[1:].strip()
    diagonal = "||" in formula
    sep = "||" if diagonal else "|"
    parts = formula.split(sep, 1)
    if len(parts) != 2:
        raise ValueError(f"Random formula must contain '|' or '||', got: '{formula}'")
    re_rhs = parts[0].strip()
    id_name = parts[1].strip()
    if id_name not in data.columns:
        raise ValueError(f"Grouping variable '{id_name}' not found in data")
    return re_rhs, id_name, diagonal


def _build_design_matrices(
    fixed_formula: str,
    re_rhs: str,
    id_name: str,
    data: pd.DataFrame,
) -> tuple[NDArray, NDArray, "patsy.DesignInfo", NDArray, NDArray, NDArray]:  # noqa: F821
    """Construct y, X, X_design_info, Z, groups, group_labels."""
    import patsy
    import re as _re
    cbind_match = _re.match(
        r'\s*cbind\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*~\s*(.*)', fixed_formula
    )
    if cbind_match:
        col1, col2, rhs = (
            cbind_match.group(1), cbind_match.group(2), cbind_match.group(3)
        )
        y = np.column_stack([data[col1].values, data[col2].values]).astype(float)
        X_mat = patsy.dmatrix(rhs, data, return_type="matrix")
        X_design_info = X_mat.design_info
        X = np.asarray(X_mat)
    else:
        y_mat, X_mat = patsy.dmatrices(fixed_formula, data, return_type="matrix")
        X_design_info = X_mat.design_info          # preserve before ndarray conversion
        y = np.asarray(y_mat).ravel()
        X = np.asarray(X_mat)
    Z = np.asarray(patsy.dmatrix(f"~ {re_rhs}", data, return_type="matrix"))
    groups_raw = data[id_name].values
    group_labels, groups = np.unique(groups_raw, return_inverse=True)
    return y, X, X_design_info, Z, groups, group_labels


def _build_zi_design_matrix(zi_formula: str, data: pd.DataFrame) -> NDArray:
    """Build X_zi from formula like '~ sex' (no response variable)."""
    import patsy
    rhs = zi_formula.strip()
    if rhs.startswith("~"):
        rhs = rhs[1:].strip()
    return np.asarray(patsy.dmatrix(f"~ {rhs}", data, return_type="matrix"))


def _split_by_group(y, X, Z, groups, n_groups):
    y_list, X_list, Z_list = [], [], []
    for i in range(n_groups):
        mask = groups == i
        y_list.append(y[mask])
        X_list.append(X[mask])
        Z_list.append(Z[mask])
    return y_list, X_list, Z_list


def _split_zi_by_group(X_zi, Z_zi, groups, n_groups):
    X_zi_list = []
    Z_zi_list = []
    for i in range(n_groups):
        mask = groups == i
        X_zi_list.append(X_zi[mask])
        Z_zi_list.append(Z_zi[mask] if Z_zi is not None else None)
    return X_zi_list, Z_zi_list


# ---------------------------------------------------------------------------
# Initial value helpers
# ---------------------------------------------------------------------------

def _initial_betas(y: NDArray, X: NDArray, family: BaseFamily) -> NDArray:
    try:
        sm_family = _to_statsmodels_family(family)
        if y.ndim == 2:
            N = y[:, 0] + y[:, 1]
            prop = y[:, 0] / np.maximum(N, 1)
            glm_mod = sm.GLM(prop, X, family=sm_family, var_weights=N).fit(disp=False)
        else:
            glm_mod = sm.GLM(y, X, family=sm_family).fit(disp=False)
        return np.asarray(glm_mod.params * np.sqrt(1.346))
    except Exception:
        y_flat = y[:, 0] / np.maximum(y[:, 0] + y[:, 1], 1) if y.ndim == 2 else y
        betas, _, _, _ = np.linalg.lstsq(X, y_flat, rcond=None)
        return np.asarray(betas)


def _to_statsmodels_family(family: BaseFamily):
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
    return smf.Gaussian()


def _initial_phis(family: BaseFamily) -> Optional[NDArray]:
    if family.n_phis == 0:
        return None
    return np.zeros(family.n_phis)


def _initial_gammas(y: NDArray, X_zi: NDArray) -> NDArray:
    """Initialise ZI fixed effects from logistic regression of I(y==0) ~ X_zi."""
    try:
        y_zi = (y[:, 0] == 0 if y.ndim == 2 else y == 0).astype(float)
        glm_mod = sm.Logit(y_zi, X_zi).fit(disp=False, method="bfgs", maxiter=100)
        return np.asarray(glm_mod.params)
    except Exception:
        return np.zeros(X_zi.shape[1])


# ---------------------------------------------------------------------------
# MixedModel class
# ---------------------------------------------------------------------------

class MixedModel:
    """
    Generalized Linear Mixed Model fitted via adaptive Gauss-Hermite quadrature.

    Parameters
    ----------
    fixed : str
        Patsy formula for the fixed effects, e.g. ``"y ~ time + sex"``.
    random : str
        lme4-style random-effects formula, e.g. ``"~ 1 | id"``.
    data : DataFrame
    family : BaseFamily, optional
        Defaults to :class:`~glmmadaptive.families.Binomial`.
    zi_fixed : str, optional
        Formula for the ZI fixed effects, e.g. ``"~ sex"``.
        Required for zero-inflated families.
    zi_random : str, optional
        lme4-style formula for ZI random effects, e.g. ``"~ 1 | id"``.
    initial_values : dict, optional
        Override starting values.  Keys: ``"betas"``, ``"D"``, ``"phis"``,
        ``"gammas"``.
    control : dict, optional
        Override optimisation control parameters.
    """

    def __init__(
        self,
        fixed: str,
        random: str,
        data: pd.DataFrame,
        family: BaseFamily | None = None,
        zi_fixed: str | None = None,
        zi_random: str | None = None,
        penalized: bool | dict = False,
        initial_values: dict | None = None,
        control: dict | None = None,
    ):
        if family is None:
            family = Binomial()

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        self.fixed_formula = fixed
        self.random_formula = random
        self.data = data.copy()
        self.family = family
        self.zi_fixed = zi_fixed
        self.zi_random = zi_random
        self.initial_values = initial_values or {}
        self.control = {**DEFAULT_CONTROL, **(control or {})}

        # -- Parse penalized (mirrors R's mixed_model() argument handling) ----
        if isinstance(penalized, bool) and not penalized:
            self._penalized = {"penalized": False}
        elif isinstance(penalized, bool) and penalized:
            self._penalized = {"penalized": True, "pen_mu": 0.0, "pen_sigma": 1.0, "pen_df": 3.0}
        elif isinstance(penalized, dict):
            allowed = {"pen_mu", "pen_sigma", "pen_df"}
            bad = set(penalized) - allowed
            if bad:
                raise ValueError(
                    f"Unknown 'penalized' keys: {bad}. "
                    f"Expected a subset of {allowed}."
                )
            self._penalized = {"penalized": True, "pen_mu": 0.0, "pen_sigma": 1.0, "pen_df": 3.0}
            self._penalized.update(penalized)
        else:
            raise TypeError("'penalized' must be bool or dict")

        # Parse random formula
        re_rhs, id_name, diagonal_D = _parse_random_formula(random, data)
        self.re_rhs = re_rhs
        self.id_name = id_name
        self.control["diagonal_D"] = self.control.get("diagonal_D", False) or diagonal_D

        # Build count-part design matrices
        (
            self._y, self._X, self._X_design_info, self._Z,
            self._groups, self._group_labels,
        ) = _build_design_matrices(fixed, re_rhs, id_name, data)

        self._n_groups = len(self._group_labels)
        self._n_betas = self._X.shape[1]
        self._n_re_count = self._Z.shape[1]

        self._y_list, self._X_list, self._Z_list = _split_by_group(
            self._y, self._X, self._Z, self._groups, self._n_groups
        )

        # Build ZI design matrices (if needed)
        self._has_zi = family.has_zi
        self._X_zi = None
        self._Z_zi = None
        self._X_zi_list = None
        self._Z_zi_list = None
        self._n_gammas = 0
        self._n_re_zi = 0
        self._zi_re_rhs = None

        if self._has_zi:
            if zi_fixed is None:
                raise ValueError(
                    f"Family '{family.family}' requires 'zi_fixed' formula."
                )
            self._X_zi = _build_zi_design_matrix(zi_fixed, data)
            self._n_gammas = self._X_zi.shape[1]

            if zi_random is not None:
                zi_re_rhs, zi_id, _ = _parse_random_formula(zi_random, data)
                if zi_id != id_name:
                    raise ValueError(
                        "ZI random effects must use the same grouping variable "
                        f"as the count part ('{id_name}')."
                    )
                self._zi_re_rhs = zi_re_rhs
                import patsy
                self._Z_zi = np.asarray(
                    patsy.dmatrix(f"~ {zi_re_rhs}", data, return_type="matrix")
                )
                self._n_re_zi = self._Z_zi.shape[1]

            self._X_zi_list, self._Z_zi_list = _split_zi_by_group(
                self._X_zi, self._Z_zi, self._groups, self._n_groups
            )

        # Total number of random effects (count + ZI)
        self._n_re = self._n_re_count + self._n_re_zi

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, verbose: bool = False, **kwargs) -> "MixModResults":  # noqa: F821
        """Fit the model and return a :class:`~glmmadaptive.results.MixModResults`."""
        from glmmadaptive.results.mixmod_results import MixModResults

        ctrl = {**self.control, "verbose": verbose, **kwargs}

        betas0 = np.asarray(
            self.initial_values.get(
                "betas", _initial_betas(self._y, self._X, self.family)
            )
        )
        D0 = np.asarray(
            self.initial_values.get("D", np.eye(self._n_re))
        )
        phis0 = self.initial_values.get("phis", _initial_phis(self.family))
        if phis0 is not None:
            phis0 = np.asarray(phis0)

        gammas0 = None
        if self._has_zi:
            gammas0 = np.asarray(
                self.initial_values.get(
                    "gammas", _initial_gammas(self._y, self._X_zi)
                )
            )

        fit_result = mixed_fit(
            betas_init=betas0,
            D_init=D0,
            phis_init=phis0,
            family=self.family,
            X_list=self._X_list,
            Z_list=self._Z_list,
            y_list=self._y_list,
            control=ctrl,
            X_zi_list=self._X_zi_list,
            Z_zi_list=self._Z_zi_list,
            gammas_init=gammas0,
            penalized=self._penalized,
        )

        return MixModResults(model=self, fit_result=fit_result)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nobs(self) -> int:
        return len(self._y)

    @property
    def n_groups(self) -> int:
        return self._n_groups

    @property
    def endog(self) -> NDArray:
        return self._y

    @property
    def exog(self) -> NDArray:
        return self._X

    def __repr__(self) -> str:
        zi_str = f"\n  zi_fixed='{self.zi_fixed}'" if self.zi_fixed else ""
        return (
            f"MixedModel(\n"
            f"  fixed='{self.fixed_formula}',\n"
            f"  random='{self.random_formula}',{zi_str}\n"
            f"  family={self.family!r},\n"
            f"  n={self.nobs}, groups={self.n_groups}\n"
            f")"
        )
