"""
Results container for a fitted GLMM.

Mirrors the ``MixMod`` class and its S3 methods in ``R/methods.R``.

Following statsmodels conventions, methods are named using Python conventions
(``summary()``, ``params``, ``bse``, ``conf_int()``, ``predict()``, etc.)
while also providing R-compatible aliases (``fixef()``, ``ranef()``,
``logLik()``, ``vcov()``, ``confint()``).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm as _norm
from scipy.special import logsumexp

from glmmadaptive.utils.linalg import cov_to_chol, chol_to_cov, log_dmvnorm, nearPD
from glmmadaptive.core.fit_funs import _pack_params, _unpack_params


class MixModResults:
    """
    Results of a fitted :class:`~glmmadaptive.core.MixedModel`.

    Attributes
    ----------
    model : MixedModel
    params : ndarray
        Fixed-effects coefficient vector β̂.
    D : ndarray
        Estimated random-effects covariance matrix.
    phis : ndarray or None
        Extra dispersion parameters (log-scale).
    logLik : float
        Marginal log-likelihood at convergence.
    converged : bool
    """

    def __init__(self, model: "MixedModel", fit_result: dict):  # noqa: F821
        self.model = model
        self.family = model.family

        # Core estimates
        self.params = fit_result["betas"]
        self.D = fit_result["D"]
        self.phis = fit_result["phis"]
        self.logLik = fit_result["logLik"]
        self.converged = fit_result["converged"]
        self.n_iter = fit_result["n_iter"]

        # Quadrature / posterior modes
        self._post_modes = fit_result["post_modes"]
        self._post_neg_hessians = fit_result["post_neg_hessians"]
        self._Hessian = fit_result["Hessian"]  # neg-Hessian of full log-lik

        # Extract parameter names from model design matrices
        self._beta_names = self._get_beta_names()

    # ------------------------------------------------------------------
    # Parameter names
    # ------------------------------------------------------------------

    def _get_beta_names(self) -> list[str]:
        """Extract column names from the fixed-effects design matrix."""
        try:
            import patsy
            _, X_dm = patsy.dmatrices(
                self.model.fixed_formula, self.model.data, return_type="dataframe"
            )
            return list(X_dm.columns)
        except Exception:
            return [f"x{i}" for i in range(len(self.params))]

    # ------------------------------------------------------------------
    # Variance-covariance of parameters  (mirrors vcov.MixMod in R)
    # ------------------------------------------------------------------

    @property
    def _vcov_betas(self) -> NDArray:
        """Covariance matrix of β̂ (from inverse Hessian, betas block)."""
        n_p = len(self.params)
        H_betas = self._Hessian[:n_p, :n_p]
        try:
            return np.linalg.inv(H_betas)
        except np.linalg.LinAlgError:
            return np.diag(1.0 / np.diag(H_betas))

    def vcov(self, parm: str = "fixed-effects") -> NDArray:
        """
        Variance-covariance matrix.

        Parameters
        ----------
        parm : str
            ``"fixed-effects"`` (default), ``"all"``, ``"var-cov"``,
            or ``"extra"`` (phis).

        Returns
        -------
        ndarray
        """
        if parm == "fixed-effects":
            return self._vcov_betas
        if parm == "all":
            return np.linalg.inv(self._Hessian)
        raise ValueError(f"Unknown parm='{parm}'")

    @property
    def bse(self) -> NDArray:
        """Standard errors of fixed-effects coefficients."""
        diag = np.diag(self._vcov_betas)
        return np.sqrt(np.maximum(diag, 0.0))

    # ------------------------------------------------------------------
    # Coefficient extraction  (mirrors coef/fixef/ranef in R)
    # ------------------------------------------------------------------

    def fixef(self) -> pd.Series:
        """
        Population-level fixed-effects coefficients.

        Mirrors ``fixef.MixMod()`` in R.
        """
        return pd.Series(self.params, index=self._beta_names)

    # Alias
    coef = fixef

    def ranef(self) -> pd.DataFrame:
        """
        Empirical Bayes estimates of random effects (posterior modes).

        Returns
        -------
        DataFrame of shape (n_groups, q) with group labels as index.
        """
        q = self.D.shape[0]
        cols = [f"b{j}" for j in range(q)]
        return pd.DataFrame(
            self._post_modes,
            index=self.model._group_labels,
            columns=cols,
        )

    # ------------------------------------------------------------------
    # Confidence intervals  (mirrors confint.MixMod in R)
    # ------------------------------------------------------------------

    def confint(self, level: float = 0.95, parm: str = "fixed-effects") -> pd.DataFrame:
        """
        Wald confidence intervals for fixed-effects coefficients.

        Parameters
        ----------
        level : float
            Confidence level (default 0.95).
        parm : str
            Currently only ``"fixed-effects"`` supported.

        Returns
        -------
        DataFrame with columns ``["2.5 %", "97.5 %"]`` (or adjusted for level).
        """
        if parm != "fixed-effects":
            raise NotImplementedError(f"confint for parm='{parm}' not yet implemented")
        alpha = 1.0 - level
        z = _norm.ppf(1.0 - alpha / 2.0)
        lo = self.params - z * self.bse
        hi = self.params + z * self.bse
        lo_pct = f"{100 * alpha / 2:.1f} %"
        hi_pct = f"{100 * (1 - alpha / 2):.1f} %"
        return pd.DataFrame(
            {lo_pct: lo, hi_pct: hi},
            index=self._beta_names,
        )

    # Alias
    conf_int = confint

    # ------------------------------------------------------------------
    # Log-likelihood and information criteria
    # ------------------------------------------------------------------

    def log_likelihood(self) -> float:
        """Marginal log-likelihood (alias of logLik attribute)."""
        return self.logLik

    @property
    def df_model(self) -> int:
        """Number of free parameters (fixed effects + D params + phis)."""
        q = self.D.shape[0]
        n_D = q if self.model.control["diagonal_D"] else q * (q + 1) // 2
        n_phis = len(self.phis) if self.phis is not None else 0
        return len(self.params) + n_D + n_phis

    @property
    def aic(self) -> float:
        return -2.0 * self.logLik + 2.0 * self.df_model

    @property
    def bic(self) -> float:
        return -2.0 * self.logLik + self.df_model * np.log(self.model.nobs)

    # ------------------------------------------------------------------
    # Predictions  (mirrors predict.MixMod in R)
    # ------------------------------------------------------------------

    def predict(
        self,
        newdata: Optional[pd.DataFrame] = None,
        type_: str = "mean_subject",
    ) -> NDArray:
        """
        Compute predictions.

        Parameters
        ----------
        newdata : DataFrame, optional
            New covariates.  If None, predictions on training data.
        type_ : str
            ``"mean_subject"`` — fixed effects only (population mean subject).
            ``"subject_specific"`` — add empirical Bayes random effects.

        Returns
        -------
        ndarray of shape (N,) — predicted means μ̂.
        """
        if newdata is None:
            X = self.model._X
            groups = self.model._groups
        else:
            import patsy
            _, X = patsy.dmatrices(
                self.model.fixed_formula, newdata, return_type="matrix"
            )
            X = np.asarray(X)
            groups_raw = newdata[self.model.id_name].values
            _, groups = np.unique(groups_raw, return_inverse=True)

        eta = X @ self.params

        if type_ == "subject_specific":
            b_all = self._post_modes  # (n_groups, q)
            if newdata is None:
                Z = self.model._Z
                for i in range(self.model._n_groups):
                    mask = groups == i
                    eta[mask] += self.model._Z[mask] @ b_all[i]
            else:
                # Reconstruct Z for newdata
                import patsy
                Z = np.asarray(patsy.dmatrix(
                    f"~ {self.model.re_rhs}", newdata, return_type="matrix"
                ))
                label_to_idx = {
                    lbl: idx for idx, lbl in enumerate(self.model._group_labels)
                }
                groups_raw = newdata[self.model.id_name].values
                for i, lbl in enumerate(np.unique(groups_raw)):
                    if lbl in label_to_idx:
                        b_i = b_all[label_to_idx[lbl]]
                        mask = groups_raw == lbl
                        eta[mask] += Z[mask] @ b_i

        return self.family.linkinv(eta)

    # ------------------------------------------------------------------
    # Fitted values and residuals
    # ------------------------------------------------------------------

    def fitted(self, type_: str = "mean_subject") -> NDArray:
        """Fitted values on the training data."""
        return self.predict(type_=type_)

    def residuals(self, type_: str = "mean_subject") -> NDArray:
        """Raw residuals: y - ŷ."""
        return self.model._y - self.fitted(type_=type_)

    # ------------------------------------------------------------------
    # Likelihood ratio test  (mirrors anova.MixMod in R)
    # ------------------------------------------------------------------

    @staticmethod
    def anova(model1: "MixModResults", model2: "MixModResults") -> pd.DataFrame:
        """
        Likelihood ratio test between two nested models.

        Parameters
        ----------
        model1, model2 : MixModResults
            The less and more complex models (order doesn't matter — the one
            with higher log-likelihood is assumed to be the larger model).

        Returns
        -------
        DataFrame with AIC, BIC, logLik, LRT statistic, df, p-value.
        """
        from scipy.stats import chi2

        ll1, ll2 = model1.logLik, model2.logLik
        df1, df2 = model1.df_model, model2.df_model

        if ll1 > ll2:
            model1, model2 = model2, model1
            ll1, ll2 = ll2, ll1
            df1, df2 = df2, df1

        lrt = 2.0 * (ll2 - ll1)
        dof = df2 - df1
        p_val = 1.0 - chi2.cdf(lrt, dof) if dof > 0 else np.nan

        return pd.DataFrame(
            {
                "AIC": [model1.aic, model2.aic],
                "BIC": [model1.bic, model2.bic],
                "logLik": [ll1, ll2],
                "LRT": [np.nan, lrt],
                "df": [np.nan, dof],
                "p-value": [np.nan, p_val],
            },
            index=["Model 1", "Model 2"],
        )

    # ------------------------------------------------------------------
    # Summary  (mirrors summary.MixMod in R)
    # ------------------------------------------------------------------

    def summary(self) -> "MixModSummary":
        """Return a :class:`MixModSummary` object (print for formatted output)."""
        return MixModSummary(self)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        converged = "converged" if self.converged else "DID NOT CONVERGE"
        return (
            f"MixModResults:\n"
            f"  Family: {self.family}\n"
            f"  logLik: {self.logLik:.4f}  AIC: {self.aic:.4f}  BIC: {self.bic:.4f}\n"
            f"  Status: {converged} after {self.n_iter} EM iterations\n"
        )


# ---------------------------------------------------------------------------
# Summary container
# ---------------------------------------------------------------------------

class MixModSummary:
    """Formatted summary of a :class:`MixModResults`."""

    def __init__(self, results: MixModResults):
        self.results = results

    def _coef_table(self) -> pd.DataFrame:
        r = self.results
        z = r.params / np.where(r.bse > 0, r.bse, np.nan)
        p = 2.0 * _norm.sf(np.abs(z))
        return pd.DataFrame(
            {
                "Estimate": r.params,
                "Std.Err": r.bse,
                "z value": z,
                "Pr(>|z|)": p,
            },
            index=r._beta_names,
        )

    def __str__(self) -> str:
        r = self.results
        m = r.model
        lines = []
        lines.append("=" * 65)
        lines.append("Generalized Linear Mixed Model (Adaptive GH Quadrature)")
        lines.append("=" * 65)
        lines.append(f"  Family : {r.family.family},  link = {r.family.link}")
        lines.append(f"  Formula: {m.fixed_formula}")
        lines.append(f"  Random : {m.random_formula}")
        lines.append(f"  Groups : {m.n_groups}  ('{m.id_name}')")
        lines.append(f"  Nobs   : {m.nobs}")
        lines.append("")
        lines.append("Fixed Effects:")
        lines.append(self._coef_table().to_string(float_format="{:.4f}".format))
        lines.append("")
        q = r.D.shape[0]
        if q == 1:
            lines.append(f"Random Effects StdDev: {np.sqrt(r.D[0,0]):.4f}")
        else:
            lines.append("Random Effects Covariance Matrix (D):")
            lines.append(pd.DataFrame(r.D).to_string(float_format="{:.4f}".format))
        if r.phis is not None:
            lines.append(f"\nExtra parameters (log-scale): {r.phis}")
        lines.append("")
        lines.append(f"  logLik : {r.logLik:.4f}")
        lines.append(f"  AIC    : {r.aic:.4f}")
        lines.append(f"  BIC    : {r.bic:.4f}")
        status = "YES" if r.converged else "NO (check convergence)"
        lines.append(f"  Converged: {status}  ({r.n_iter} EM iterations)")
        lines.append("=" * 65)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()
