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

        # ZI parameters
        self.gammas = fit_result.get("gammas", None)
        self._gamma_names = self._get_gamma_names() if self.gammas is not None else []

        # Per-group score contributions (for sandwich estimator)
        self._score_contributions = fit_result.get("score_contributions", None)

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

    def _get_gamma_names(self) -> list[str]:
        """Extract column names from the ZI fixed-effects design matrix."""
        try:
            import patsy
            zi_formula = self.model.zi_fixed
            if zi_formula is None:
                return []
            rhs = zi_formula.strip().lstrip("~").strip()
            X_zi_dm = patsy.dmatrix(f"~ {rhs}", self.model.data, return_type="dataframe")
            return [f"ZI:{c}" for c in X_zi_dm.columns]
        except Exception:
            return [f"gamma{i}" for i in range(len(self.gammas))]

    # ------------------------------------------------------------------
    # Variance-covariance of parameters  (mirrors vcov.MixMod in R)
    # ------------------------------------------------------------------

    @property
    def _vcov_betas(self) -> NDArray:
        """Covariance matrix of β̂ (from inverse Hessian, betas block)."""
        return self.vcov(parm="fixed-effects")

    def _param_slices(self) -> dict:
        """
        Return a dict mapping parameter block names to (start, stop) index
        slices in the full parameter vector θ = [betas | D_chol | phis | gammas].
        """
        n_b = len(self.params)
        n_re = self.D.shape[0]
        diagonal_D = self.model.control.get("diagonal_D", False)
        n_D = n_re if diagonal_D else n_re * (n_re + 1) // 2
        n_phi = len(self.phis) if self.phis is not None else 0
        n_gam = len(self.gammas) if self.gammas is not None else 0
        return {
            "fixed-effects": (0, n_b),
            "var-cov": (n_b, n_b + n_D),
            "extra": (n_b + n_D, n_b + n_D + n_phi),
            "zero_part": (n_b + n_D + n_phi, n_b + n_D + n_phi + n_gam),
        }

    def vcov(self, parm: str = "fixed-effects", sandwich: bool = False) -> NDArray:
        """
        Variance-covariance matrix.

        Parameters
        ----------
        parm : str
            ``"fixed-effects"`` (default), ``"all"``, ``"var-cov"``,
            ``"extra"`` (phis), or ``"zero_part"`` (gammas).
        sandwich : bool
            If True, return the sandwich (robust) estimator
            H^{-1} (∑_i sᵢ sᵢᵀ) H^{-1}.  Requires per-group score
            contributions stored at fit time.

        Returns
        -------
        ndarray
        """
        V = np.linalg.inv(self._Hessian)
        if sandwich:
            S = getattr(self, "_score_contributions", None)
            if S is None:
                raise RuntimeError(
                    "Sandwich estimator requires per-group score contributions. "
                    "Re-fit the model (score contributions are stored automatically)."
                )
            meat = S.T @ S
            V = V @ meat @ V

        if parm == "all":
            return V
        slices = self._param_slices()
        if parm not in slices:
            raise ValueError(
                f"Unknown parm='{parm}'. "
                f"Choose from: {list(slices.keys()) + ['all']}"
            )
        s, e = slices[parm]
        if e <= s:
            raise ValueError(
                f"Model has no '{parm}' parameters."
            )
        return V[s:e, s:e]

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

    def confint(
        self,
        level: float = 0.95,
        parm: str = "fixed-effects",
        sandwich: bool = False,
    ) -> pd.DataFrame:
        """
        Wald confidence intervals.

        Parameters
        ----------
        level : float
            Confidence level (default 0.95).
        parm : str
            ``"fixed-effects"`` (default), ``"var-cov"`` (variance components),
            ``"extra"`` (dispersion phis), ``"zero_part"`` (ZI gammas).
        sandwich : bool
            If True, use sandwich standard errors.

        Returns
        -------
        DataFrame with columns ``["2.5 %", "Estimate", "97.5 %"]``
        (percentages adjusted for level).
        """
        from glmmadaptive.utils.linalg import cov_to_chol, chol_to_cov
        alpha = 1.0 - level
        z = _norm.ppf(1.0 - alpha / 2.0)
        lo_pct = f"{100 * alpha / 2:.1f} %"
        hi_pct = f"{100 * (1 - alpha / 2):.1f} %"

        if parm == "fixed-effects":
            V_block = self.vcov(parm="fixed-effects", sandwich=sandwich)
            ses = np.sqrt(np.maximum(np.diag(V_block), 0.0))
            lo = self.params - z * ses
            hi = self.params + z * ses
            return pd.DataFrame(
                {lo_pct: lo, "Estimate": self.params, hi_pct: hi},
                index=self._beta_names,
            )

        if parm == "var-cov":
            diagonal_D = self.model.control.get("diagonal_D", False)
            D = self.D
            V_block = self.vcov(parm="var-cov", sandwich=sandwich)
            ses = np.sqrt(np.maximum(np.diag(V_block), 0.0))
            if diagonal_D or D.shape[0] == 1:
                # Unconstrained: log(diag(D)); CI → exponentiate
                unconstr = np.log(np.diag(D))
                lo_u = unconstr - z * ses
                hi_u = unconstr + z * ses
                lo_v = np.exp(lo_u)
                hi_v = np.exp(hi_u)
                est_v = np.exp(unconstr)
                n_re = D.shape[0]
                row_names = [f"var.b{j}" for j in range(n_re)]
            else:
                # Full D: Cholesky parameterisation
                unconstr = cov_to_chol(D, diagonal=False)
                lo_u = unconstr - z * ses
                hi_u = unconstr + z * ses
                # back-transform each endpoint
                n_re = D.shape[0]
                def _chol_diag(v):
                    M = chol_to_cov(v, q=n_re, diagonal=False)
                    idx = np.tril_indices(n_re)
                    return M[idx]
                lo_v = _chol_diag(lo_u)
                hi_v = _chol_diag(hi_u)
                est_v = _chol_diag(unconstr)
                idx = np.tril_indices(n_re)
                row_names = [
                    f"var.b{i}" if i == j else f"cov.b{j}_b{i}"
                    for i, j in zip(idx[0], idx[1])
                ]
            return pd.DataFrame(
                {lo_pct: lo_v, "Estimate": est_v, hi_pct: hi_v},
                index=row_names,
            )

        if parm == "extra":
            if self.phis is None:
                raise ValueError("Model has no extra (phis) parameters.")
            V_block = self.vcov(parm="extra", sandwich=sandwich)
            ses = np.sqrt(np.maximum(np.diag(V_block), 0.0))
            lo = self.phis - z * ses
            hi = self.phis + z * ses
            row_names = [f"phi_{i}" for i in range(len(self.phis))]
            return pd.DataFrame(
                {lo_pct: lo, "Estimate": self.phis, hi_pct: hi},
                index=row_names,
            )

        if parm == "zero_part":
            if self.gammas is None:
                raise ValueError("Model has no zero-part (gammas) parameters.")
            V_block = self.vcov(parm="zero_part", sandwich=sandwich)
            ses = np.sqrt(np.maximum(np.diag(V_block), 0.0))
            lo = self.gammas - z * ses
            hi = self.gammas + z * ses
            return pd.DataFrame(
                {lo_pct: lo, "Estimate": self.gammas, hi_pct: hi},
                index=self._gamma_names if self._gamma_names else
                      [f"gamma_{i}" for i in range(len(self.gammas))],
            )

        raise ValueError(
            f"Unknown parm='{parm}'. Choose from: "
            "'fixed-effects', 'var-cov', 'extra', 'zero_part'."
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
        **marginal_kwargs,
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
            ``"marginal"`` — population-averaged via ``marginal_coefs()``.
        **marginal_kwargs
            Passed to :meth:`marginal_coefs` when ``type_="marginal"``.

        Returns
        -------
        ndarray of shape (N,) — predicted means μ̂.
        """
        if type_ == "marginal":
            betas_marg = self.marginal_coefs(**marginal_kwargs)["betas"]
            if newdata is None:
                X = self.model._X
            else:
                import patsy
                _, X = patsy.dmatrices(
                    self.model.fixed_formula, newdata, return_type="matrix"
                )
                X = np.asarray(X)
            eta = X @ betas_marg
            return self.family.linkinv(eta)

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
    # Dynamic predictions  (mirrors predict.MixMod in R with newdata2)
    # ------------------------------------------------------------------

    def _get_subject_posterior(
        self, subject_data: pd.DataFrame
    ) -> tuple:
        """
        Find posterior mode and covariance of b_j for a new subject.

        Uses the stored parameter estimates (betas, D, phis, gammas) and
        minimises the negative log-posterior via L-BFGS-B, then returns
        the Hessian as the precision (inverse covariance) of the Laplace
        approximation.

        Returns
        -------
        mode : ndarray (q,)
        cov_b : ndarray (q, q)  — posterior covariance (Laplace approx)
        X_j, Z_j : design matrices for count part
        X_zi_j, Z_zi_j : design matrices for ZI part (or None)
        y_j : response vector
        """
        import patsy
        from glmmadaptive.utils.linalg import log_dmvnorm, nearPD
        from glmmadaptive.utils.quadrature import find_posterior_mode

        model = self.model
        betas = self.params
        D = self.D
        phis = self.phis
        gammas = self.gammas

        y_j, X_j = patsy.dmatrices(
            model.fixed_formula, subject_data, return_type="matrix"
        )
        y_j = np.asarray(y_j).ravel()
        X_j = np.asarray(X_j)

        Z_j = np.asarray(
            patsy.dmatrix(f"~ {model.re_rhs}", subject_data, return_type="matrix")
        )

        X_zi_j, Z_zi_j = None, None
        if model._has_zi and model.zi_fixed is not None:
            rhs_zi = model.zi_fixed.strip().lstrip("~").strip()
            X_zi_j = np.asarray(
                patsy.dmatrix(f"~ {rhs_zi}", subject_data, return_type="matrix")
            )
            if model._zi_re_rhs is not None:
                Z_zi_j = np.asarray(
                    patsy.dmatrix(
                        f"~ {model._zi_re_rhs}", subject_data, return_type="matrix"
                    )
                )

        D_inv = np.linalg.inv(D)
        _, log_det_D = np.linalg.slogdet(D)
        ncz = Z_j.shape[1]
        q = D.shape[0]

        def neg_log_post(b):
            eta = X_j @ betas + Z_j @ b[:ncz]
            eta_zi = None
            if X_zi_j is not None and gammas is not None:
                eta_zi = X_zi_j @ gammas
                if Z_zi_j is not None and Z_zi_j.shape[1] > 0:
                    eta_zi = eta_zi + Z_zi_j @ b[ncz:]
            ll = np.sum(model.family.log_dens(y_j, eta, phis=phis, eta_zi=eta_zi))
            lp = log_dmvnorm(b, cov_inv=D_inv, log_det_cov=log_det_D)
            return -(ll + lp)

        mode, neg_H = find_posterior_mode(neg_log_post, np.zeros(q))
        try:
            cov_b = np.linalg.inv(neg_H)
            cov_b = nearPD(0.5 * (cov_b + cov_b.T))
        except np.linalg.LinAlgError:
            cov_b = np.diag(1.0 / np.maximum(np.diag(neg_H), 1e-8))

        return mode, cov_b, X_j, Z_j, X_zi_j, Z_zi_j, y_j

    def predict_dynamic(
        self,
        newdata: pd.DataFrame,
        newdata2: Optional[pd.DataFrame] = None,
        se_fit: bool = True,
        return_newdata: bool = True,
        level: float = 0.95,
        n_mc: int = 200,
        seed: int = 42,
    ) -> dict:
        """
        Dynamic subject-specific predictions for new subjects.

        For each subject in *newdata* the random effects are estimated from
        their observed measurements.  Predictions are then computed for both
        the information period (*newdata*) and the future period (*newdata2*),
        with optional Monte Carlo confidence intervals.

        Mirrors ``predict.MixMod(newdata=…, newdata2=…, type='subject_specific',
        se.fit=TRUE)`` from the R package.

        Parameters
        ----------
        newdata : DataFrame
            Observed data used to estimate each subject's random effects
            (the "information period").
        newdata2 : DataFrame, optional
            Future time points at which to predict (the "prediction period").
            Must contain the same subjects and covariates as *newdata*.
        se_fit : bool
            If True, compute 95 % (or *level*) Monte Carlo confidence intervals
            by sampling from the posterior of the random effects.
        return_newdata : bool
            If True, return augmented copies of *newdata* (and *newdata2*) with
            columns ``pred``, ``low``, ``upp`` (and ``zi_probs``,
            ``zi_probs_low``, ``zi_probs_upp`` for ZI families).
        level : float
            Confidence level (default 0.95).
        n_mc : int
            Number of Monte Carlo draws (ignored when se_fit=False).
        seed : int
            Random seed.

        Returns
        -------
        dict with key ``"newdata"`` (and ``"newdata2"`` if provided), each
        being an augmented DataFrame.
        """
        import patsy
        from scipy.special import expit

        rng = np.random.default_rng(seed)
        alpha = 1.0 - level
        model = self.model
        id_col = model.id_name
        ncz = model._n_re_count

        # ---- 1. Estimate posterior for each new subject --------------------
        subject_ids = newdata[id_col].unique()
        subject_info: dict = {}
        for sid in subject_ids:
            subj_mask = newdata[id_col] == sid
            mode, cov_b, _, _, _, _, _ = self._get_subject_posterior(
                newdata[subj_mask].reset_index(drop=True)
            )
            subject_info[sid] = {"mode": mode, "cov_b": cov_b}

        # ---- 2. Helper: predict on a dataset for n_reps parameter draws ----
        def _predict_dataset(df: pd.DataFrame, n_reps: int) -> dict:
            """
            Returns dict with arrays pred_mc (n_reps, N), zi_mc (n_reps, N).
            """
            y_dm, X_all = patsy.dmatrices(
                model.fixed_formula, df, return_type="matrix"
            )
            X_all = np.asarray(X_all)
            Z_all = np.asarray(
                patsy.dmatrix(f"~ {model.re_rhs}", df, return_type="matrix")
            )

            X_zi_all, Z_zi_all = None, None
            if model._has_zi and model.zi_fixed is not None:
                rhs_zi = model.zi_fixed.strip().lstrip("~").strip()
                X_zi_all = np.asarray(
                    patsy.dmatrix(f"~ {rhs_zi}", df, return_type="matrix")
                )
                if model._zi_re_rhs is not None:
                    Z_zi_all = np.asarray(
                        patsy.dmatrix(
                            f"~ {model._zi_re_rhs}", df, return_type="matrix"
                        )
                    )

            N = len(df)
            ids_arr = df[id_col].values
            pred_mc = np.zeros((n_reps, N))
            zi_mc = np.zeros((n_reps, N)) if model._has_zi else None

            for m in range(n_reps):
                eta = X_all @ self.params
                eta_zi = (
                    X_zi_all @ self.gammas
                    if (X_zi_all is not None and self.gammas is not None)
                    else None
                )

                for sid in subject_ids:
                    mask = ids_arr == sid
                    if not np.any(mask):
                        continue
                    info = subject_info[sid]
                    if n_reps > 1:
                        b_j = rng.multivariate_normal(info["mode"], info["cov_b"])
                    else:
                        b_j = info["mode"]

                    eta[mask] += Z_all[mask] @ b_j[:ncz]
                    if eta_zi is not None and Z_zi_all is not None and Z_zi_all.shape[1] > 0:
                        eta_zi[mask] += Z_zi_all[mask] @ b_j[ncz:]

                mu = model.family.linkinv(eta)
                if eta_zi is not None:
                    pi_arr = expit(eta_zi)
                    if zi_mc is not None:
                        zi_mc[m] = pi_arr
                    mu = (1.0 - pi_arr) * mu

                pred_mc[m] = mu

            return {"pred_mc": pred_mc, "zi_mc": zi_mc}

        # ---- 3. Run predictions --------------------------------------------
        n_reps = n_mc if se_fit else 1
        res_nd = _predict_dataset(newdata, n_reps)
        res_nd2 = _predict_dataset(newdata2, n_reps) if newdata2 is not None else None

        # ---- 4. Summarise MC draws -----------------------------------------
        def _summarise(res, df):
            pm = res["pred_mc"]
            pred = pm.mean(axis=0)
            out = df.copy()
            out["pred"] = pred
            if se_fit and n_reps > 1:
                out["low"] = np.quantile(pm, alpha / 2.0, axis=0)
                out["upp"] = np.quantile(pm, 1.0 - alpha / 2.0, axis=0)
            else:
                out["low"] = pred
                out["upp"] = pred

            if res["zi_mc"] is not None:
                zm = res["zi_mc"]
                out["zi_probs"] = zm.mean(axis=0)
                if se_fit and n_reps > 1:
                    out["zi_probs_low"] = np.quantile(zm, alpha / 2.0, axis=0)
                    out["zi_probs_upp"] = np.quantile(zm, 1.0 - alpha / 2.0, axis=0)
            return out

        result: dict = {"newdata": _summarise(res_nd, newdata)}
        if res_nd2 is not None:
            result["newdata2"] = _summarise(res_nd2, newdata2)
        return result

    # ------------------------------------------------------------------
    # Fitted values and residuals
    # ------------------------------------------------------------------

    def fitted(self, type_: str = "mean_subject", **kwargs) -> NDArray:
        """Fitted values on the training data."""
        return self.predict(type_=type_, **kwargs)

    def residuals(self, type_: str = "mean_subject") -> NDArray:
        """Raw residuals: y - ŷ."""
        return self.model._y - self.fitted(type_=type_)

    # ------------------------------------------------------------------
    # simulate  (mirrors simulate.MixMod in R)
    # ------------------------------------------------------------------

    def simulate(
        self,
        nsim: int = 1,
        seed: Optional[int] = None,
        type_: str = "subject_specific",
        new_RE: bool = False,
        acount_MLEs_var: bool = False,
    ) -> pd.DataFrame:
        """
        Simulate response data from the fitted model.

        Parameters
        ----------
        nsim : int
            Number of simulated datasets to generate.
        seed : int, optional
            Random seed for reproducibility.
        type_ : str
            ``"subject_specific"`` (default) — uses empirical Bayes random
            effects; ``"mean_subject"`` — sets all random effects to zero.
        new_RE : bool
            If True, draw new random effects from N(0, D) instead of using
            the empirical Bayes estimates.
        acount_MLEs_var : bool
            If True, sample model parameters from MVN(θ̂, V) for each
            simulation, propagating MLE uncertainty.

        Returns
        -------
        DataFrame of shape (n_obs, nsim) with simulated responses.
        """
        rng = np.random.default_rng(seed)

        X = self.model._X                    # (N, p)
        Z = self.model._Z                    # (N, q)
        groups = self.model._groups          # (N,) integer group indices
        n_groups = self.model._n_groups
        n_re_count = self.model._n_re_count  # q for count part

        X_zi = self.model._X_zi              # (N, g) or None
        Z_zi = self.model._Z_zi              # (N, qz) or None

        betas_fit = self.params.copy()
        phis_fit = self.phis.copy() if self.phis is not None else None
        gammas_fit = self.gammas.copy() if self.gammas is not None else None
        D_fit = self.D.copy()

        if acount_MLEs_var:
            from glmmadaptive.utils.linalg import cov_to_chol, chol_to_cov
            V_full = np.linalg.inv(self._Hessian)

        # empirical Bayes b (n_groups, q_total)
        b_bayes = self._post_modes  # shape (n_groups, q_total)

        out = np.empty((len(self.model._y), nsim))

        for s in range(nsim):
            betas = betas_fit
            phis = phis_fit
            gammas = gammas_fit
            D = D_fit

            if acount_MLEs_var:
                # sample θ_new ~ MVN(θ̂, V)
                n_betas = len(betas_fit)
                n_re = self.D.shape[0]
                diagonal_D = self.model.control.get("diagonal_D", False)
                n_D = n_re if diagonal_D else n_re * (n_re + 1) // 2
                n_phis = len(phis_fit) if phis_fit is not None else 0
                n_gammas = len(gammas_fit) if gammas_fit is not None else 0
                n_total = n_betas + n_D + n_phis + n_gammas
                V_use = V_full[:n_total, :n_total]
                theta_hat = np.concatenate([
                    betas_fit,
                    cov_to_chol(D_fit, diagonal=diagonal_D),
                ] + ([phis_fit] if phis_fit is not None else [])
                  + ([gammas_fit] if gammas_fit is not None else []))
                theta_new = rng.multivariate_normal(theta_hat, V_use)
                betas = theta_new[:n_betas]
                idx = n_betas
                D_raw = theta_new[idx:idx + n_D]
                D = chol_to_cov(D_raw, q=n_re, diagonal=diagonal_D)
                idx += n_D
                if n_phis > 0:
                    phis = theta_new[idx:idx + n_phis]
                    idx += n_phis
                if n_gammas > 0:
                    gammas = theta_new[idx:idx + n_gammas]

            # draw random effects
            if new_RE:
                b_sim = rng.multivariate_normal(np.zeros(D.shape[0]),
                                                D, size=n_groups)
            else:
                b_sim = b_bayes  # (n_groups, q_total)

            if type_ == "mean_subject":
                b_sim = np.zeros_like(b_sim)

            # count-part linear predictor
            eta = X @ betas
            for i in range(n_groups):
                mask = groups == i
                eta[mask] += Z[mask] @ b_sim[i, :n_re_count]

            mu = self.family.linkinv(eta)

            # zero-inflation linear predictor
            eta_zi_arr = None
            if X_zi is not None and gammas is not None:
                eta_zi_arr = X_zi @ gammas
                if Z_zi is not None:
                    n_re_zi = Z_zi.shape[1]
                    for i in range(n_groups):
                        mask = groups == i
                        eta_zi_arr[mask] += Z_zi[mask] @ b_sim[i, n_re_count:n_re_count + n_re_zi]

            out[:, s] = self.family.simulate_response(mu, phis, eta_zi_arr, rng)

        return pd.DataFrame(out, columns=[f"sim_{s+1}" for s in range(nsim)])

    # ------------------------------------------------------------------
    # marginal_coefs  (mirrors marginal_coefs.MixMod in R)
    # ------------------------------------------------------------------

    def marginal_coefs(
        self,
        std_errors: bool = False,
        M: int = 3000,
        K: int = 100,
        seed: int = 1,
    ) -> dict:
        """
        Population-averaged (marginal) coefficients via Monte Carlo integration.

        Follows the Hedeker et al. (2017) approach: for each subject sample
        M random-effect vectors from N(0, D), compute the average inverse-link
        response, back-transform with the forward link, then solve for the
        marginal regression coefficients by OLS.

        Parameters
        ----------
        std_errors : bool
            If True, compute SEs by repeating with K perturbed parameter draws
            from MVN(θ̂, V).
        M : int
            Number of random-effect samples per subject per iteration (default 3000).
        K : int
            Number of parameter perturbation iterations for SEs (default 100).
        seed : int
            Random seed (default 1).

        Returns
        -------
        dict with keys:
            ``"betas"`` — ndarray of marginal coefficients,
            ``"coef_table"`` — DataFrame (if std_errors=True),
            ``"vcov"`` — ndarray (if std_errors=True).
        """
        from scipy.linalg import eigh

        X = self.model._X           # (N, p)
        Z = self.model._Z           # (N, q_count)
        groups = self.model._groups
        n_groups = self.model._n_groups
        n_re_count = self.model._n_re_count
        X_zi = self.model._X_zi
        Z_zi = self.model._Z_zi

        def _compute_marg_betas(betas, D, gammas, seed_offset):
            rng_mc = np.random.default_rng(seed + seed_offset)
            # Cholesky-based sampling from N(0, D)
            try:
                evals, evecs = eigh(D)
                sqrtD = evecs @ np.diag(np.sqrt(np.maximum(evals, 0.0)))
            except Exception:
                sqrtD = np.eye(D.shape[0])

            Xbetas = X @ betas
            if gammas is not None and X_zi is not None:
                eta_zi_fixed = X_zi @ gammas
            else:
                eta_zi_fixed = None

            marg_link = np.empty(len(Xbetas))
            id_ = groups  # integer group labels 0..n_groups-1
            n_re = D.shape[0]

            for i in range(n_groups):
                mask = id_ == i
                # sample M random effects: (n_re, M)
                b_mc = sqrtD @ rng_mc.standard_normal((n_re, M))
                # count-part linear predictor (n_obs_i x M)
                Zb = Z[mask, :] @ b_mc[:n_re_count, :]  # (n_obs_i, M)
                mu_mc = self.family.linkinv(Xbetas[mask, np.newaxis] + Zb)  # (n_obs_i, M)

                if eta_zi_fixed is not None:
                    eta_zi_i = eta_zi_fixed[mask]  # (n_obs_i,)
                    if Z_zi is not None:
                        n_re_zi = Z_zi.shape[1]
                        Zb_zi = Z_zi[mask, :] @ b_mc[n_re_count:n_re_count + n_re_zi, :]
                        from scipy.special import expit as _expit
                        pi_mc = _expit(eta_zi_i[:, np.newaxis] + Zb_zi)
                    else:
                        from scipy.special import expit as _expit
                        pi_mc = _expit(eta_zi_i[:, np.newaxis])
                    mu_mc = (1.0 - pi_mc) * mu_mc

                # average over M samples → marg_mu, then apply forward link
                marg_mu_i = mu_mc.mean(axis=1)  # (n_obs_i,)
                marg_link[mask] = self.family.link_fun(marg_mu_i)

            # OLS: β_marg = (X'X)^{-1} X' marg_link
            betas_marg, _, _, _ = np.linalg.lstsq(X, marg_link, rcond=None)
            return betas_marg

        betas_marg = _compute_marg_betas(self.params, self.D, self.gammas, 0)
        out = {"betas": betas_marg}

        if std_errors:
            from glmmadaptive.utils.linalg import cov_to_chol, chol_to_cov
            diagonal_D = self.model.control.get("diagonal_D", False)
            n_betas = len(self.params)
            n_re = self.D.shape[0]
            n_D = n_re if diagonal_D else n_re * (n_re + 1) // 2
            n_phis = len(self.phis) if self.phis is not None else 0
            n_gammas = len(self.gammas) if self.gammas is not None else 0
            n_total = n_betas + n_D + n_phis + n_gammas
            V = np.linalg.inv(self._Hessian)[:n_total, :n_total]
            theta_hat = np.concatenate([
                self.params,
                cov_to_chol(self.D, diagonal=diagonal_D),
            ] + ([self.phis] if self.phis is not None else [])
              + ([self.gammas] if self.gammas is not None else []))
            rng_se = np.random.default_rng(seed + 1000)
            mc_betas = np.empty((K, n_betas))
            for k in range(K):
                theta_k = rng_se.multivariate_normal(theta_hat, V)
                betas_k = theta_k[:n_betas]
                D_raw_k = theta_k[n_betas:n_betas + n_D]
                D_k = chol_to_cov(D_raw_k, q=n_re, diagonal=diagonal_D)
                idx = n_betas + n_D
                gammas_k = theta_k[idx:idx + n_gammas] if n_gammas > 0 else None
                mc_betas[k] = _compute_marg_betas(betas_k, D_k, gammas_k, k + 1)
            vcov_marg = np.cov(mc_betas.T)
            ses = np.sqrt(np.diag(vcov_marg))
            z_vals = betas_marg / np.where(ses > 0, ses, np.nan)
            from scipy.stats import norm as sp_norm_
            p_vals = 2.0 * sp_norm_.sf(np.abs(z_vals))
            coef_table = pd.DataFrame({
                "Estimate": betas_marg,
                "Std.Err": ses,
                "z-value": z_vals,
                "p-value": p_vals,
            }, index=self._beta_names)
            out["vcov"] = vcov_marg
            out["coef_table"] = coef_table

        return out

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

    def _zi_coef_table(self) -> Optional[pd.DataFrame]:
        r = self.results
        if r.gammas is None:
            return None
        n_p = len(r.params)
        n_g = len(r.gammas)
        try:
            vcov_all = np.linalg.inv(r._Hessian)
            vcov_g = vcov_all[n_p : n_p + n_g, n_p : n_p + n_g]
            se_g = np.sqrt(np.maximum(np.diag(vcov_g), 0.0))
        except Exception:
            se_g = np.full(n_g, np.nan)
        z = r.gammas / np.where(se_g > 0, se_g, np.nan)
        p = 2.0 * _norm.sf(np.abs(z))
        return pd.DataFrame(
            {"Estimate": r.gammas, "Std.Err": se_g, "z value": z, "Pr(>|z|)": p},
            index=r._gamma_names,
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
        lines.append("Fixed Effects (count part):")
        lines.append(self._coef_table().to_string(float_format="{:.4f}".format))
        zi_tbl = self._zi_coef_table()
        if zi_tbl is not None:
            lines.append("")
            lines.append("Fixed Effects (zero-inflation part):")
            lines.append(zi_tbl.to_string(float_format="{:.4f}".format))
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
