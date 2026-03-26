"""
marginaleffects adapter for GLMMadaptive.

Wraps a fitted :class:`~glmmadaptive.results.MixModResults` object in a
``marginaleffects``-compatible model class, enabling ``predictions()``,
``comparisons()``, ``slopes()``, and ``hypotheses()`` from the
``marginaleffects`` Python package.

Predictions are **mean-subject** (population-level, fixed effects only).
Subject-specific predictions are not supported — they require knowing which
group each new row belongs to, which is incompatible with the delta-method
uncertainty propagation used by ``marginaleffects``.

Example
-------
>>> from glmmadaptive import MixedModel
>>> from glmmadaptive.families import Binomial
>>> from glmmadaptive.marginaleffects import wrap_mixmod
>>> from marginaleffects import predictions, comparisons, datagrid
>>>
>>> fit = MixedModel("y ~ sex + time", random="~ 1 | id",
...                  data=df, family=Binomial()).fit()
>>> mfit = wrap_mixmod(fit)
>>>
>>> # Estimated marginal means on a sex × time grid
>>> grid = datagrid(model=mfit, sex=["male", "female"], time=[1, 2, 3])
>>> predictions(mfit, newdata=grid)
>>>
>>> # Pairwise sex comparison within each time point
>>> comparisons(mfit, variables={"sex": "pairwise"}, by="time")
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import patsy

from marginaleffects.classes.model import ModelAbstract, ModelVault


def _to_polars(df) -> pl.DataFrame:
    """Convert a pandas DataFrame to polars, casting str columns to Categorical."""
    pf = pl.from_pandas(df.reset_index(drop=True))
    str_cols = [c for c, t in zip(pf.columns, pf.dtypes) if t == pl.Utf8]
    if str_cols:
        pf = pf.with_columns([pl.col(c).cast(pl.Categorical) for c in str_cols])
    return pf


class ModelGLMMadaptive(ModelAbstract):
    """
    ``marginaleffects`` adapter for a fitted ``MixModResults`` object.

    Parameters
    ----------
    fitted : MixModResults
        A fitted GLMMadaptive model (returned by ``MixedModel.fit()``).
    vcov : str or None
        Which variance-covariance to expose.  ``"fixed-effects"`` (default)
        uses the inverse-Hessian block for the fixed effects.
        ``"sandwich"`` uses the robust sandwich estimator (requires the
        model to have been fitted with ``score_contributions`` stored).
    """

    def __init__(self, fitted, vcov: str = "fixed-effects"):
        self._fitted = fitted
        self._family = fitted.family
        self._beta_names = fitted._beta_names

        # Build patsy DesignInfo once from training data so that factor
        # levels and encodings are identical for any newdata passed later.
        formula_rhs = fitted.model.fixed_formula.split("~", 1)[1].strip()
        training_data = fitted.model.data
        dm = patsy.dmatrix(formula_rhs, training_data, return_type="matrix")
        self._design_info = dm.design_info

        # Store categorical column levels from training data so we can
        # restore them in get_exog (marginaleffects may pass DataFrames
        # with only a subset of values, losing the full level set).
        self._cat_levels: dict[str, list] = {}
        for col in training_data.columns:
            if hasattr(training_data[col], "cat"):
                self._cat_levels[col] = list(training_data[col].cat.categories)

        betas = fitted.fixef().values.astype(float)
        V = fitted.vcov(parm="fixed-effects",
                        sandwich=(vcov == "sandwich")).astype(float)

        # modeldata as polars DataFrame (marginaleffects works with polars).
        # String columns must be Categorical — marginaleffects rejects plain String.
        modeldata = _to_polars(fitted.model.data)

        vault = ModelVault(
            coef=betas,
            coefnames=np.asarray(self._beta_names),
            formula=fitted.model.fixed_formula,
            formula_engine="patsy",
            design_info_patsy=self._design_info,
            modeldata=modeldata,
            vcov=V,
            package="glmmadaptive",
        )
        super().__init__(fitted, vault)

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    def get_coef(self) -> np.ndarray:
        return self.vault.coef

    def get_vcov(self, vcov=True) -> Optional[np.ndarray]:
        if isinstance(vcov, bool):
            return self.vault.vcov if vcov else None
        raise ValueError(
            "`vcov` must be True or False for GLMMadaptive models. "
            "To use the sandwich estimator, pass vcov='sandwich' to wrap_mixmod()."
        )

    def get_exog(self, newdata: pl.DataFrame) -> np.ndarray:
        """Build the fixed-effects design matrix using the stored DesignInfo.

        Preserves factor levels and encodings from the training data.
        Categorical columns are re-encoded with the training-data levels so
        that patsy produces the same dummy coding even when only a subset of
        levels appears in ``newdata``.
        """
        df = newdata.to_pandas()
        # Restore categorical dtypes with training-data levels so patsy
        # sees the same encoding as during model fitting.
        for col, levels in self._cat_levels.items():
            if col in df.columns:
                df[col] = pd.Categorical(df[col], categories=levels)
        return np.asarray(
            patsy.build_design_matrices([self._design_info], df, return_type="matrix")[0]
        )

    def get_predict(self, params: np.ndarray, newdata) -> pl.DataFrame:
        """Mean-subject predictions: μ = linkinv(X @ params).

        Parameters
        ----------
        params : ndarray of shape (n_betas,)
            Coefficient vector — possibly perturbed by the delta method.
        newdata : polars DataFrame or ndarray
            New covariates.  When called internally by ``marginaleffects``
            during the delta-method loop, this may already be a numpy design
            matrix (pre-computed by ``get_exog``).

        Returns
        -------
        polars DataFrame with columns ``{"rowid": int32, "estimate": float64}``
        """
        if isinstance(newdata, np.ndarray):
            X = newdata
        else:
            X = self.get_exog(newdata)
        eta = X @ params
        mu = self._family.linkinv(eta)
        return pl.DataFrame(
            {"rowid": list(range(len(mu))), "estimate": list(mu)}
        ).with_columns(pl.col("rowid").cast(pl.Int32))


def wrap_mixmod(fitted, vcov: str = "fixed-effects") -> ModelGLMMadaptive:
    """Wrap a ``MixModResults`` object for use with ``marginaleffects``.

    Parameters
    ----------
    fitted : MixModResults
        A fitted GLMMadaptive model.
    vcov : str
        ``"fixed-effects"`` (default) or ``"sandwich"``.

    Returns
    -------
    ModelGLMMadaptive
        A ``marginaleffects``-compatible model object.

    Example
    -------
    >>> mfit = wrap_mixmod(fit)
    >>> from marginaleffects import predictions, comparisons
    >>> predictions(mfit)
    >>> comparisons(mfit, variables={"time": "pairwise"})
    """
    return ModelGLMMadaptive(fitted, vcov=vcov)
