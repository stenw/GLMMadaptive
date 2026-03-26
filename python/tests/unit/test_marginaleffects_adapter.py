"""
Unit tests for the marginaleffects adapter (glmmadaptive.marginaleffects).

All tests use a small simulated binary longitudinal dataset so they run
quickly without R or any external dependencies beyond marginaleffects.
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared fixture: small binary longitudinal dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_fit():
    """Fit a simple binary mixed model for use across tests."""
    from glmmadaptive import MixedModel
    from glmmadaptive.families import Binomial

    rng = np.random.default_rng(0)
    n, K = 60, 4
    ids = np.repeat(np.arange(n), K)
    time = np.tile(np.arange(K), n).astype(float)
    sex = np.repeat(rng.choice(["male", "female"], n), K)
    b = rng.normal(0, 1, n)[ids]
    eta = -1.5 + 0.4 * time + 0.8 * (sex == "female") + b
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    df = pd.DataFrame({"id": ids, "y": y, "time": time, "sex": pd.Categorical(sex)})

    fit = MixedModel(
        "y ~ sex + time", random="~ 1 | id", data=df, family=Binomial()
    ).fit()
    return fit, df


@pytest.fixture(scope="module")
def wrapped(binary_fit):
    from glmmadaptive.marginaleffects import wrap_mixmod

    fit, _ = binary_fit
    return wrap_mixmod(fit)


# ---------------------------------------------------------------------------
# 1. Adapter construction
# ---------------------------------------------------------------------------


class TestAdapterConstruction:
    def test_wrap_returns_model_abstract(self, wrapped):
        from marginaleffects.classes.model import ModelAbstract

        assert isinstance(wrapped, ModelAbstract)

    def test_wrap_mixmod_convenience(self, binary_fit):
        from glmmadaptive.marginaleffects import wrap_mixmod

        fit, _ = binary_fit
        m = wrap_mixmod(fit)
        from marginaleffects.classes.model import ModelAbstract

        assert isinstance(m, ModelAbstract)

    def test_no_import_of_marginaleffects_from_core(self):
        """Core glmmadaptive package must not import marginaleffects."""
        import glmmadaptive

        assert not hasattr(glmmadaptive, "wrap_mixmod"), (
            "wrap_mixmod must not be exported from the top-level glmmadaptive package"
        )


# ---------------------------------------------------------------------------
# 2. get_coef and get_vcov
# ---------------------------------------------------------------------------


class TestCoefVcov:
    def test_get_coef_matches_fixef(self, wrapped, binary_fit):
        fit, _ = binary_fit
        np.testing.assert_array_equal(wrapped.get_coef(), fit.fixef().values)

    def test_get_coef_is_ndarray(self, wrapped):
        assert isinstance(wrapped.get_coef(), np.ndarray)

    def test_get_vcov_shape(self, wrapped, binary_fit):
        fit, _ = binary_fit
        n_p = len(fit.fixef())
        V = wrapped.get_vcov()
        assert V.shape == (n_p, n_p)

    def test_get_vcov_symmetric(self, wrapped):
        V = wrapped.get_vcov()
        np.testing.assert_allclose(V, V.T, atol=1e-12)

    def test_get_vcov_positive_diagonal(self, wrapped):
        V = wrapped.get_vcov()
        assert np.all(np.diag(V) > 0)

    def test_get_vcov_false_returns_none(self, wrapped):
        assert wrapped.get_vcov(vcov=False) is None

    def test_get_vcov_matches_model_vcov(self, wrapped, binary_fit):
        fit, _ = binary_fit
        np.testing.assert_allclose(wrapped.get_vcov(), fit.vcov(parm="fixed-effects"))


# ---------------------------------------------------------------------------
# 3. get_predict
# ---------------------------------------------------------------------------


class TestGetPredict:
    def test_get_predict_returns_polars(self, wrapped, binary_fit):
        import polars as pl

        fit, df = binary_fit
        import polars as pl
        from glmmadaptive.marginaleffects.adapter import _to_polars

        nd = _to_polars(df)
        result = wrapped.get_predict(wrapped.get_coef(), nd)
        assert isinstance(result, pl.DataFrame)

    def test_get_predict_columns(self, wrapped, binary_fit):
        from glmmadaptive.marginaleffects.adapter import _to_polars

        fit, df = binary_fit
        nd = _to_polars(df)
        result = wrapped.get_predict(wrapped.get_coef(), nd)
        assert "rowid" in result.columns
        assert "estimate" in result.columns

    def test_get_predict_length(self, wrapped, binary_fit):
        from glmmadaptive.marginaleffects.adapter import _to_polars

        fit, df = binary_fit
        nd = _to_polars(df)
        result = wrapped.get_predict(wrapped.get_coef(), nd)
        assert len(result) == len(df)

    def test_get_predict_binary_in_unit_interval(self, wrapped, binary_fit):
        from glmmadaptive.marginaleffects.adapter import _to_polars

        fit, df = binary_fit
        nd = _to_polars(df)
        est = wrapped.get_predict(wrapped.get_coef(), nd)["estimate"].to_numpy()
        assert np.all(est >= 0) and np.all(est <= 1)

    def test_get_predict_numpy_newdata(self, wrapped, binary_fit):
        """get_predict must accept a pre-built numpy design matrix."""
        import polars as pl
        from glmmadaptive.marginaleffects.adapter import _to_polars

        fit, df = binary_fit
        nd = _to_polars(df)
        X = wrapped.get_exog(nd)
        result = wrapped.get_predict(wrapped.get_coef(), X)
        assert len(result) == len(df)

    def test_get_predict_matches_model_predict(self, wrapped, binary_fit):
        """predictions with MLE params must match model.predict(type_='mean_subject')."""
        from glmmadaptive.marginaleffects.adapter import _to_polars

        fit, df = binary_fit
        nd = _to_polars(df)
        me_pred = wrapped.get_predict(wrapped.get_coef(), nd)["estimate"].to_numpy()
        model_pred = fit.predict(type_="mean_subject")
        np.testing.assert_allclose(me_pred, model_pred, rtol=1e-6)


# ---------------------------------------------------------------------------
# 4. predictions() API
# ---------------------------------------------------------------------------


class TestPredictionsAPI:
    def test_predictions_runs(self, wrapped):
        from marginaleffects import predictions

        out = predictions(wrapped)
        assert out is not None

    def test_predictions_shape(self, wrapped, binary_fit):
        from marginaleffects import predictions

        fit, df = binary_fit
        out = predictions(wrapped)
        assert out.shape[0] == len(df)

    def test_predictions_has_conf_intervals(self, wrapped):
        from marginaleffects import predictions

        out = predictions(wrapped)
        assert "conf_low" in out.columns
        assert "conf_high" in out.columns

    def test_predictions_estimate_in_unit_interval(self, wrapped):
        from marginaleffects import predictions

        out = predictions(wrapped)
        est = out["estimate"].to_numpy(allow_copy=True)
        assert np.all(est >= 0) and np.all(est <= 1)

    def test_predictions_on_datagrid(self, wrapped, binary_fit):
        from marginaleffects import predictions, datagrid

        fit, df = binary_fit
        grid = datagrid(model=wrapped, sex=["male", "female"], time=[0.0, 1.0])
        out = predictions(wrapped, newdata=grid)
        assert out.shape[0] == 4  # 2 sexes × 2 time points


# ---------------------------------------------------------------------------
# 5. comparisons() API
# ---------------------------------------------------------------------------


class TestComparisonsAPI:
    def test_comparisons_sex_pairwise(self, wrapped):
        from marginaleffects import comparisons

        out = comparisons(wrapped, variables={"sex": "pairwise"})
        assert out is not None
        assert out.shape[0] > 0

    def test_comparisons_by_time(self, wrapped, binary_fit):
        from marginaleffects import comparisons

        fit, df = binary_fit
        n_time = df["time"].nunique()
        out = comparisons(wrapped, variables={"sex": "pairwise"}, by="time")
        assert out.shape[0] == n_time

    def test_comparisons_estimate_is_finite(self, wrapped):
        from marginaleffects import comparisons

        out = comparisons(wrapped, variables={"sex": "pairwise"})
        est = out["estimate"].to_numpy(allow_copy=True)
        assert np.all(np.isfinite(est))

    def test_comparisons_p_value_in_unit_interval(self, wrapped):
        from marginaleffects import comparisons

        out = comparisons(wrapped, variables={"sex": "pairwise"})
        pvals = out["p_value"].to_numpy(allow_copy=True)
        assert np.all(pvals >= 0) and np.all(pvals <= 1)
