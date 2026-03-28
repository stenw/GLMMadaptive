"""
Regression tests for grouped (aggregated) binomial responses.

Tests that ``cbind(successes, failures) ~ x`` syntax works end-to-end,
including fitting, predictions, residuals, and LRT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_grouped_binomial(n_subjects=80, n_time=5, seed=42):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_subjects), n_time)
    time = np.tile(np.arange(n_time), n_subjects)
    b = rng.normal(0, 0.5, n_subjects)
    eta = -0.5 + 0.3 * time + b[ids]
    mu = expit(eta)
    N = rng.integers(5, 20, size=len(ids))
    successes = rng.binomial(N, mu)
    failures = N - successes
    return pd.DataFrame({
        "successes": successes,
        "failures": failures,
        "time": time,
        "id": ids,
    })


def _fit_grouped(data, formula="cbind(successes, failures) ~ time", control=None):
    from glmmadaptive import MixedModel
    from glmmadaptive.families import Binomial
    ctrl = {"iter_em": 50, "verbose": False}
    if control:
        ctrl.update(control)
    return MixedModel(
        fixed=formula,
        random="~ 1 | id",
        data=data,
        family=Binomial(),
        control=ctrl,
    ).fit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.regression
class TestGroupedBinomialFit:

    @pytest.fixture(scope="class")
    def data(self):
        return _simulate_grouped_binomial()

    @pytest.fixture(scope="class")
    def result(self, data):
        return _fit_grouped(data)

    def test_fit_converges(self, result):
        assert result.converged

    def test_params_shape(self, result):
        """Intercept + time → 2 parameters."""
        assert len(result.params) == 2

    def test_params_finite(self, result):
        assert np.all(np.isfinite(result.params))

    def test_predict_is_probability(self, result):
        """Mean-subject predictions must lie strictly in (0, 1)."""
        preds = result.fitted(type_="mean_subject")
        assert np.all(preds > 0) and np.all(preds < 1)

    def test_subject_specific_is_probability(self, result):
        preds = result.fitted(type_="subject_specific")
        assert np.all(preds > 0) and np.all(preds < 1)

    def test_predict_newdata_no_response_columns(self, data, result):
        """predict(newdata) without the response columns must work."""
        newdata = data[["time", "id"]].head(20).copy()
        preds = result.predict(newdata=newdata, type_="mean_subject")
        assert preds.shape == (20,)
        assert np.all(preds > 0) and np.all(preds < 1)

    def test_predict_newdata_marginal_no_response(self, data, result):
        newdata = data[["time", "id"]].head(20).copy()
        preds = result.predict(newdata=newdata, type_="marginal")
        assert preds.shape == (20,)
        assert np.all(preds > 0) and np.all(preds < 1)

    def test_residuals_shape(self, data, result):
        assert result.residuals().shape == (len(data),)

    def test_residuals_near_zero_mean(self, result):
        """Mean proportion residual should be close to zero."""
        resid = result.residuals()
        assert abs(resid.mean()) < 0.05

    def test_loglik_finite(self, result):
        assert np.isfinite(result.logLik)

    def test_aic_bic_plausible(self, result):
        assert result.aic > 0
        assert result.bic > 0

    def test_anova_two_models(self, data):
        """LRT between intercept-only and time model should run without error."""
        from glmmadaptive.results import MixModResults
        fm1 = _fit_grouped(data, formula="cbind(successes, failures) ~ 1")
        fm2 = _fit_grouped(data, formula="cbind(successes, failures) ~ time")
        tbl = MixModResults.anova(fm1, fm2)
        assert tbl.shape[0] == 2
        # time model should fit at least as well
        assert fm2.logLik >= fm1.logLik - 1e-6

    def test_summary_runs(self, result):
        s = result.summary()
        assert s is not None

    def test_grouped_vs_expanded_params_close(self, data):
        """
        Fitting grouped cbind should give similar beta estimates to expanding
        the data to individual binary observations.
        """
        from glmmadaptive import MixedModel
        from glmmadaptive.families import Binomial

        # Expand to binary rows
        rows = []
        for _, row in data.iterrows():
            for _ in range(int(row["successes"])):
                rows.append({"y": 1, "time": row["time"], "id": row["id"]})
            for _ in range(int(row["failures"])):
                rows.append({"y": 0, "time": row["time"], "id": row["id"]})
        expanded = pd.DataFrame(rows)

        ctrl = {"iter_em": 50, "verbose": False}
        fm_grouped = _fit_grouped(data)
        fm_expanded = MixedModel(
            fixed="y ~ time",
            random="~ 1 | id",
            data=expanded,
            family=Binomial(),
            control=ctrl,
        ).fit()

        # Parameters should be close (within 10%)
        assert_allclose(fm_grouped.params, fm_expanded.params, rtol=0.10, atol=0.10)
