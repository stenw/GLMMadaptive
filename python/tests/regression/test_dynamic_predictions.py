"""
Regression tests for predict(), fitted(), residuals(), and ranef().

Covers:
* Internal consistency (fitted = predict on training data, residuals = y - fitted)
* Shape and type contracts
* Comparison of Python predictions and random effects to pre-saved R outputs
  stored in ``tests/fixtures/binary_ri_predictions.json`` and
  ``tests/fixtures/poisson_ri_predictions.json``.

To regenerate fixtures::

    Rscript tests/fixtures/generate_r_fixtures.R
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_fixture(name: str) -> dict:
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"Fixture '{name}.json' not found — run generate_r_fixtures.R")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_binary(data, control=None):
    from glmmadaptive import MixedModel
    from glmmadaptive.families import Binomial
    ctrl = {"iter_em": 50, "verbose": False}
    if control:
        ctrl.update(control)
    return MixedModel(
        fixed="y ~ time",
        random="~ 1 | id",
        data=data,
        family=Binomial(),
        control=ctrl,
    ).fit()


def _fit_poisson(data, control=None):
    from glmmadaptive import MixedModel
    from glmmadaptive.families import Poisson
    ctrl = {"iter_em": 50, "verbose": False}
    if control:
        ctrl.update(control)
    return MixedModel(
        fixed="y ~ x",
        random="~ 1 | id",
        data=data,
        family=Poisson(),
        control=ctrl,
    ).fit()


# ---------------------------------------------------------------------------
# Internal consistency tests  (no R reference needed)
# ---------------------------------------------------------------------------

@pytest.mark.regression
class TestPredictInternalConsistency:
    """
    These tests check invariants that must hold regardless of R comparison:
    fitted() == predict() on training data, residuals() == y - fitted(), etc.
    """

    @pytest.fixture(scope="class")
    def result(self, sim_binary_data):
        return _fit_binary(sim_binary_data)

    def test_fitted_equals_predict_default(self, result):
        """fitted() with default type must equal predict() with no newdata."""
        assert_allclose(result.fitted(), result.predict(), atol=1e-12)

    def test_fitted_mean_subject_shape(self, result):
        assert result.fitted(type_="mean_subject").shape == (result.model.nobs,)

    def test_fitted_subject_specific_shape(self, result):
        assert result.fitted(type_="subject_specific").shape == (result.model.nobs,)

    def test_mean_subject_differs_from_subject_specific(self, result):
        """Population-average and subject-specific predictions should differ."""
        ms = result.fitted(type_="mean_subject")
        ss = result.fitted(type_="subject_specific")
        assert not np.allclose(ms, ss), (
            "mean_subject and subject_specific predictions are identical — "
            "random effects may not be applied correctly."
        )

    def test_residuals_formula(self, result):
        """residuals() must equal y - fitted()."""
        y = result.model.endog
        resid = result.residuals()
        assert_allclose(resid, y - result.fitted(), atol=1e-12)

    def test_residuals_shape(self, result):
        assert result.residuals().shape == (result.model.nobs,)

    def test_mean_subject_range(self, result):
        """For a Binomial family, predicted probabilities must be in (0, 1)."""
        preds = result.fitted(type_="mean_subject")
        assert np.all(preds > 0) and np.all(preds < 1)

    def test_subject_specific_range(self, result):
        preds = result.fitted(type_="subject_specific")
        assert np.all(preds > 0) and np.all(preds < 1)


@pytest.mark.regression
class TestRanefInternalConsistency:

    @pytest.fixture(scope="class")
    def result(self, sim_binary_data):
        return _fit_binary(sim_binary_data)

    def test_ranef_shape(self, result):
        """ranef() must return (n_groups, q) DataFrame."""
        re = result.ranef()
        n_groups = result.model.n_groups
        q = result.D.shape[0]
        assert re.shape == (n_groups, q)

    def test_ranef_index_matches_group_labels(self, result):
        re = result.ranef()
        expected_labels = result.model._group_labels
        assert list(re.index) == list(expected_labels)

    def test_ranef_mean_near_zero(self, result):
        """Empirical Bayes random effects should be roughly centred at zero."""
        re_mean = result.ranef().values.mean()
        assert abs(re_mean) < 0.5, (
            f"Mean random effect {re_mean:.3f} is unexpectedly large."
        )

    def test_ranef_columns_named(self, result):
        re = result.ranef()
        assert all(col.startswith("b") for col in re.columns)


@pytest.mark.regression
class TestPredictNewdata:
    """
    Tests for predict(newdata=...) — predictions for new data using the
    fitted model.
    """

    @pytest.fixture(scope="class")
    def result(self, sim_binary_data):
        return _fit_binary(sim_binary_data)

    @pytest.fixture(scope="class")
    def newdata_existing_groups(self, sim_binary_data):
        """Subset: first 5 subjects, first 2 time points each."""
        sub = sim_binary_data[sim_binary_data["id"] < 5].copy()
        return sub[sub["time"] < 2]

    def test_predict_newdata_mean_subject_shape(self, result, newdata_existing_groups):
        preds = result.predict(newdata=newdata_existing_groups, type_="mean_subject")
        assert preds.shape == (len(newdata_existing_groups),)

    def test_predict_newdata_mean_subject_range(self, result, newdata_existing_groups):
        preds = result.predict(newdata=newdata_existing_groups, type_="mean_subject")
        assert np.all(preds > 0) and np.all(preds < 1)

    def test_predict_newdata_subject_specific_shape(self, result, newdata_existing_groups):
        preds = result.predict(newdata=newdata_existing_groups, type_="subject_specific")
        assert preds.shape == (len(newdata_existing_groups),)

    def test_predict_newdata_mean_matches_formula(self, result, newdata_existing_groups):
        """
        mean_subject predictions on new data = linkinv(X_new @ betas),
        regardless of random effects.
        """
        import patsy
        _, X_new = patsy.dmatrices(
            result.model.fixed_formula, newdata_existing_groups, return_type="matrix"
        )
        X_new = np.asarray(X_new)
        expected = result.family.linkinv(X_new @ result.params)
        actual = result.predict(newdata=newdata_existing_groups, type_="mean_subject")
        assert_allclose(actual, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# R comparison tests (require pre-saved fixtures)
# ---------------------------------------------------------------------------

@pytest.mark.regression
class TestBinaryPredictionsVsR:
    """
    Compare Python predict() and ranef() to R's outputs pre-saved in
    ``binary_ri_predictions.json``.
    """

    @pytest.fixture(scope="class")
    def ref(self):
        return load_fixture("binary_ri_predictions")

    @pytest.fixture(scope="class")
    def result(self, ref):
        data = pd.DataFrame(ref["data"])
        return _fit_binary(data)

    def test_predict_mean_subject_close_to_r(self, result, ref):
        """Population-average predictions should match R within 5%."""
        r_preds = np.array(ref["predictions_mean_subject"])
        py_preds = result.predict(type_="mean_subject")
        assert_allclose(py_preds, r_preds, rtol=0.05, atol=0.02)

    def test_predict_subject_specific_close_to_r(self, result, ref):
        """Subject-specific predictions should match R within 10%."""
        r_preds = np.array(ref["predictions_subject_specific"])
        py_preds = result.predict(type_="subject_specific")
        assert_allclose(py_preds, r_preds, rtol=0.10, atol=0.05)

    def test_ranef_close_to_r(self, result, ref):
        """Random effects (posterior modes) should match R within 15%."""
        r_ranef = np.array(ref["ranef"])          # shape (n_groups, 1)
        py_ranef = result.ranef().values           # shape (n_groups, q)
        assert_allclose(py_ranef.ravel(), r_ranef.ravel(), rtol=0.15, atol=0.10)

    def test_loglik_close_to_r(self, result, ref):
        assert_allclose(result.logLik, ref["logLik"], atol=0.5)


@pytest.mark.regression
class TestPoissonPredictionsVsR:
    """
    Compare Python predict() and ranef() to R's outputs pre-saved in
    ``poisson_ri_predictions.json``.
    """

    @pytest.fixture(scope="class")
    def ref(self):
        return load_fixture("poisson_ri_predictions")

    @pytest.fixture(scope="class")
    def result(self, ref):
        data = pd.DataFrame(ref["data"])
        return _fit_poisson(data)

    def test_predict_mean_subject_close_to_r(self, result, ref):
        r_preds = np.array(ref["predictions_mean_subject"])
        py_preds = result.predict(type_="mean_subject")
        assert_allclose(py_preds, r_preds, rtol=0.05, atol=0.05)

    def test_predict_subject_specific_close_to_r(self, result, ref):
        r_preds = np.array(ref["predictions_subject_specific"])
        py_preds = result.predict(type_="subject_specific")
        assert_allclose(py_preds, r_preds, rtol=0.10, atol=0.10)

    def test_ranef_close_to_r(self, result, ref):
        r_ranef = np.array(ref["ranef"])
        py_ranef = result.ranef().values
        assert_allclose(py_ranef.ravel(), r_ranef.ravel(), rtol=0.15, atol=0.10)

    def test_loglik_close_to_r(self, result, ref):
        assert_allclose(result.logLik, ref["logLik"], atol=0.5)
