"""
Regression tests: Gaussian random-intercept model vs pre-computed R fixture.

The R fixture (gaussian_ri.json) was generated with ``students.t(df=1e6)``
because GLMMadaptive R rejects ``gaussian()`` and redirects to lme4/nlme.
At df=1e6, students.t is numerically identical to Gaussian, so the
Python Gaussian family and the R students.t(df=1e6) results should match
to within optimizer noise.

Fixture path: python/tests/fixtures/gaussian_ri.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from glmmadaptive import MixedModel
from glmmadaptive.families import Gaussian
from glmmadaptive.results import MixModResults

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def fixture():
    with open(FIXTURES / "gaussian_ri.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fitted(fixture):
    df = pd.DataFrame(fixture["data"])
    model = MixedModel(
        fixed="y ~ time",
        random="~ 1 | id",
        data=df,
        family=Gaussian(),
        control={"iter_em": 50, "verbose": False},
    )
    return model.fit()


class TestGaussianRIvsR:
    def test_converged(self, fitted):
        assert fitted.converged

    def test_n_obs(self, fixture, fitted):
        n = len(np.asarray(fixture["data"]["y"]).ravel())
        assert len(fitted.fitted()) == n

    def test_betas_close(self, fixture, fitted):
        r_betas = np.asarray(fixture["betas"])
        assert_allclose(fitted.fixef().values, r_betas, rtol=0.05,
                        err_msg="Fixed-effect coefficients differ from R")

    def test_sigma_close(self, fixture, fitted):
        """Residual SD: sigma = exp(phis[0])."""
        r_sigma = np.exp(np.asarray(fixture["phis"])[0])
        py_sigma = np.exp(fitted.phis[0])
        assert_allclose(py_sigma, r_sigma, rtol=0.10,
                        err_msg="Residual SD sigma differs from R")

    def test_D_close(self, fixture, fitted):
        """Random-intercept variance D[0,0]."""
        # R fixture stores D columns by name "(Intercept)"
        r_D11 = float(fixture["D"]["(Intercept)"][0])
        py_D11 = fitted.D[0, 0]
        assert_allclose(py_D11, r_D11, rtol=0.10,
                        err_msg="Random-effect variance D[0,0] differs from R")

    def test_loglik_close(self, fixture, fitted):
        r_ll = float(np.asarray(fixture["logLik"]).ravel()[0])
        assert_allclose(fitted.logLik, r_ll, atol=1.0,
                        err_msg="Log-likelihood differs from R")

    def test_bse_positive(self, fitted):
        assert np.all(fitted.bse > 0)

    @pytest.mark.regression
    def test_intercept_close(self, fixture, fitted):
        assert_allclose(fitted.fixef().values[0], fixture["betas"][0], rtol=0.05)

    @pytest.mark.regression
    def test_time_slope_close(self, fixture, fitted):
        assert_allclose(fitted.fixef().values[1], fixture["betas"][1], rtol=0.05)


class TestGaussianRISummary:
    def test_summary_runs(self, fitted):
        s = fitted.summary()
        assert s is not None

    def test_anova_two_models(self, fixture):
        """LRT comparing intercept-only vs time model; time should improve fit."""
        df = pd.DataFrame(fixture["data"])
        m0 = MixedModel(
            fixed="y ~ 1", random="~ 1 | id",
            data=df, family=Gaussian(),
            control={"iter_em": 50, "verbose": False},
        ).fit()
        m1 = MixedModel(
            fixed="y ~ time", random="~ 1 | id",
            data=df, family=Gaussian(),
            control={"iter_em": 50, "verbose": False},
        ).fit()
        result = MixModResults.anova(m0, m1)
        assert result is not None
        assert m1.logLik > m0.logLik


class TestGaussianRIPredict:
    def test_fitted_values_shape(self, fixture, fitted):
        n = len(np.asarray(fixture["data"]["y"]).ravel())
        assert fitted.fitted().shape == (n,)

    def test_residuals_finite(self, fitted):
        resid = fitted.residuals()
        assert np.all(np.isfinite(resid))

    def test_predict_mean_subject(self, fixture, fitted):
        df = pd.DataFrame(fixture["data"])
        preds = fitted.predict(newdata=df, type_="mean_subject")
        assert preds.shape == (len(df),)
        assert np.all(np.isfinite(preds))
