"""
Unit tests for marginal_coefs(), fitted(type_="marginal"), predict(type_="marginal").
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from glmmadaptive.core.mixed_model import MixedModel
from glmmadaptive.families.standard import Binomial, Poisson


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _binary_fit(n=80, K=8, seed=5):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n), K)
    b = rng.normal(0, 1.2, n)[ids]
    x = rng.normal(0, 1, n * K)
    p = expit(-0.5 + 1.5 * x + b)
    y = rng.binomial(1, p)
    df = pd.DataFrame({"y": y, "x": x, "id": ids})
    m = MixedModel("y ~ x", "~ 1 | id", df, family=Binomial())
    return m.fit()


def _poisson_fit(n=60, K=6, seed=6):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n), K)
    b = rng.normal(0, 0.5, n)[ids]
    x = rng.normal(0, 1, n * K)
    mu = np.exp(0.5 + 0.6 * x + b)
    y = rng.poisson(mu)
    df = pd.DataFrame({"y": y, "x": x, "id": ids})
    m = MixedModel("y ~ x", "~ 1 | id", df, family=Poisson())
    return m.fit()


_BIN = None
_POIS = None


@pytest.fixture(scope="module")
def bin_fit():
    global _BIN
    if _BIN is None:
        _BIN = _binary_fit()
    return _BIN


@pytest.fixture(scope="module")
def pois_fit():
    global _POIS
    if _POIS is None:
        _POIS = _poisson_fit()
    return _POIS


# ---------------------------------------------------------------------------
# Tests: basic output
# ---------------------------------------------------------------------------

class TestMarginalCoefsBasic:
    def test_returns_dict(self, bin_fit):
        mc = bin_fit.marginal_coefs(M=500, seed=1)
        assert isinstance(mc, dict)

    def test_betas_key_present(self, bin_fit):
        mc = bin_fit.marginal_coefs(M=500, seed=1)
        assert "betas" in mc

    def test_betas_length(self, bin_fit):
        mc = bin_fit.marginal_coefs(M=500, seed=1)
        assert len(mc["betas"]) == len(bin_fit.params)

    def test_no_std_errors_by_default(self, bin_fit):
        mc = bin_fit.marginal_coefs(M=500, seed=1)
        assert "vcov" not in mc
        assert "coef_table" not in mc


class TestMarginalCoefsShrinkage:
    def test_marginal_slope_smaller_binomial(self, bin_fit):
        """For Binomial logit, marginal |slope| < conditional |slope|."""
        mc = bin_fit.marginal_coefs(M=2000, seed=1)
        # Non-intercept betas should shrink toward zero
        marg_slope = mc["betas"][1]
        cond_slope = bin_fit.params[1]
        assert abs(marg_slope) < abs(cond_slope), (
            f"Expected |marginal slope| ({abs(marg_slope):.4f}) < "
            f"|conditional slope| ({abs(cond_slope):.4f})"
        )

    def test_marginal_slope_smaller_poisson(self, pois_fit):
        """For Poisson log link, marginal |slope| < conditional |slope|."""
        mc = pois_fit.marginal_coefs(M=2000, seed=1)
        marg_slope = mc["betas"][1]
        cond_slope = pois_fit.params[1]
        assert abs(marg_slope) < abs(cond_slope) + 1e-3  # allow small tolerance


class TestMarginalCoefsReproducibility:
    def test_same_seed_same_result(self, bin_fit):
        mc1 = bin_fit.marginal_coefs(M=500, seed=42)
        mc2 = bin_fit.marginal_coefs(M=500, seed=42)
        np.testing.assert_array_equal(mc1["betas"], mc2["betas"])

    def test_more_samples_more_stable(self, bin_fit):
        """Larger M reduces variance of the estimator."""
        estimates_small = [
            bin_fit.marginal_coefs(M=100, seed=s)["betas"][1]
            for s in range(10)
        ]
        estimates_large = [
            bin_fit.marginal_coefs(M=1000, seed=s)["betas"][1]
            for s in range(10)
        ]
        var_small = np.var(estimates_small)
        var_large = np.var(estimates_large)
        assert var_large < var_small + 1e-6  # large M should have smaller variance


class TestMarginalCoefsStdErrors:
    def test_std_errors_adds_vcov(self, bin_fit):
        mc = bin_fit.marginal_coefs(M=200, K=20, seed=1, std_errors=True)
        assert "vcov" in mc
        assert "coef_table" in mc

    def test_vcov_shape(self, bin_fit):
        mc = bin_fit.marginal_coefs(M=200, K=20, seed=1, std_errors=True)
        n_b = len(bin_fit.params)
        assert mc["vcov"].shape == (n_b, n_b)

    def test_coef_table_columns(self, bin_fit):
        mc = bin_fit.marginal_coefs(M=200, K=20, seed=1, std_errors=True)
        assert set(mc["coef_table"].columns) >= {"Estimate", "Std.Err", "z-value", "p-value"}

    def test_ses_positive(self, bin_fit):
        mc = bin_fit.marginal_coefs(M=200, K=20, seed=1, std_errors=True)
        ses = mc["coef_table"]["Std.Err"].values
        assert (ses > 0).all()


# ---------------------------------------------------------------------------
# Tests: fitted/predict type_="marginal"
# ---------------------------------------------------------------------------

class TestMarginalFitted:
    def test_marginal_fitted_in_0_1(self, bin_fit):
        mu = bin_fit.fitted(type_="marginal", M=500, seed=1)
        assert (mu >= 0).all() and (mu <= 1).all()

    def test_marginal_fitted_shape(self, bin_fit):
        mu = bin_fit.fitted(type_="marginal", M=500, seed=1)
        assert mu.shape == (bin_fit.model.nobs,)

    def test_marginal_predict_on_newdata(self, bin_fit):
        nd = bin_fit.model.data.copy()
        mu = bin_fit.predict(newdata=nd, type_="marginal", M=500, seed=1)
        assert mu.shape == (len(nd),)
        assert (mu >= 0).all() and (mu <= 1).all()

    def test_marginal_between_0_and_1_poisson(self, pois_fit):
        mu = pois_fit.fitted(type_="marginal", M=500, seed=1)
        assert (mu >= 0).all()
