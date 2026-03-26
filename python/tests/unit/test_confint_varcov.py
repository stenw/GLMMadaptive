"""
Unit tests for confint(parm="var-cov") and vcov(parm="var-cov"/"extra"/"zero_part").
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from glmmadaptive.core.mixed_model import MixedModel
from glmmadaptive.families.standard import Binomial, NegativeBinomial
from glmmadaptive.families.zero_inflated import ZIPoisson


# ---------------------------------------------------------------------------
# Shared fitted models
# ---------------------------------------------------------------------------

def _binary_fit():
    rng = np.random.default_rng(42)
    n, K = 80, 6
    ids = np.repeat(np.arange(n), K)
    b = rng.normal(0, 1.0, n)[ids]
    x = rng.normal(0, 1, n * K)
    p = expit(-0.3 + 0.8 * x + b)
    y = rng.binomial(1, p)
    df = pd.DataFrame({"y": y, "x": x, "id": ids})
    m = MixedModel("y ~ x", "~ 1 | id", df, family=Binomial())
    return m.fit()


def _nb_fit():
    rng = np.random.default_rng(7)
    n, K = 60, 5
    ids = np.repeat(np.arange(n), K)
    b = rng.normal(0, 0.5, n)[ids]
    x = rng.normal(0, 1, n * K)
    mu = np.exp(0.5 + 0.4 * x + b)
    y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))
    df = pd.DataFrame({"y": y, "x": x, "id": ids})
    m = MixedModel("y ~ x", "~ 1 | id", df, family=NegativeBinomial())
    return m.fit()


def _zip_fit():
    rng = np.random.default_rng(3)
    n, K = 50, 5
    ids = np.repeat(np.arange(n), K)
    b = rng.normal(0, 0.5, n)[ids]
    x = rng.normal(0, 1, n * K)
    mu = np.exp(0.3 + 0.3 * x + b)
    lambda_ = rng.poisson(mu)
    zi = rng.binomial(1, 0.3, n * K).astype(bool)
    y = np.where(zi, 0, lambda_)
    df = pd.DataFrame({"y": y, "x": x, "id": ids})
    m = MixedModel("y ~ x", "~ 1 | id", df, family=ZIPoisson(), zi_fixed="~ 1")
    return m.fit()


_BIN = None
_NB = None
_ZIP = None


@pytest.fixture(scope="module")
def bin_fit():
    global _BIN
    if _BIN is None:
        _BIN = _binary_fit()
    return _BIN


@pytest.fixture(scope="module")
def nb_fit():
    global _NB
    if _NB is None:
        _NB = _nb_fit()
    return _NB


@pytest.fixture(scope="module")
def zip_fit():
    global _ZIP
    if _ZIP is None:
        _ZIP = _zip_fit()
    return _ZIP


# ---------------------------------------------------------------------------
# Tests: vcov parm values
# ---------------------------------------------------------------------------

class TestVcovParms:
    def test_varcov_shape_scalar_re(self, bin_fit):
        V = bin_fit.vcov(parm="var-cov")
        assert V.shape == (1, 1)

    def test_varcov_positive_diagonal(self, bin_fit):
        V = bin_fit.vcov(parm="var-cov")
        assert V[0, 0] > 0.0

    def test_fixef_shape(self, bin_fit):
        V = bin_fit.vcov(parm="fixed-effects")
        n_b = len(bin_fit.params)
        assert V.shape == (n_b, n_b)

    def test_extra_shape_nb(self, nb_fit):
        V = nb_fit.vcov(parm="extra")
        assert V.shape == (1, 1)

    def test_extra_raises_for_no_phis(self, bin_fit):
        with pytest.raises(ValueError, match="no.*extra"):
            bin_fit.vcov(parm="extra")

    def test_zero_part_shape(self, zip_fit):
        V = zip_fit.vcov(parm="zero_part")
        assert V.shape[0] >= 1

    def test_zero_part_raises_for_no_gammas(self, bin_fit):
        with pytest.raises(ValueError, match="no.*zero_part"):
            bin_fit.vcov(parm="zero_part")

    def test_unknown_parm_raises(self, bin_fit):
        with pytest.raises(ValueError, match="Unknown parm"):
            bin_fit.vcov(parm="bogus")


# ---------------------------------------------------------------------------
# Tests: confint parm="var-cov"
# ---------------------------------------------------------------------------

class TestConfintVarcov:
    def test_lower_positive(self, bin_fit):
        ci = bin_fit.confint(parm="var-cov")
        lo_col = ci.columns[0]
        assert (ci[lo_col].values > 0).all(), "Lower CI for variance should be positive"

    def test_estimate_equals_D_diag(self, bin_fit):
        ci = bin_fit.confint(parm="var-cov")
        est_col = "Estimate"
        np.testing.assert_allclose(
            ci[est_col].values, np.diag(bin_fit.D), rtol=1e-6
        )

    def test_lower_lt_upper(self, bin_fit):
        ci = bin_fit.confint(parm="var-cov")
        lo_col = ci.columns[0]
        hi_col = ci.columns[2]
        assert (ci[lo_col].values < ci[hi_col].values).all()

    def test_level_affects_width(self, bin_fit):
        ci_90 = bin_fit.confint(level=0.90, parm="var-cov")
        ci_95 = bin_fit.confint(level=0.95, parm="var-cov")
        lo90, hi90 = ci_90.columns[0], ci_90.columns[2]
        lo95, hi95 = ci_95.columns[0], ci_95.columns[2]
        width_90 = ci_90[hi90].values - ci_90[lo90].values
        width_95 = ci_95[hi95].values - ci_95[lo95].values
        assert (width_95 > width_90).all()

    def test_confint_extra_present_for_nb(self, nb_fit):
        ci = nb_fit.confint(parm="extra")
        assert ci.shape[0] == 1

    def test_confint_zero_part_present_for_zip(self, zip_fit):
        ci = zip_fit.confint(parm="zero_part")
        assert ci.shape[0] >= 1


# ---------------------------------------------------------------------------
# Tests: sandwich vcov
# ---------------------------------------------------------------------------

class TestSandwichVcov:
    def test_sandwich_shape(self, bin_fit):
        V = bin_fit.vcov(parm="fixed-effects", sandwich=True)
        n_b = len(bin_fit.params)
        assert V.shape == (n_b, n_b)

    def test_sandwich_psd(self, bin_fit):
        V = bin_fit.vcov(parm="fixed-effects", sandwich=True)
        evals = np.linalg.eigvalsh(V)
        assert (evals >= -1e-10).all(), "Sandwich vcov should be positive semi-definite"

    def test_sandwich_close_to_hessian_large_n(self, bin_fit):
        """For a fitted model, sandwich SEs should be in the same ballpark as Hessian SEs."""
        se_hess = bin_fit.bse
        se_sand = np.sqrt(np.diag(bin_fit.vcov(parm="fixed-effects", sandwich=True)))
        # Ratio should be between 0.2 and 5.0 (wide tolerance — sandwich can differ)
        ratio = se_sand / se_hess
        assert (ratio > 0.2).all() and (ratio < 5.0).all()
