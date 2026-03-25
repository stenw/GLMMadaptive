"""
Unit tests for zero-inflated and hurdle family objects.

Implementation status
---------------------
* ZIPoisson           — **fully implemented**
* ZINegativeBinomial  — **fully implemented**
* ZIBinomial          — stub (log_dens raises NotImplementedError)
* HurdlePoisson       — stub
* HurdleNegativeBinomial — stub
* HurdleBeta          — stub
* HurdleLogNormal     — stub

Tests for implemented families verify numerical correctness against analytic
results and scipy reference implementations.  Tests for stubs check attribute
metadata and that NotImplementedError is raised correctly.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit, gammaln
from scipy.stats import poisson, nbinom

from glmmadaptive.families.zero_inflated import ZIPoisson, ZINegativeBinomial, ZIBinomial
from glmmadaptive.families.hurdle import (
    HurdlePoisson,
    HurdleNegativeBinomial,
    HurdleBeta,
    HurdleLogNormal,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

rng = np.random.default_rng(0)
N = 20
ETA     = rng.normal(0, 1, N)          # linear predictor (count part)
ETA_ZI  = rng.normal(-0.5, 0.5, N)    # linear predictor (zero part)
Y_COUNT = rng.poisson(2, N).astype(float)


# ---------------------------------------------------------------------------
# Helper: ZIP log-density computed manually
# ---------------------------------------------------------------------------

def _zip_log_dens_manual(y, eta, eta_zi):
    mu = np.exp(eta)
    pi = expit(eta_zi)
    log_pois = poisson.logpmf(y.astype(int), mu)
    log_p0_pois = poisson.logpmf(0, mu)
    return np.where(
        y == 0,
        np.log(pi + (1 - pi) * np.exp(log_p0_pois)),
        np.log(1 - pi) + log_pois,
    )


# ---------------------------------------------------------------------------
# ZIPoisson — fully implemented
# ---------------------------------------------------------------------------

class TestZIPoisson:

    @pytest.fixture
    def fam(self):
        return ZIPoisson()

    def test_attributes(self, fam):
        assert fam.has_zi is True
        assert fam.family == "zi_poisson"
        assert fam.link == "log"
        assert fam.n_phis == 0

    def test_linkinv(self, fam):
        assert_allclose(fam.linkinv(ETA), np.exp(ETA), atol=1e-12)

    def test_log_dens_positive_y(self, fam):
        """For y > 0: log p = log(1-π) + log Poisson(y; μ)."""
        y_pos = np.maximum(Y_COUNT, 1.0)  # force all positive
        ref = _zip_log_dens_manual(y_pos, ETA, ETA_ZI)
        val = fam.log_dens(y_pos, ETA, eta_zi=ETA_ZI)
        assert_allclose(val, ref, atol=1e-8)

    def test_log_dens_zeros(self, fam):
        """For y = 0: log p = log(π + (1-π)·exp(-μ))."""
        y_zero = np.zeros(N)
        ref = _zip_log_dens_manual(y_zero, ETA, ETA_ZI)
        val = fam.log_dens(y_zero, ETA, eta_zi=ETA_ZI)
        assert_allclose(val, ref, atol=1e-8)

    def test_log_dens_mixed(self, fam):
        """Mixed y: compare to manual calculation."""
        ref = _zip_log_dens_manual(Y_COUNT, ETA, ETA_ZI)
        val = fam.log_dens(Y_COUNT, ETA, eta_zi=ETA_ZI)
        assert_allclose(val, ref, atol=1e-8)

    def test_log_dens_all_leq_zero(self, fam):
        """log-density must be ≤ 0 everywhere."""
        val = fam.log_dens(Y_COUNT, ETA, eta_zi=ETA_ZI)
        assert np.all(val <= 1e-10)

    def test_log_dens_requires_eta_zi(self, fam):
        with pytest.raises((ValueError, TypeError)):
            fam.log_dens(Y_COUNT, ETA)

    def test_score_eta_shape(self, fam):
        s = fam.score_eta(Y_COUNT, ETA, eta_zi=ETA_ZI)
        assert s.shape == (N,)

    def test_score_eta_zi_shape(self, fam):
        s = fam.score_eta_zi(Y_COUNT, ETA, eta_zi=ETA_ZI)
        assert s.shape == (N,)

    def test_score_eta_positive_y(self, fam):
        """For y > 0: score_eta should equal y - μ (standard Poisson score)."""
        y_pos = np.maximum(Y_COUNT, 1.0)
        mu = np.exp(ETA)
        s = fam.score_eta(y_pos, ETA, eta_zi=ETA_ZI)
        assert_allclose(s, y_pos - mu, atol=1e-8)

    def test_mu_eta(self, fam):
        assert_allclose(fam.mu_eta(ETA), np.exp(ETA), atol=1e-12)

    def test_variance(self, fam):
        mu = np.exp(ETA)
        assert_allclose(fam.variance(mu), mu, atol=1e-12)


# ---------------------------------------------------------------------------
# ZINegativeBinomial — fully implemented
# ---------------------------------------------------------------------------

class TestZINegativeBinomial:

    @pytest.fixture
    def fam(self):
        return ZINegativeBinomial()  # free theta

    @pytest.fixture
    def fam_fixed(self):
        return ZINegativeBinomial(theta=2.0)

    def test_attributes(self, fam):
        assert fam.has_zi is True
        assert fam.family == "zi_negative_binomial"
        assert fam.link == "log"
        assert fam.n_phis == 1  # free theta

    def test_attributes_fixed_theta(self, fam_fixed):
        assert fam_fixed.n_phis == 0  # theta fixed → no phis parameter

    def test_linkinv(self, fam):
        assert_allclose(fam.linkinv(ETA), np.exp(ETA), atol=1e-12)

    def test_log_dens_positive_y_vs_scipy(self, fam_fixed):
        """For y > 0, ZINB reduces to log(1-π) + log NB(y; μ, θ)."""
        theta = 2.0
        mu = np.exp(ETA)
        pi = expit(ETA_ZI)
        y_pos = np.maximum(Y_COUNT, 1.0)
        p_nb = theta / (theta + mu)
        ref_nb = nbinom.logpmf(y_pos.astype(int), n=theta, p=p_nb)
        ref = np.log(1 - pi) + ref_nb

        val = fam_fixed.log_dens(y_pos, ETA, eta_zi=ETA_ZI)
        assert_allclose(val, ref, atol=1e-7)

    def test_log_dens_zeros(self, fam_fixed):
        """For y = 0: log p = log(π + (1-π)·NB(0; μ, θ))."""
        theta = 2.0
        mu = np.exp(ETA)
        pi = expit(ETA_ZI)
        p_nb = theta / (theta + mu)
        log_nb0 = nbinom.logpmf(0, n=theta, p=p_nb)
        ref = np.log(pi + (1 - pi) * np.exp(log_nb0))

        y_zero = np.zeros(N)
        val = fam_fixed.log_dens(y_zero, ETA, eta_zi=ETA_ZI)
        assert_allclose(val, ref, atol=1e-7)

    def test_log_dens_all_leq_zero(self, fam_fixed):
        val = fam_fixed.log_dens(Y_COUNT, ETA, eta_zi=ETA_ZI)
        assert np.all(val <= 1e-10)

    def test_log_dens_free_theta_via_phis(self, fam):
        phis = np.array([np.log(2.0)])  # theta = 2
        fam2 = ZINegativeBinomial(theta=2.0)
        val_free  = fam.log_dens(Y_COUNT, ETA, phis=phis, eta_zi=ETA_ZI)
        val_fixed = fam2.log_dens(Y_COUNT, ETA, eta_zi=ETA_ZI)
        assert_allclose(val_free, val_fixed, atol=1e-10)

    def test_log_dens_requires_eta_zi(self, fam_fixed):
        with pytest.raises((ValueError, TypeError)):
            fam_fixed.log_dens(Y_COUNT, ETA)

    def test_score_eta_shape(self, fam_fixed):
        s = fam_fixed.score_eta(Y_COUNT, ETA, eta_zi=ETA_ZI)
        assert s.shape == (N,)

    def test_score_eta_zi_shape(self, fam_fixed):
        s = fam_fixed.score_eta_zi(Y_COUNT, ETA, eta_zi=ETA_ZI)
        assert s.shape == (N,)

    def test_score_phis_returns_scalar_array(self, fam):
        phis = np.array([np.log(2.0)])
        s = fam.score_phis(Y_COUNT, ETA, phis=phis, eta_zi=ETA_ZI)
        assert s is not None
        assert s.shape == (1,)

    def test_score_phis_none_for_fixed_theta(self, fam_fixed):
        s = fam_fixed.score_phis(Y_COUNT, ETA, eta_zi=ETA_ZI)
        assert s is None


# ---------------------------------------------------------------------------
# ZIBinomial — stub (only log_dens is NotImplemented)
# ---------------------------------------------------------------------------

class TestZIBinomialStub:

    def test_attributes(self):
        fam = ZIBinomial()
        assert fam.has_zi is True
        assert fam.family == "zi_binomial"
        assert fam.link == "logit"
        assert fam.n_phis == 0

    def test_linkinv_works(self):
        """linkinv IS implemented for ZIBinomial."""
        fam = ZIBinomial()
        assert_allclose(fam.linkinv(ETA), expit(ETA), atol=1e-12)

    def test_log_dens_raises(self):
        fam = ZIBinomial()
        with pytest.raises(NotImplementedError):
            fam.log_dens(Y_COUNT, ETA, eta_zi=ETA_ZI)


# ---------------------------------------------------------------------------
# Hurdle stubs — all methods raise NotImplementedError
# ---------------------------------------------------------------------------

class TestHurdleFamilyAttributes:

    @pytest.mark.parametrize("cls, family_name, link, n_phis", [
        (HurdlePoisson,         "hurdle_poisson",          "log",   0),
        (HurdleNegativeBinomial,"hurdle_negative_binomial","log",   1),
        (HurdleBeta,            "hurdle_beta",             "logit", 1),
        (HurdleLogNormal,       "hurdle_lognormal",        "log",   1),
    ])
    def test_has_zi_true(self, cls, family_name, link, n_phis):
        assert cls().has_zi is True

    @pytest.mark.parametrize("cls, family_name, link, n_phis", [
        (HurdlePoisson,         "hurdle_poisson",          "log",   0),
        (HurdleNegativeBinomial,"hurdle_negative_binomial","log",   1),
        (HurdleBeta,            "hurdle_beta",             "logit", 1),
        (HurdleLogNormal,       "hurdle_lognormal",        "log",   1),
    ])
    def test_family_name(self, cls, family_name, link, n_phis):
        assert cls().family == family_name

    @pytest.mark.parametrize("cls, family_name, link, n_phis", [
        (HurdlePoisson,         "hurdle_poisson",          "log",   0),
        (HurdleNegativeBinomial,"hurdle_negative_binomial","log",   1),
        (HurdleBeta,            "hurdle_beta",             "logit", 1),
        (HurdleLogNormal,       "hurdle_lognormal",        "log",   1),
    ])
    def test_link(self, cls, family_name, link, n_phis):
        assert cls().link == link

    @pytest.mark.parametrize("cls, family_name, link, n_phis", [
        (HurdlePoisson,         "hurdle_poisson",          "log",   0),
        (HurdleNegativeBinomial,"hurdle_negative_binomial","log",   1),
        (HurdleBeta,            "hurdle_beta",             "logit", 1),
        (HurdleLogNormal,       "hurdle_lognormal",        "log",   1),
    ])
    def test_n_phis(self, cls, family_name, link, n_phis):
        assert cls().n_phis == n_phis


class TestHurdleFamilyNotImplemented:

    @pytest.mark.parametrize("cls", [
        HurdlePoisson, HurdleNegativeBinomial, HurdleBeta, HurdleLogNormal
    ])
    def test_log_dens_raises(self, cls):
        with pytest.raises(NotImplementedError):
            cls().log_dens(Y_COUNT, ETA)

    @pytest.mark.parametrize("cls", [
        HurdlePoisson, HurdleNegativeBinomial, HurdleBeta, HurdleLogNormal
    ])
    def test_linkinv_raises(self, cls):
        with pytest.raises(NotImplementedError):
            cls().linkinv(ETA)


# ---------------------------------------------------------------------------
# MixedModel: ZI families with zi_fixed accepted; hurdle stubs rejected
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_count_data():
    n, K = 20, 4
    rng2 = np.random.default_rng(1)
    ids = np.repeat(np.arange(n), K)
    time = np.tile(np.arange(K), n)
    y = rng2.poisson(2, n * K)
    return pd.DataFrame({"id": ids, "time": time, "y": y})


class TestMixedModelZIAccepted:
    """ZIPoisson and ZINegativeBinomial can be constructed (and fit) with MixedModel."""

    def test_zip_construction_accepted(self, minimal_count_data):
        from glmmadaptive import MixedModel
        # Should NOT raise — ZIPoisson is implemented
        model = MixedModel(
            fixed="y ~ time",
            random="~ 1 | id",
            data=minimal_count_data,
            family=ZIPoisson(),
            zi_fixed="~ 1",
        )
        assert model is not None

    def test_zinb_construction_accepted(self, minimal_count_data):
        from glmmadaptive import MixedModel
        model = MixedModel(
            fixed="y ~ time",
            random="~ 1 | id",
            data=minimal_count_data,
            family=ZINegativeBinomial(),
            zi_fixed="~ 1",
        )
        assert model is not None

    def test_zi_family_requires_zi_fixed(self, minimal_count_data):
        """Omitting zi_fixed for a ZI family must raise ValueError."""
        from glmmadaptive import MixedModel
        with pytest.raises(ValueError, match="zi_fixed"):
            MixedModel(
                fixed="y ~ time",
                random="~ 1 | id",
                data=minimal_count_data,
                family=ZIPoisson(),
                # zi_fixed omitted intentionally
            )


class TestMixedModelHurdleStillRejected:
    """Hurdle families are still stubs — MixedModel should fail when fitting."""

    @pytest.mark.parametrize("cls", [
        HurdlePoisson, HurdleNegativeBinomial, HurdleBeta, HurdleLogNormal,
    ])
    def test_fit_raises(self, cls, minimal_count_data):
        from glmmadaptive import MixedModel
        model = MixedModel(
            fixed="y ~ time",
            random="~ 1 | id",
            data=minimal_count_data,
            family=cls(),
            zi_fixed="~ 1",
        )
        with pytest.raises(NotImplementedError):
            model.fit()
