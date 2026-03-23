"""
Unit tests for family log-density, linkinv, variance, and score functions.

Each test compares against scipy reference implementations or analytic results.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit, logit
from scipy.stats import binom, poisson, nbinom, gamma as sp_gamma, beta as sp_beta

from glmmadaptive.families.standard import Binomial, Poisson, NegativeBinomial, Gamma, Beta


rng = np.random.default_rng(123)
ETA = rng.normal(0, 1, 20)
Y_BIN = rng.binomial(1, 0.5, 20).astype(float)
Y_COUNT = rng.poisson(3, 20).astype(float)
Y_POS = np.abs(rng.normal(1, 0.5, 20)) + 0.01
Y_PROP = rng.beta(2, 3, 20)


# ---------------------------------------------------------------------------
# Binomial
# ---------------------------------------------------------------------------

class TestBinomial:
    @pytest.fixture
    def fam(self):
        return Binomial()

    def test_linkinv_logit(self, fam):
        assert_allclose(fam.linkinv(ETA), expit(ETA), atol=1e-12)

    def test_linkinv_probit(self):
        from scipy.stats import norm
        fam = Binomial(link="probit")
        assert_allclose(fam.linkinv(ETA), norm.cdf(ETA), atol=1e-12)

    def test_linkinv_cloglog(self):
        fam = Binomial(link="cloglog")
        mu = fam.linkinv(ETA)
        assert np.all(mu > 0) and np.all(mu < 1)

    def test_log_dens_binary(self, fam):
        mu = expit(ETA)
        ref = Y_BIN * np.log(mu) + (1 - Y_BIN) * np.log(1 - mu)
        val = fam.log_dens(Y_BIN, ETA)
        assert_allclose(val, ref, atol=1e-10)

    def test_log_dens_nonnegative_for_valid_y(self, fam):
        """log-density must always be ≤ 0 for probabilities in (0,1)."""
        val = fam.log_dens(Y_BIN, ETA)
        assert np.all(val <= 0)

    def test_score_eta_logit(self, fam):
        """Score = y - μ for logit link."""
        mu = expit(ETA)
        ref = Y_BIN - mu
        val = fam.score_eta(Y_BIN, ETA)
        assert_allclose(val, ref, atol=1e-10)

    def test_variance(self, fam):
        mu = expit(ETA)
        assert_allclose(fam.variance(mu), mu * (1 - mu), atol=1e-12)

    def test_mu_eta_logit(self, fam):
        mu = expit(ETA)
        assert_allclose(fam.mu_eta(ETA), mu * (1 - mu), atol=1e-12)

    def test_invalid_link(self):
        with pytest.raises(ValueError):
            Binomial(link="identity")


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

class TestPoisson:
    @pytest.fixture
    def fam(self):
        return Poisson()

    def test_linkinv_log(self, fam):
        assert_allclose(fam.linkinv(ETA), np.exp(ETA), atol=1e-12)

    def test_log_dens(self, fam):
        mu = np.exp(ETA)
        from scipy.special import gammaln
        ref = Y_COUNT * np.log(mu) - mu - gammaln(Y_COUNT + 1)
        val = fam.log_dens(Y_COUNT, ETA)
        assert_allclose(val, ref, atol=1e-10)

    def test_log_dens_vs_scipy(self, fam):
        mu = np.exp(ETA)
        ref = poisson.logpmf(Y_COUNT.astype(int), mu)
        val = fam.log_dens(Y_COUNT, ETA)
        assert_allclose(val, ref, atol=1e-8)

    def test_score_log_link(self, fam):
        """Score = y - μ for log link."""
        mu = np.exp(ETA)
        val = fam.score_eta(Y_COUNT, ETA)
        assert_allclose(val, Y_COUNT - mu, atol=1e-10)

    def test_variance(self, fam):
        mu = np.exp(ETA)
        assert_allclose(fam.variance(mu), mu, atol=1e-12)


# ---------------------------------------------------------------------------
# NegativeBinomial
# ---------------------------------------------------------------------------

class TestNegativeBinomial:
    @pytest.fixture
    def fam(self):
        return NegativeBinomial()  # theta estimated

    @pytest.fixture
    def fam_fixed(self):
        return NegativeBinomial(theta=2.0)

    def test_log_dens_fixed_theta(self, fam_fixed):
        """Compare against scipy.stats.nbinom (uses p parameterisation)."""
        ETA_pos = np.abs(ETA[:5])  # use positive eta for non-trivial test
        mu = np.exp(ETA_pos)
        theta = 2.0
        p = theta / (theta + mu)
        ref = nbinom.logpmf(Y_COUNT[:5].astype(int), n=theta, p=p)
        val = fam_fixed.log_dens(Y_COUNT[:5], ETA_pos)
        assert_allclose(val, ref, atol=1e-8)

    def test_log_dens_with_phis(self, fam):
        phis = np.array([np.log(3.0)])  # theta = 3
        fam3 = NegativeBinomial(theta=3.0)
        val1 = fam.log_dens(Y_COUNT[:5], ETA[:5], phis=phis)
        val2 = fam3.log_dens(Y_COUNT[:5], ETA[:5])
        assert_allclose(val1, val2, atol=1e-8)

    def test_score_eta_shape(self, fam_fixed):
        val = fam_fixed.score_eta(Y_COUNT[:5], ETA[:5])
        assert val.shape == (5,)

    def test_score_phis_returns_array(self, fam):
        phis = np.array([0.5])
        val = fam.score_phis(Y_COUNT[:5], ETA[:5], phis=phis)
        assert val is not None
        assert val.shape == (1,)

    def test_score_phis_none_for_fixed_theta(self, fam_fixed):
        val = fam_fixed.score_phis(Y_COUNT[:5], ETA[:5])
        assert val is None

    def test_requires_phis_when_free(self, fam):
        with pytest.raises((ValueError, TypeError)):
            fam.log_dens(Y_COUNT[:5], ETA[:5])  # no phis provided


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------

class TestGamma:
    @pytest.fixture
    def fam(self):
        return Gamma()

    def test_log_dens_vs_scipy(self, fam):
        """Compare against scipy.stats.gamma."""
        eta = np.abs(ETA[:5])  # log link; use positive eta
        mu = np.exp(eta)
        phis = np.array([np.log(2.0)])  # alpha = 2
        alpha = 2.0
        val = fam.log_dens(Y_POS[:5], eta, phis=phis)
        # scipy.stats.gamma: shape=alpha, scale=mu/alpha
        ref = sp_gamma.logpdf(Y_POS[:5], a=alpha, scale=mu / alpha)
        assert_allclose(val, ref, atol=1e-8)

    def test_requires_phis(self, fam):
        with pytest.raises(ValueError):
            fam.log_dens(Y_POS[:5], ETA[:5])


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------

class TestBeta:
    @pytest.fixture
    def fam(self):
        return Beta()

    def test_linkinv_logit(self, fam):
        assert_allclose(fam.linkinv(ETA), expit(ETA), atol=1e-12)

    def test_log_dens_vs_scipy(self, fam):
        """Compare against scipy.stats.beta."""
        phi = 5.0
        phis = np.array([np.log(phi)])
        mu = expit(ETA[:5])
        a = mu * phi
        b = (1 - mu) * phi
        val = fam.log_dens(Y_PROP[:5], ETA[:5], phis=phis)
        ref = np.array([
            sp_beta.logpdf(Y_PROP[j], a[j], b[j]) for j in range(5)
        ])
        assert_allclose(val, ref, atol=1e-8)

    def test_requires_phis(self, fam):
        with pytest.raises(ValueError):
            fam.log_dens(Y_PROP[:5], ETA[:5])
