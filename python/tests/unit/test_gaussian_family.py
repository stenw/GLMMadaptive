"""
Unit tests for Gaussian and StudentsT families.

Tests cover:
- Attributes (family name, link, n_phis, has_zi)
- linkinv / mu_eta correctness
- log_dens numerical values vs scipy reference
- score_eta and score_phis (finite-difference check)
- score_phis sums to zero at MLE sigma
- MixedModel accepts Gaussian / rejects missing phis
- StudentsT converges to Gaussian as df → ∞
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm as sp_norm, t as sp_t

from glmmadaptive.families import Gaussian, StudentsT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
Y = RNG.normal(2.0, 1.5, 50)
ETA = RNG.normal(2.0, 0.3, 50)
PHIS = np.array([np.log(1.5)])  # sigma = 1.5


def _fd_score_eta(fam, y, eta, phis, h=1e-5):
    """Finite-difference approximation of score_eta."""
    return (fam.log_dens(y, eta + h, phis) - fam.log_dens(y, eta - h, phis)) / (2 * h)


def _fd_score_phis(fam, y, eta, phis, h=1e-5):
    """Finite-difference approximation of score_phis (summed over obs)."""
    p_plus = phis.copy(); p_plus[0] += h
    p_minus = phis.copy(); p_minus[0] -= h
    s_plus = np.sum(fam.log_dens(y, eta, p_plus))
    s_minus = np.sum(fam.log_dens(y, eta, p_minus))
    return np.array([(s_plus - s_minus) / (2 * h)])


# ===========================================================================
# Gaussian
# ===========================================================================

class TestGaussianAttributes:
    def test_family_name(self):
        assert Gaussian().family == "gaussian"

    def test_default_link(self):
        assert Gaussian().link == "identity"

    def test_n_phis(self):
        assert Gaussian().n_phis == 1

    def test_has_zi(self):
        assert Gaussian().has_zi is False

    def test_log_link(self):
        fam = Gaussian(link="log")
        assert fam.link == "log"

    def test_inverse_link(self):
        fam = Gaussian(link="inverse")
        assert fam.link == "inverse"

    def test_bad_link(self):
        with pytest.raises(ValueError, match="Unknown link"):
            Gaussian(link="logit")


class TestGaussianLinkinv:
    def test_identity(self):
        eta = np.array([-1.0, 0.0, 2.5])
        assert_allclose(Gaussian().linkinv(eta), eta)

    def test_log(self):
        eta = np.array([0.0, 1.0, 2.0])
        assert_allclose(Gaussian(link="log").linkinv(eta), np.exp(eta))

    def test_inverse(self):
        eta = np.array([0.5, 1.0, 2.0])
        assert_allclose(Gaussian(link="inverse").linkinv(eta), 1.0 / eta)

    def test_mu_eta_identity(self):
        eta = np.array([1.0, 2.0, 3.0])
        assert_allclose(Gaussian().mu_eta(eta), np.ones(3))

    def test_mu_eta_log(self):
        eta = np.array([0.5, 1.0])
        assert_allclose(Gaussian(link="log").mu_eta(eta), np.exp(eta))

    def test_variance(self):
        mu = np.array([1.0, 2.0, 3.0])
        assert_allclose(Gaussian().variance(mu), np.ones(3))


class TestGaussianLogDens:
    def test_matches_scipy(self):
        fam = Gaussian()
        sigma = np.exp(PHIS[0])
        expected = sp_norm.logpdf(Y, loc=ETA, scale=sigma)
        assert_allclose(fam.log_dens(Y, ETA, PHIS), expected, rtol=1e-12)

    def test_requires_phis(self):
        with pytest.raises(ValueError, match="requires phis"):
            Gaussian().log_dens(Y, ETA, phis=None)

    def test_log_link(self):
        fam = Gaussian(link="log")
        eta_pos = np.abs(ETA)  # keep eta positive for log link
        sigma = np.exp(PHIS[0])
        mu = np.exp(eta_pos)
        expected = sp_norm.logpdf(Y, loc=mu, scale=sigma)
        assert_allclose(fam.log_dens(Y, eta_pos, PHIS), expected, rtol=1e-12)

    def test_sum_finite(self):
        ld = Gaussian().log_dens(Y, ETA, PHIS)
        assert np.all(np.isfinite(ld))


class TestGaussianScores:
    def test_score_eta_matches_fd(self):
        fam = Gaussian()
        analytic = fam.score_eta(Y, ETA, PHIS)
        fd = _fd_score_eta(fam, Y, ETA, PHIS)
        assert_allclose(analytic, fd, rtol=1e-5, atol=1e-8)

    def test_score_phis_matches_fd(self):
        fam = Gaussian()
        analytic = fam.score_phis(Y, ETA, PHIS)
        fd = _fd_score_phis(fam, Y, ETA, PHIS)
        assert_allclose(analytic, fd, rtol=1e-5)

    def test_score_phis_zero_at_mle_sigma(self):
        """At the MLE sigma (with mu=eta=0), score_phis sums to exactly 0."""
        y = RNG.normal(0.0, 2.0, 500)
        eta = np.zeros(500)
        # MLE of sigma when mu fixed at 0: sigma_mle = sqrt(mean(y^2))
        sigma_mle = np.sqrt(np.mean(y ** 2))
        phis_mle = np.array([np.log(sigma_mle)])
        score = Gaussian().score_phis(y, eta, phis_mle)
        assert_allclose(score[0], 0.0, atol=1e-10)

    def test_score_eta_identity_formula(self):
        """For identity link: score_eta = (y - eta) / sigma²."""
        fam = Gaussian()
        sigma2 = np.exp(2.0 * PHIS[0])
        expected = (Y - ETA) / sigma2
        assert_allclose(fam.score_eta(Y, ETA, PHIS), expected, rtol=1e-12)

    def test_score_phis_requires_phis(self):
        with pytest.raises(ValueError, match="requires phis"):
            Gaussian().score_phis(Y, ETA, phis=None)


# ===========================================================================
# StudentsT
# ===========================================================================

class TestStudentsTAttributes:
    def test_family_name(self):
        assert StudentsT(df=5).family == "students_t"

    def test_df_stored(self):
        assert StudentsT(df=3.0).df == 3.0

    def test_default_link(self):
        assert StudentsT(df=5).link == "identity"

    def test_n_phis(self):
        assert StudentsT(df=5).n_phis == 1

    def test_has_zi(self):
        assert StudentsT(df=5).has_zi is False

    def test_bad_df(self):
        with pytest.raises(ValueError, match="df must be positive"):
            StudentsT(df=-1)

    def test_bad_link(self):
        with pytest.raises(ValueError, match="Unknown link"):
            StudentsT(df=5, link="logit")


class TestStudentsTLogDens:
    def test_matches_scipy(self):
        fam = StudentsT(df=5)
        sigma = np.exp(PHIS[0])
        expected = sp_t.logpdf((Y - ETA) / sigma, df=5) - PHIS[0]
        assert_allclose(fam.log_dens(Y, ETA, PHIS), expected, rtol=1e-12)

    def test_requires_phis(self):
        with pytest.raises(ValueError, match="requires phis"):
            StudentsT(df=5).log_dens(Y, ETA, phis=None)

    def test_converges_to_gaussian_large_df(self):
        """StudentsT(df=1e6) log_dens ≈ Gaussian log_dens."""
        ld_t = StudentsT(df=1e6).log_dens(Y, ETA, PHIS)
        ld_n = Gaussian().log_dens(Y, ETA, PHIS)
        assert_allclose(ld_t, ld_n, atol=1e-4)


class TestStudentsTScores:
    def test_score_eta_matches_fd(self):
        fam = StudentsT(df=5)
        analytic = fam.score_eta(Y, ETA, PHIS)
        fd = _fd_score_eta(fam, Y, ETA, PHIS)
        assert_allclose(analytic, fd, rtol=1e-5, atol=1e-8)

    def test_score_phis_matches_fd(self):
        fam = StudentsT(df=5)
        analytic = fam.score_phis(Y, ETA, PHIS)
        fd = _fd_score_phis(fam, Y, ETA, PHIS)
        assert_allclose(analytic, fd, rtol=1e-5)

    def test_score_eta_converges_to_gaussian(self):
        """score_eta for StudentsT(df=1e6) ≈ Gaussian score_eta."""
        s_t = StudentsT(df=1e6).score_eta(Y, ETA, PHIS)
        s_n = Gaussian().score_eta(Y, ETA, PHIS)
        assert_allclose(s_t, s_n, atol=1e-4)

    def test_score_phis_converges_to_gaussian(self):
        s_t = StudentsT(df=1e6).score_phis(Y, ETA, PHIS)
        s_n = Gaussian().score_phis(Y, ETA, PHIS)
        assert_allclose(s_t, s_n, atol=1e-3)


# ===========================================================================
# MixedModel integration (construction only)
# ===========================================================================

class TestGaussianMixedModelConstruction:
    """Check that MixedModel accepts Gaussian/StudentsT without error."""

    def _make_data(self):
        import pandas as pd
        rng = np.random.default_rng(1)
        n, K = 30, 4
        ids = np.repeat(np.arange(n), K)
        time = np.tile(np.arange(K), n).astype(float)
        b = rng.normal(0, 0.5, n)
        y = 1.0 + 0.3 * time + b[ids] + rng.normal(0, 1.0, n * K)
        return pd.DataFrame({"id": ids, "time": time, "y": y})

    def test_gaussian_accepted(self):
        from glmmadaptive import MixedModel
        df = self._make_data()
        m = MixedModel(
            fixed="y ~ time", random="~ 1 | id",
            data=df, family=Gaussian(),
        )
        assert m is not None

    def test_students_t_accepted(self):
        from glmmadaptive import MixedModel
        df = self._make_data()
        m = MixedModel(
            fixed="y ~ time", random="~ 1 | id",
            data=df, family=StudentsT(df=5),
        )
        assert m is not None
