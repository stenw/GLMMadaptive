"""
Unit tests for linear-algebra utilities.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from glmmadaptive.utils.linalg import nearPD, cov_to_chol, chol_to_cov, log_dmvnorm, dmvnorm, dmvt_log, dmvt_log_grad


class TestNearPD:
    def test_already_pd(self):
        """An already PD matrix should be returned essentially unchanged."""
        A = np.array([[2.0, 0.5], [0.5, 1.0]])
        B = nearPD(A)
        assert_allclose(B, A, atol=1e-8)

    def test_not_pd_is_projected(self):
        """A matrix with negative eigenvalue should become PD."""
        A = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: -1, 3
        B = nearPD(A)
        eigvals = np.linalg.eigvalsh(B)
        assert np.all(eigvals > 0)

    def test_symmetric_output(self):
        A = np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.9], [0.8, 0.9, 1.0]])
        B = nearPD(A)
        assert_allclose(B, B.T, atol=1e-10)

    def test_identity_unchanged(self):
        A = np.eye(3)
        B = nearPD(A)
        assert_allclose(B, A, atol=1e-8)

    def test_near_singular(self):
        """Near-singular matrix should become strictly PD."""
        A = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
        B = nearPD(A)
        eigvals = np.linalg.eigvalsh(B)
        assert np.all(eigvals > 0)


class TestCholTransf:
    def test_roundtrip_1d(self):
        """cov_to_chol then chol_to_cov should recover D."""
        D = np.array([[2.5]])
        params = cov_to_chol(D)
        D_rec = chol_to_cov(params, q=1)
        assert_allclose(D_rec, D, atol=1e-10)

    def test_roundtrip_2d(self):
        D = np.array([[1.0, 0.4], [0.4, 0.9]])
        params = cov_to_chol(D)
        D_rec = chol_to_cov(params, q=2)
        assert_allclose(D_rec, D, atol=1e-10)

    def test_roundtrip_3d(self):
        from scipy.linalg import cholesky
        L = np.array([[1.0, 0.0, 0.0], [0.5, 1.2, 0.0], [0.2, 0.3, 0.8]])
        D = L @ L.T
        params = cov_to_chol(D)
        D_rec = chol_to_cov(params, q=3)
        assert_allclose(D_rec, D, atol=1e-8)

    def test_diagonal_roundtrip(self):
        D = np.diag([1.0, 2.0, 3.0])
        params = cov_to_chol(D, diagonal=True)
        assert len(params) == 3
        D_rec = chol_to_cov(params, q=3, diagonal=True)
        assert_allclose(D_rec, D, atol=1e-10)

    def test_output_is_pd(self):
        """Recovered matrix should be positive definite."""
        params = np.array([0.5, 0.2, -0.3])  # arbitrary unconstrained values
        D = chol_to_cov(params, q=2)
        eigvals = np.linalg.eigvalsh(D)
        assert np.all(eigvals > 0)


class TestLogDmvnorm:
    def test_univariate_standard_normal(self):
        """log φ(0) = -0.5 log(2π)."""
        val = log_dmvnorm(np.array([0.0]), cov=np.array([[1.0]]))
        assert_allclose(val, -0.5 * np.log(2 * np.pi), atol=1e-10)

    def test_univariate_nonzero(self):
        from scipy.stats import norm
        x = np.array([1.5])
        val = log_dmvnorm(x, cov=np.array([[2.0]]))
        ref = norm.logpdf(1.5, scale=np.sqrt(2.0))
        assert_allclose(val, ref, atol=1e-10)

    def test_multivariate(self):
        """Compare against scipy.stats.multivariate_normal."""
        from scipy.stats import multivariate_normal
        mean = np.array([1.0, -0.5])
        cov = np.array([[2.0, 0.5], [0.5, 1.5]])
        x = np.array([0.5, 0.3])
        val = log_dmvnorm(x, mean=mean, cov=cov)
        ref = multivariate_normal.logpdf(x, mean=mean, cov=cov)
        assert_allclose(val, ref, atol=1e-10)

    def test_batch_evaluation(self):
        """Should return an array when x has shape (n, q)."""
        from scipy.stats import multivariate_normal
        cov = np.eye(2)
        X = np.random.default_rng(0).normal(size=(10, 2))
        vals = log_dmvnorm(X, cov=cov)
        refs = multivariate_normal.logpdf(X, cov=cov)
        assert_allclose(vals, refs, atol=1e-10)

    def test_precomputed_inv(self):
        """Passing cov_inv and log_det_cov should give same result."""
        cov = np.array([[3.0, 1.0], [1.0, 2.0]])
        x = np.array([0.2, -0.1])
        cov_inv = np.linalg.inv(cov)
        _, log_det = np.linalg.slogdet(cov)
        val1 = log_dmvnorm(x, cov=cov)
        val2 = log_dmvnorm(x, cov_inv=cov_inv, log_det_cov=log_det)
        assert_allclose(val1, val2, atol=1e-12)

    def test_vector_input_returns_scalar(self):
        """A single observation should return a scalar float."""
        cov = np.eye(2)
        x = np.array([0.1, -0.2])
        val = log_dmvnorm(x, cov=cov)
        assert np.isscalar(val)


class TestDmvtLog:
    """Tests for dmvt_log() and dmvt_log_grad() — Student's-t penalty functions."""

    def _default_args(self, p=3):
        """Standard hyperparameters: mu=0, sigma=1 (inv_sigma_diag=1), df=3."""
        mu = np.zeros(p)
        inv_sigma_diag = np.ones(p)
        df = 3.0
        return mu, inv_sigma_diag, df

    def test_maximum_at_mode(self):
        """dmvt_log(mu, mu, ...) == 0 (log-density is maximised at the mode)."""
        p = 4
        mu, isd, df = self._default_args(p)
        val = dmvt_log(mu, mu, isd, df)
        assert_allclose(val, 0.0, atol=1e-12)

    def test_decreasing_away_from_mode(self):
        """Log-density decreases as we move away from mu."""
        mu, isd, df = self._default_args(p=2)
        v0 = dmvt_log(np.array([0.0, 0.0]), mu, isd, df)
        v1 = dmvt_log(np.array([1.0, 0.0]), mu, isd, df)
        v2 = dmvt_log(np.array([2.0, 0.0]), mu, isd, df)
        assert v0 > v1 > v2

    def test_approaches_normal_as_df_increases(self):
        """For large df, dmvt_log should approach the normal log-density (up to constant)."""
        from scipy.stats import multivariate_normal
        p = 3
        x = np.array([0.5, -0.3, 1.2])
        mu = np.zeros(p)
        isd = np.ones(p)
        df = 1e6  # very large df → Normal

        t_val = dmvt_log(x, mu, isd, df)
        n_val = multivariate_normal.logpdf(x, mean=mu, cov=np.eye(p))
        # dmvt_log is proportional; difference should be (roughly) the normalising constant
        # What matters is that the *shape* matches: both should decrease by same amount for a shift
        x2 = x + 0.1
        t_diff = dmvt_log(x2, mu, isd, df) - t_val
        n_diff = multivariate_normal.logpdf(x2, mean=mu, cov=np.eye(p)) - n_val
        assert_allclose(t_diff, n_diff, atol=1e-4)

    def test_scale_effect(self):
        """Larger pen_sigma (smaller inv_sigma_diag) → weaker penalty (higher density away from mu)."""
        x = np.array([1.0, 1.0])
        mu = np.zeros(2)
        df = 5.0
        # tight penalty: sigma=0.5 → inv_sigma = 4
        val_tight = dmvt_log(x, mu, np.full(2, 4.0), df)
        # loose penalty: sigma=2 → inv_sigma = 0.25
        val_loose = dmvt_log(x, mu, np.full(2, 0.25), df)
        assert val_loose > val_tight

    def test_returns_float(self):
        mu, isd, df = self._default_args()
        val = dmvt_log(np.ones(3), mu, isd, df)
        assert isinstance(val, float)

    def test_gradient_against_numerical(self):
        """dmvt_log_grad should match a numerical gradient."""
        from glmmadaptive.utils.numdiff import cd_grad
        p = 4
        rng = np.random.default_rng(42)
        x = rng.normal(size=p)
        mu = rng.normal(size=p) * 0.5
        isd = np.exp(rng.normal(size=p))   # positive
        df = 4.0

        analytic = dmvt_log_grad(x, mu, isd, df)
        numeric = cd_grad(lambda xx: dmvt_log(xx, mu, isd, df), x)
        assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)

    def test_gradient_zero_at_mode(self):
        """Gradient at x=mu should be zero."""
        p = 3
        mu = np.array([1.0, -0.5, 2.0])
        isd = np.array([1.0, 2.0, 0.5])
        df = 5.0
        grad = dmvt_log_grad(mu, mu, isd, df)
        assert_allclose(grad, np.zeros(p), atol=1e-12)

    def test_gradient_points_toward_mode(self):
        """Gradient at x > mu should be negative (points toward lower x, i.e. toward mu=0)."""
        mu = np.zeros(2)
        isd = np.ones(2)
        df = 3.0
        x = np.array([1.0, 1.0])   # both components above mu
        grad = dmvt_log_grad(x, mu, isd, df)
        assert np.all(grad < 0)
