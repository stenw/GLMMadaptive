"""
Unit tests for linear-algebra utilities.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from glmmadaptive.utils.linalg import nearPD, cov_to_chol, chol_to_cov, log_dmvnorm, dmvnorm


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
