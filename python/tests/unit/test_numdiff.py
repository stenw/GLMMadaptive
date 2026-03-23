"""
Unit tests for numerical differentiation utilities.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from glmmadaptive.utils.numdiff import fd_grad, cd_grad, fd_hess, cd_hess


# ---------------------------------------------------------------------------
# Analytic test functions
# ---------------------------------------------------------------------------

def f_quadratic(x):
    """f(x) = x^T A x / 2 + b^T x.  Hessian = A."""
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([0.5, -0.3])
    return 0.5 * x @ A @ x + b @ x


def grad_quadratic(x):
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([0.5, -0.3])
    return A @ x + b


def hess_quadratic():
    return np.array([[3.0, 1.0], [1.0, 2.0]])


def f_sine(x):
    return np.sum(np.sin(x))


def grad_sine(x):
    return np.cos(x)


# ---------------------------------------------------------------------------
# fd_grad
# ---------------------------------------------------------------------------

class TestFDGrad:
    def test_quadratic(self):
        x = np.array([1.0, -0.5])
        g = fd_grad(f_quadratic, x)
        assert_allclose(g, grad_quadratic(x), rtol=1e-3)

    def test_sine(self):
        x = np.array([0.3, 1.2, -0.7])
        g = fd_grad(f_sine, x)
        assert_allclose(g, grad_sine(x), rtol=1e-3)


# ---------------------------------------------------------------------------
# cd_grad
# ---------------------------------------------------------------------------

class TestCDGrad:
    def test_quadratic(self):
        x = np.array([1.0, -0.5])
        g = cd_grad(f_quadratic, x)
        assert_allclose(g, grad_quadratic(x), rtol=1e-6)

    def test_sine(self):
        x = np.array([0.3, 1.2, -0.7])
        g = cd_grad(f_sine, x)
        assert_allclose(g, grad_sine(x), rtol=1e-6)

    def test_more_accurate_than_fd(self):
        """Central differences should be more accurate than forward differences."""
        x = np.array([2.0, -1.0])
        err_fd = np.max(np.abs(fd_grad(f_quadratic, x) - grad_quadratic(x)))
        err_cd = np.max(np.abs(cd_grad(f_quadratic, x) - grad_quadratic(x)))
        assert err_cd <= err_fd + 1e-12  # CD is at least as accurate


# ---------------------------------------------------------------------------
# fd_hess
# ---------------------------------------------------------------------------

class TestFDHess:
    def test_quadratic(self):
        """For a quadratic the FD Hessian should be close to the true Hessian."""
        x = np.array([1.0, -0.5])
        H = fd_hess(f_quadratic, x)
        assert_allclose(H, hess_quadratic(), rtol=5e-3, atol=5e-3)

    def test_symmetric(self):
        x = np.array([0.5, -0.3])
        H = fd_hess(f_quadratic, x)
        assert_allclose(H, H.T, atol=1e-8)


# ---------------------------------------------------------------------------
# cd_hess
# ---------------------------------------------------------------------------

class TestCDHess:
    def test_quadratic(self):
        x = np.array([1.0, -0.5])
        H = cd_hess(f_quadratic, x)
        assert_allclose(H, hess_quadratic(), rtol=1e-4, atol=1e-4)

    def test_symmetric(self):
        x = np.array([0.5, -0.3])
        H = cd_hess(f_quadratic, x)
        assert_allclose(H, H.T, atol=1e-8)

    def test_more_accurate_than_fd(self):
        x = np.array([1.0, -0.5])
        err_fd = np.max(np.abs(fd_hess(f_quadratic, x) - hess_quadratic()))
        err_cd = np.max(np.abs(cd_hess(f_quadratic, x) - hess_quadratic()))
        assert err_cd <= err_fd + 1e-8
