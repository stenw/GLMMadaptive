"""
Unit tests for Gauss-Hermite quadrature utilities.

These tests are pure Python and do not require R.  They verify numerical
correctness of ``gauher()``, ``gh_adaptive()``, and ``find_posterior_mode()``
against analytic results.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from glmmadaptive.utils.quadrature import gauher, gh_adaptive, GHQuadrature


# ---------------------------------------------------------------------------
# gauher() — nodes and weights
# ---------------------------------------------------------------------------

class TestGauher:
    def test_returns_n_points(self):
        nodes, weights = gauher(5)
        assert len(nodes) == 5
        assert len(weights) == 5

    def test_nodes_symmetric(self):
        for n in (3, 5, 7, 11):
            nodes, _ = gauher(n)
            # Nodes should be symmetric about zero
            assert_allclose(nodes, -nodes[::-1], atol=1e-12)

    def test_weights_positive(self):
        for n in (3, 5, 7, 11):
            _, weights = gauher(n)
            assert np.all(weights > 0)

    def test_weights_sum(self):
        """Sum of weights = sqrt(pi) for the exp(-x^2) Gauss-Hermite rule."""
        for n in (3, 5, 7, 11):
            _, weights = gauher(n)
            assert_allclose(np.sum(weights), np.sqrt(np.pi), rtol=1e-10)

    def test_integrate_constant(self):
        """Integral of exp(-x^2) from -inf to inf = sqrt(pi)."""
        nodes, weights = gauher(11)
        result = np.sum(weights * np.ones_like(nodes))
        assert_allclose(result, np.sqrt(np.pi), rtol=1e-10)

    def test_integrate_quadratic(self):
        """Integral of x^2 exp(-x^2) dx = sqrt(pi)/2."""
        nodes, weights = gauher(11)
        result = np.sum(weights * nodes ** 2)
        assert_allclose(result, np.sqrt(np.pi) / 2.0, rtol=1e-8)

    def test_integrate_gaussian_pdf(self):
        """
        Integrate the standard normal PDF using GH quadrature.

        int exp(-t^2/2) / sqrt(2π) dt = 1

        The standard GH rule is: int g(x) exp(-x^2) dx ≈ sum_k w_k g(x_k).
        Change of variable t = sqrt(2) x:
            int f(t) dt = sqrt(2) * int f(sqrt(2) x) dx
                        = sqrt(2) * sum_k w_k f(sqrt(2) x_k) exp(x_k^2)
        where the exp(x_k^2) re-absorbs the exp(-x^2) denominator.
        """
        nodes, weights = gauher(11)
        t = np.sqrt(2) * nodes
        f_vals = np.exp(-t ** 2 / 2.0) / np.sqrt(2 * np.pi)
        # Re-absorb the exp(-x^2) factor that was pulled into the weights
        result = np.sqrt(2) * np.sum(weights * f_vals * np.exp(nodes ** 2))
        assert_allclose(result, 1.0, rtol=1e-8)

    def test_n_1(self):
        nodes, weights = gauher(1)
        assert len(nodes) == 1
        assert_allclose(nodes[0], 0.0, atol=1e-14)
        assert_allclose(weights[0], np.sqrt(np.pi), rtol=1e-10)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            gauher(0)


# ---------------------------------------------------------------------------
# gh_adaptive() — grid construction
# ---------------------------------------------------------------------------

class TestGHAdaptive:
    def _make_trivial_gh(self, n_agh=5, q=1, n_groups=3):
        """Create a GHQuadrature with identity covariance (no adaptation)."""
        modes = np.zeros((n_groups, q))
        neg_hessians = [np.eye(q) for _ in range(n_groups)]
        return gh_adaptive(n_agh, q, modes, neg_hessians)

    def test_returns_gh_quadrature(self):
        gh = self._make_trivial_gh()
        assert isinstance(gh, GHQuadrature)

    def test_node_count_1d(self):
        n_agh, q, n_groups = 7, 1, 4
        gh = self._make_trivial_gh(n_agh, q, n_groups)
        for i in range(n_groups):
            assert gh.b_nodes[i].shape == (n_agh, q)

    def test_node_count_2d(self):
        n_agh, q, n_groups = 5, 2, 3
        gh = self._make_trivial_gh(n_agh, q, n_groups)
        for i in range(n_groups):
            # n_agh^q points
            assert gh.b_nodes[i].shape == (n_agh ** q, q)

    def test_log_weights_finite(self):
        gh = self._make_trivial_gh()
        for i in range(3):
            assert np.all(np.isfinite(gh.log_weights[i]))

    def test_nodes_centred_at_mode(self):
        """With identity Hessian the nodes should be centred at the mode."""
        modes = np.array([[2.0], [-3.0]])
        neg_hessians = [np.eye(1), np.eye(1)]
        gh = gh_adaptive(5, 1, modes, neg_hessians)
        # Mean of nodes should be close to mode (symmetric quadrature rule)
        for i, mode in enumerate(modes):
            # Weighted mean of nodes ≈ mode (log-weights can differ)
            lw = gh.log_weights[i]
            w = np.exp(lw - np.max(lw))
            w /= w.sum()
            mean_node = np.sum(w[:, None] * gh.b_nodes[i], axis=0)
            assert_allclose(mean_node, mode, atol=0.5)  # loose tolerance


# ---------------------------------------------------------------------------
# find_posterior_mode()
# ---------------------------------------------------------------------------

class TestFindPosteriorMode:
    def test_gaussian_posterior(self):
        """For a Gaussian log-posterior the mode should be at the mean."""
        from glmmadaptive.utils.quadrature import find_posterior_mode

        true_mode = np.array([1.5, -0.7])

        def neg_log_post(b):
            return 0.5 * np.sum((b - true_mode) ** 2)

        mode, H = find_posterior_mode(neg_log_post, np.zeros(2))
        assert_allclose(mode, true_mode, atol=1e-4)
        # Hessian should be (close to) identity
        assert_allclose(H, np.eye(2), atol=0.05)

    def test_mode_positive_definite_hessian(self):
        from glmmadaptive.utils.quadrature import find_posterior_mode

        def neg_log_post(b):
            return 0.5 * (b[0] ** 2 + 4.0 * b[1] ** 2)

        mode, H = find_posterior_mode(neg_log_post, np.array([1.0, 1.0]))
        eigvals = np.linalg.eigvalsh(H)
        assert np.all(eigvals > 0), "Hessian should be positive definite"
