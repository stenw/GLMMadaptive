"""
Unit tests for MixModResults.simulate().

Tests shape, reproducibility, type_="mean_subject", new_RE, and that simulated
means are close to fitted values for large nsim.
"""

import numpy as np
import pandas as pd
import pytest

from glmmadaptive.core.mixed_model import MixedModel
from glmmadaptive.families.standard import Binomial, Poisson


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_binary_fit(n=80, K=6, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n), K)
    b = rng.normal(0, 1.0, n)[ids]
    x = rng.normal(0, 1, n * K)
    from scipy.special import expit
    p = expit(-0.3 + 0.8 * x + b)
    y = rng.binomial(1, p)
    df = pd.DataFrame({"y": y, "x": x, "id": ids})
    m = MixedModel("y ~ x", "~ 1 | id", df, family=Binomial())
    return m.fit()


def _make_poisson_fit(n=60, K=5, seed=1):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n), K)
    b = rng.normal(0, 0.5, n)[ids]
    x = rng.normal(0, 1, n * K)
    mu = np.exp(0.5 + 0.4 * x + b)
    y = rng.poisson(mu)
    df = pd.DataFrame({"y": y, "x": x, "id": ids})
    m = MixedModel("y ~ x", "~ 1 | id", df, family=Poisson())
    return m.fit()


# Cache to avoid re-fitting
_binary_fit = None
_poisson_fit = None


@pytest.fixture(scope="module")
def binary_fit():
    global _binary_fit
    if _binary_fit is None:
        _binary_fit = _make_binary_fit()
    return _binary_fit


@pytest.fixture(scope="module")
def poisson_fit():
    global _poisson_fit
    if _poisson_fit is None:
        _poisson_fit = _make_poisson_fit()
    return _poisson_fit


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSimulateShape:
    def test_default_shape(self, binary_fit):
        sims = binary_fit.simulate(nsim=3, seed=42)
        n_obs = binary_fit.model.nobs
        assert sims.shape == (n_obs, 3)

    def test_column_names(self, binary_fit):
        sims = binary_fit.simulate(nsim=2, seed=1)
        assert list(sims.columns) == ["sim_1", "sim_2"]

    def test_single_sim(self, binary_fit):
        sims = binary_fit.simulate(nsim=1, seed=5)
        assert sims.shape == (binary_fit.model.nobs, 1)


class TestSimulateReproducibility:
    def test_same_seed_same_result(self, binary_fit):
        s1 = binary_fit.simulate(nsim=5, seed=77)
        s2 = binary_fit.simulate(nsim=5, seed=77)
        np.testing.assert_array_equal(s1.values, s2.values)

    def test_different_seed_different_result(self, binary_fit):
        s1 = binary_fit.simulate(nsim=5, seed=1)
        s2 = binary_fit.simulate(nsim=5, seed=2)
        assert not np.allclose(s1.values, s2.values)


class TestSimulateBinaryValues:
    def test_binary_in_0_1(self, binary_fit):
        sims = binary_fit.simulate(nsim=20, seed=3)
        vals = sims.values.ravel()
        assert set(vals).issubset({0.0, 1.0})

    def test_poisson_non_negative(self, poisson_fit):
        sims = poisson_fit.simulate(nsim=10, seed=3)
        assert (sims.values >= 0).all()


class TestSimulateMeanSubject:
    def test_type_mean_subject_shape(self, binary_fit):
        sims = binary_fit.simulate(nsim=5, seed=9, type_="mean_subject")
        assert sims.shape == (binary_fit.model.nobs, 5)

    def test_mean_close_to_fitted(self, binary_fit):
        """Mean over many simulations should approximate mean_subject fitted values."""
        sims = binary_fit.simulate(nsim=500, seed=0, type_="mean_subject")
        sim_mean = sims.values.mean(axis=1)
        fitted_vals = binary_fit.fitted(type_="mean_subject")
        # Binomial mean should be close with many samples (allow 10% tolerance)
        np.testing.assert_allclose(sim_mean, fitted_vals, atol=0.10)


class TestSimulateNewRE:
    def test_new_re_shape(self, binary_fit):
        sims = binary_fit.simulate(nsim=5, seed=10, new_RE=True)
        assert sims.shape == (binary_fit.model.nobs, 5)

    def test_new_re_binary(self, binary_fit):
        sims = binary_fit.simulate(nsim=10, seed=10, new_RE=True)
        vals = sims.values.ravel()
        assert set(vals).issubset({0.0, 1.0})
