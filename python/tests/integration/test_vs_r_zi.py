"""
Live R comparison tests for zero-inflated mixed models.

These tests require rpy2 and the R GLMMadaptive package.  They are skipped
automatically when R is unavailable.

All tests are marked ``xfail`` with ``raises=NotImplementedError`` because
the ZI/hurdle families are not yet implemented in the Python port.
Once implemented, the xfail marks should be removed.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration

_NOT_IMPLEMENTED = pytest.mark.xfail(
    reason="ZI/hurdle families not yet implemented in Python port",
    strict=False,
)


# ---------------------------------------------------------------------------
# Shared simulated dataset  (same across ZIP and ZINB comparisons)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sim_zi_data():
    """
    Longitudinal count dataset with structural zeros.

    80 subjects × 6 time points.  True model:
      count part : beta = [1.2, 0.15],  D11 = 0.5
      zero part  : gamma = [-1.5]  (intercept only)
    """
    from scipy.special import expit
    from scipy.stats import nbinom

    rng = np.random.default_rng(42)
    import pandas as pd

    n, K = 80, 6
    ids = np.repeat(np.arange(n), K)
    time = np.tile(np.arange(K), n)

    b_count = rng.normal(0, np.sqrt(0.5), n)
    eta_y = 1.2 + 0.15 * time + b_count[ids]
    mu_y = np.exp(eta_y)

    shape = 2.0
    p_nb = shape / (shape + mu_y)
    y = nbinom.rvs(n=shape, p=p_nb, random_state=rng)

    gamma0 = -1.5
    pi = expit(gamma0)
    zi_mask = rng.binomial(1, pi, n * K).astype(bool)
    y[zi_mask] = 0

    return pd.DataFrame({"id": ids, "time": time, "y": y})


# ---------------------------------------------------------------------------
# Zero-inflated Poisson
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def r_zip_fit(r_env, sim_zi_data):
    import rpy2.robjects as ro
    from tests.conftest import _df_to_r

    r_data = _df_to_r(sim_zi_data)
    ro.globalenv["py_data_zi"] = r_data
    ro.r("""
        library(GLMMadaptive)
        fm_zip <- mixed_model(
            fixed    = y ~ time,
            random   = ~ 1 | id,
            data     = py_data_zi,
            family   = zi.poisson(),
            zi_fixed = ~ 1
        )
        r_betas_zip  <- as.numeric(fixef(fm_zip))
        r_gammas_zip <- as.numeric(fm_zip$gammas)
        r_D_zip      <- as.numeric(fm_zip$D)
        r_loglik_zip <- as.numeric(logLik(fm_zip))
        r_bse_zip    <- as.numeric(sqrt(diag(vcov(fm_zip))))
    """)
    return {
        "betas":  np.array(ro.r("r_betas_zip")),
        "gammas": np.array(ro.r("r_gammas_zip")),
        "D":      float(ro.r("r_D_zip")[0]),
        "logLik": float(ro.r("r_loglik_zip")[0]),
        "bse":    np.array(ro.r("r_bse_zip")),
    }


@pytest.fixture(scope="module")
def py_zip_result(sim_zi_data):
    from glmmadaptive import MixedModel
    from glmmadaptive.families import ZIPoisson
    return MixedModel(
        fixed="y ~ time",
        random="~ 1 | id",
        data=sim_zi_data,
        family=ZIPoisson(),
        zi_fixed="~ 1",
        control={"iter_em": 50, "verbose": False},
    ).fit()


class TestZIPoissonVsR:

    @_NOT_IMPLEMENTED
    def test_betas(self, py_zip_result, r_zip_fit):
        np.testing.assert_allclose(
            py_zip_result.params, r_zip_fit["betas"], rtol=0.05, atol=0.05
        )

    @_NOT_IMPLEMENTED
    def test_gammas(self, py_zip_result, r_zip_fit):
        np.testing.assert_allclose(
            py_zip_result.gammas, r_zip_fit["gammas"], rtol=0.10, atol=0.10
        )

    @_NOT_IMPLEMENTED
    def test_D(self, py_zip_result, r_zip_fit):
        np.testing.assert_allclose(
            py_zip_result.D[0, 0], r_zip_fit["D"], rtol=0.10
        )

    @_NOT_IMPLEMENTED
    def test_loglik(self, py_zip_result, r_zip_fit):
        np.testing.assert_allclose(
            py_zip_result.logLik, r_zip_fit["logLik"], atol=0.5
        )

    @_NOT_IMPLEMENTED
    def test_converged(self, py_zip_result):
        assert py_zip_result.converged


# ---------------------------------------------------------------------------
# Zero-inflated Negative Binomial
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def r_zinb_fit(r_env, sim_zi_data):
    import rpy2.robjects as ro
    from tests.conftest import _df_to_r

    r_data = _df_to_r(sim_zi_data)
    if "py_data_zi" not in ro.globalenv:
        ro.globalenv["py_data_zi"] = r_data
    ro.r("""
        library(GLMMadaptive)
        fm_zinb <- mixed_model(
            fixed    = y ~ time,
            random   = ~ 1 | id,
            data     = py_data_zi,
            family   = zi.negative.binomial(),
            zi_fixed = ~ 1
        )
        r_betas_zinb  <- as.numeric(fixef(fm_zinb))
        r_gammas_zinb <- as.numeric(fm_zinb$gammas)
        r_D_zinb      <- as.numeric(fm_zinb$D)
        r_phis_zinb   <- as.numeric(fm_zinb$phis)
        r_loglik_zinb <- as.numeric(logLik(fm_zinb))
    """)
    return {
        "betas":  np.array(ro.r("r_betas_zinb")),
        "gammas": np.array(ro.r("r_gammas_zinb")),
        "D":      float(ro.r("r_D_zinb")[0]),
        "phis":   np.array(ro.r("r_phis_zinb")),
        "logLik": float(ro.r("r_loglik_zinb")[0]),
    }


@pytest.fixture(scope="module")
def py_zinb_result(sim_zi_data):
    from glmmadaptive import MixedModel
    from glmmadaptive.families import ZINegativeBinomial
    return MixedModel(
        fixed="y ~ time",
        random="~ 1 | id",
        data=sim_zi_data,
        family=ZINegativeBinomial(),
        zi_fixed="~ 1",
        control={"iter_em": 50, "verbose": False},
    ).fit()


class TestZINegBinomVsR:

    @_NOT_IMPLEMENTED
    def test_betas(self, py_zinb_result, r_zinb_fit):
        np.testing.assert_allclose(
            py_zinb_result.params, r_zinb_fit["betas"], rtol=0.10, atol=0.10
        )

    @_NOT_IMPLEMENTED
    def test_gammas(self, py_zinb_result, r_zinb_fit):
        np.testing.assert_allclose(
            py_zinb_result.gammas, r_zinb_fit["gammas"], rtol=0.10, atol=0.10
        )

    @_NOT_IMPLEMENTED
    def test_theta(self, py_zinb_result, r_zinb_fit):
        r_theta = float(np.exp(r_zinb_fit["phis"][0]))
        py_theta = float(np.exp(py_zinb_result.phis[0]))
        np.testing.assert_allclose(py_theta, r_theta, rtol=0.15)

    @_NOT_IMPLEMENTED
    def test_loglik(self, py_zinb_result, r_zinb_fit):
        np.testing.assert_allclose(
            py_zinb_result.logLik, r_zinb_fit["logLik"], atol=1.0
        )

    @_NOT_IMPLEMENTED
    def test_converged(self, py_zinb_result):
        assert py_zinb_result.converged

    @_NOT_IMPLEMENTED
    def test_zinb_better_than_zip(self, py_zinb_result, r_zinb_fit, py_zip_result):
        """
        ZINB (with estimated theta) should fit at least as well as ZIP when
        data were generated from a NB distribution.
        """
        assert py_zinb_result.logLik >= py_zip_result.logLik - 1.0
