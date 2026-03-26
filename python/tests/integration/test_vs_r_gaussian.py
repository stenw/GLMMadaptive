"""
Live R comparison tests for the Gaussian random-intercept model.

Requires rpy2 and the R GLMMadaptive package (skipped otherwise).

The R package rejects ``gaussian()`` and redirects users to lme4/nlme, so we
compare against ``students.t(df=1e6)`` which is numerically identical to
Gaussian but accepted by GLMMadaptive R.

Covered comparisons:
* fixef (betas)             — rtol=0.02
* phis  (log sigma)         — rtol=0.02
* D[0,0] (RE variance)      — rtol=0.10
* logLik                    — atol=0.5
* predict(type="mean_subject") — rtol=0.02
* ranef                     — rtol=0.10
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sim_gaussian_data():
    """Simulated longitudinal Gaussian data (same seed as R fixture)."""
    rng = np.random.default_rng(42)
    n, K = 150, 5
    ids = np.repeat(np.arange(n), K)
    time = np.tile(np.arange(K), n).astype(float)
    b = rng.normal(0, np.sqrt(0.8), n)
    eps = rng.normal(0, 1.2, n * K)
    y = 2.0 + 0.5 * time + b[ids] + eps
    return pd.DataFrame({"id": ids, "time": time, "y": y})


@pytest.fixture(scope="module")
def r_gaussian(r_env, sim_gaussian_data):
    """Fit students.t(df=1e6) in R (≡ Gaussian) and return key quantities."""
    import rpy2.robjects as ro
    from tests.conftest import _df_to_r

    r_data = _df_to_r(sim_gaussian_data)
    ro.globalenv["py_data_gauss"] = r_data
    ro.r("""
        library(GLMMadaptive)
        fm_gauss_r <- mixed_model(
            fixed  = y ~ time,
            random = ~ 1 | id,
            data   = py_data_gauss,
            family = students.t(df = 1e6),
            n_phis = 1
        )
        r_betas_gauss   <- as.numeric(fixef(fm_gauss_r))
        r_phis_gauss    <- as.numeric(fm_gauss_r$phis)
        r_D11_gauss     <- as.numeric(fm_gauss_r$D[1, 1])
        r_loglik_gauss  <- as.numeric(logLik(fm_gauss_r))
        r_preds_gauss   <- as.numeric(predict(fm_gauss_r, type = "mean_subject"))
        r_ranef_gauss   <- as.numeric(ranef(fm_gauss_r)[, 1])
    """)
    return {
        "betas":               np.array(ro.r("r_betas_gauss")),
        "phis":                np.array(ro.r("r_phis_gauss")),
        "D11":                 float(ro.r("r_D11_gauss")[0]),
        "logLik":              float(ro.r("r_loglik_gauss")[0]),
        "predictions_mean_subject": np.array(ro.r("r_preds_gauss")),
        "ranef":               np.array(ro.r("r_ranef_gauss")),
    }


@pytest.fixture(scope="module")
def py_gaussian(sim_gaussian_data):
    from glmmadaptive import MixedModel
    from glmmadaptive.families import Gaussian
    return MixedModel(
        fixed="y ~ time",
        random="~ 1 | id",
        data=sim_gaussian_data,
        family=Gaussian(),
        control={"iter_em": 50, "verbose": False},
    ).fit()


class TestGaussianVsR:
    def test_converged(self, py_gaussian):
        assert py_gaussian.converged

    def test_betas_close(self, r_gaussian, py_gaussian):
        assert_allclose(
            py_gaussian.fixef().values, r_gaussian["betas"],
            rtol=0.02, err_msg="betas differ from R students.t(df=1e6)",
        )

    def test_phis_close(self, r_gaussian, py_gaussian):
        """log(sigma) should agree — both parameterise sigma = exp(phis[0])."""
        assert_allclose(
            py_gaussian.phis[0], r_gaussian["phis"][0],
            rtol=0.02, err_msg="phis[0] = log(sigma) differs from R",
        )

    def test_D_close(self, r_gaussian, py_gaussian):
        assert_allclose(
            py_gaussian.D[0, 0], r_gaussian["D11"],
            rtol=0.10, err_msg="D[0,0] differs from R",
        )

    def test_loglik_close(self, r_gaussian, py_gaussian):
        assert_allclose(
            py_gaussian.logLik, r_gaussian["logLik"],
            atol=0.5, err_msg="logLik differs from R",
        )

    def test_predictions_close(self, r_gaussian, py_gaussian, sim_gaussian_data):
        py_preds = py_gaussian.predict(
            newdata=sim_gaussian_data, type_="mean_subject"
        )
        assert_allclose(
            py_preds, r_gaussian["predictions_mean_subject"],
            rtol=0.02, err_msg="mean_subject predictions differ from R",
        )

    def test_ranef_close(self, r_gaussian, py_gaussian):
        re_df = py_gaussian.ranef()
        # Sort to match R's output ordering; use atol because near-zero REs
        # have unbounded relative error even when numerically identical
        r_re_sorted = np.sort(r_gaussian["ranef"])
        py_re_sorted = np.sort(re_df.iloc[:, 0].values)
        assert_allclose(
            py_re_sorted, r_re_sorted,
            rtol=0.10, atol=1e-3,
            err_msg="random effects (sorted) differ from R",
        )

    def test_fitted_equals_predict(self, py_gaussian, sim_gaussian_data):
        """fitted() should equal predict(type_='mean_subject') on training data."""
        preds = py_gaussian.predict(newdata=sim_gaussian_data, type_="mean_subject")
        fitted = py_gaussian.fitted()
        assert_allclose(preds, fitted, rtol=1e-6)
