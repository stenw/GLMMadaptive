"""
Live R comparison tests for predict(), ranef(), and fitted().

These tests require rpy2 and the R GLMMadaptive package.  They are skipped
automatically when R is unavailable.

Covered comparisons:
* predict(type_="mean_subject")   vs R  predict(fm, type="mean_subject")
* predict(type_="subject_specific") vs R predict(fm, type="subject_specific")
* ranef()                          vs R  ranef(fm)
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Binary RI: R fixture and Python result
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def r_binary_dynamic(r_env, sim_binary_data):
    """
    Fit the binary RI model in R and return predictions + random effects.
    """
    import rpy2.robjects as ro
    from tests.conftest import _df_to_r

    r_data = _df_to_r(sim_binary_data)
    ro.globalenv["py_data_bin"] = r_data
    ro.r("""
        library(GLMMadaptive)
        fm_bin_dyn <- mixed_model(
            fixed  = y ~ time,
            random = ~ 1 | id,
            data   = py_data_bin,
            family = binomial()
        )
        r_preds_ms_bin  <- as.numeric(predict(fm_bin_dyn, type = "mean_subject"))
        r_preds_ss_bin  <- as.numeric(predict(fm_bin_dyn, type = "subject_specific"))
        r_ranef_bin     <- as.numeric(ranef(fm_bin_dyn)[, 1])
        r_loglik_bin    <- as.numeric(logLik(fm_bin_dyn))
    """)
    return {
        "predictions_mean_subject":    np.array(ro.r("r_preds_ms_bin")),
        "predictions_subject_specific":np.array(ro.r("r_preds_ss_bin")),
        "ranef":                       np.array(ro.r("r_ranef_bin")),
        "logLik":                      float(ro.r("r_loglik_bin")[0]),
    }


@pytest.fixture(scope="module")
def py_binary_dynamic(sim_binary_data):
    from glmmadaptive import MixedModel
    from glmmadaptive.families import Binomial
    return MixedModel(
        fixed="y ~ time",
        random="~ 1 | id",
        data=sim_binary_data,
        family=Binomial(),
        control={"iter_em": 50, "verbose": False},
    ).fit()


class TestBinaryDynamicVsR:

    def test_predict_mean_subject(self, py_binary_dynamic, r_binary_dynamic):
        np.testing.assert_allclose(
            py_binary_dynamic.predict(type_="mean_subject"),
            r_binary_dynamic["predictions_mean_subject"],
            rtol=0.05, atol=0.02,
        )

    def test_predict_subject_specific(self, py_binary_dynamic, r_binary_dynamic):
        np.testing.assert_allclose(
            py_binary_dynamic.predict(type_="subject_specific"),
            r_binary_dynamic["predictions_subject_specific"],
            rtol=0.10, atol=0.05,
        )

    def test_ranef(self, py_binary_dynamic, r_binary_dynamic):
        py_re = py_binary_dynamic.ranef().values.ravel()
        np.testing.assert_allclose(
            py_re,
            r_binary_dynamic["ranef"],
            rtol=0.15, atol=0.10,
        )

    def test_loglik(self, py_binary_dynamic, r_binary_dynamic):
        np.testing.assert_allclose(
            py_binary_dynamic.logLik,
            r_binary_dynamic["logLik"],
            atol=0.2,
        )

    def test_converged(self, py_binary_dynamic):
        assert py_binary_dynamic.converged

    def test_fitted_equals_predict_training(self, py_binary_dynamic):
        np.testing.assert_allclose(
            py_binary_dynamic.fitted(),
            py_binary_dynamic.predict(),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Poisson RI: R fixture and Python result
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def r_poisson_dynamic(r_env, sim_poisson_data):
    import rpy2.robjects as ro
    from tests.conftest import _df_to_r

    r_data = _df_to_r(sim_poisson_data)
    ro.globalenv["py_data_pois"] = r_data
    ro.r("""
        library(GLMMadaptive)
        fm_pois_dyn <- mixed_model(
            fixed  = y ~ x,
            random = ~ 1 | id,
            data   = py_data_pois,
            family = poisson()
        )
        r_preds_ms_pois  <- as.numeric(predict(fm_pois_dyn, type = "mean_subject"))
        r_preds_ss_pois  <- as.numeric(predict(fm_pois_dyn, type = "subject_specific"))
        r_ranef_pois     <- as.numeric(ranef(fm_pois_dyn)[, 1])
        r_loglik_pois    <- as.numeric(logLik(fm_pois_dyn))
    """)
    return {
        "predictions_mean_subject":    np.array(ro.r("r_preds_ms_pois")),
        "predictions_subject_specific":np.array(ro.r("r_preds_ss_pois")),
        "ranef":                       np.array(ro.r("r_ranef_pois")),
        "logLik":                      float(ro.r("r_loglik_pois")[0]),
    }


@pytest.fixture(scope="module")
def py_poisson_dynamic(sim_poisson_data):
    from glmmadaptive import MixedModel
    from glmmadaptive.families import Poisson
    return MixedModel(
        fixed="y ~ x",
        random="~ 1 | id",
        data=sim_poisson_data,
        family=Poisson(),
        control={"iter_em": 50, "verbose": False},
    ).fit()


class TestPoissonDynamicVsR:

    def test_predict_mean_subject(self, py_poisson_dynamic, r_poisson_dynamic):
        np.testing.assert_allclose(
            py_poisson_dynamic.predict(type_="mean_subject"),
            r_poisson_dynamic["predictions_mean_subject"],
            rtol=0.05, atol=0.05,
        )

    def test_predict_subject_specific(self, py_poisson_dynamic, r_poisson_dynamic):
        np.testing.assert_allclose(
            py_poisson_dynamic.predict(type_="subject_specific"),
            r_poisson_dynamic["predictions_subject_specific"],
            rtol=0.10, atol=0.10,
        )

    def test_ranef(self, py_poisson_dynamic, r_poisson_dynamic):
        py_re = py_poisson_dynamic.ranef().values.ravel()
        np.testing.assert_allclose(
            py_re,
            r_poisson_dynamic["ranef"],
            rtol=0.15, atol=0.10,
        )

    def test_loglik(self, py_poisson_dynamic, r_poisson_dynamic):
        np.testing.assert_allclose(
            py_poisson_dynamic.logLik,
            r_poisson_dynamic["logLik"],
            atol=0.5,
        )

    def test_converged(self, py_poisson_dynamic):
        assert py_poisson_dynamic.converged
