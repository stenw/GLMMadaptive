"""
Live R comparison tests for Poisson and NegativeBinomial random-intercept models.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def r_poisson_fit(r_env, sim_poisson_data):
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    r_data = pandas2ri.py2rpy(sim_poisson_data)
    ro.globalenv["py_data"] = r_data
    ro.r("""
        library(GLMMadaptive)
        fm_pois <- mixed_model(
            fixed  = y ~ x,
            random = ~ 1 | id,
            data   = py_data,
            family = poisson()
        )
        r_betas_pois  <- as.numeric(fixef(fm_pois))
        r_D_pois      <- as.numeric(fm_pois$D)
        r_loglik_pois <- as.numeric(logLik(fm_pois))
        r_bse_pois    <- sqrt(diag(vcov(fm_pois)))
    """)
    return {
        "betas": np.array(ro.r("r_betas_pois")),
        "D": float(ro.r("r_D_pois")[0]),
        "logLik": float(ro.r("r_loglik_pois")[0]),
        "bse": np.array(ro.r("r_bse_pois")),
    }


@pytest.fixture(scope="module")
def py_poisson_result(sim_poisson_data):
    from glmmadaptive import MixedModel
    from glmmadaptive.families import Poisson
    model = MixedModel(
        fixed="y ~ x",
        random="~ 1 | id",
        data=sim_poisson_data,
        family=Poisson(),
        control={"iter_em": 50, "verbose": False},
    )
    return model.fit()


class TestPoissonVsR:
    def test_betas(self, py_poisson_result, r_poisson_fit):
        np.testing.assert_allclose(
            py_poisson_result.params, r_poisson_fit["betas"], rtol=0.02
        )

    def test_D(self, py_poisson_result, r_poisson_fit):
        np.testing.assert_allclose(
            py_poisson_result.D[0, 0], r_poisson_fit["D"], rtol=0.05
        )

    def test_loglik(self, py_poisson_result, r_poisson_fit):
        np.testing.assert_allclose(
            py_poisson_result.logLik, r_poisson_fit["logLik"], atol=0.2
        )

    def test_converged(self, py_poisson_result):
        assert py_poisson_result.converged


# ---------------------------------------------------------------------------
# NegativeBinomial
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def r_nb_fit(r_env, sim_negbinom_data):
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    r_data = pandas2ri.py2rpy(sim_negbinom_data)
    ro.globalenv["py_data_nb"] = r_data
    ro.r("""
        library(GLMMadaptive)
        fm_nb <- mixed_model(
            fixed  = y ~ x,
            random = ~ 1 | id,
            data   = py_data_nb,
            family = negative.binomial()
        )
        r_betas_nb  <- as.numeric(fixef(fm_nb))
        r_D_nb      <- as.numeric(fm_nb$D)
        r_loglik_nb <- as.numeric(logLik(fm_nb))
        r_phis_nb   <- as.numeric(fm_nb$phis)
    """)
    return {
        "betas": np.array(ro.r("r_betas_nb")),
        "D": float(ro.r("r_D_nb")[0]),
        "logLik": float(ro.r("r_loglik_nb")[0]),
        "phis": np.array(ro.r("r_phis_nb")),
    }


@pytest.fixture(scope="module")
def py_nb_result(sim_negbinom_data):
    from glmmadaptive import MixedModel
    from glmmadaptive.families import NegativeBinomial
    model = MixedModel(
        fixed="y ~ x",
        random="~ 1 | id",
        data=sim_negbinom_data,
        family=NegativeBinomial(),
        control={"iter_em": 50, "verbose": False},
    )
    return model.fit()


class TestNegBinomVsR:
    def test_betas(self, py_nb_result, r_nb_fit):
        np.testing.assert_allclose(
            py_nb_result.params, r_nb_fit["betas"], rtol=0.05, atol=0.05
        )

    def test_loglik(self, py_nb_result, r_nb_fit):
        np.testing.assert_allclose(
            py_nb_result.logLik, r_nb_fit["logLik"], atol=1.0
        )
