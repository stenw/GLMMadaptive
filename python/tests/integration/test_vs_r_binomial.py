"""
Live R comparison tests for the binary random-intercept model.

These tests require both ``rpy2`` and the R ``GLMMadaptive`` package to be
installed.  They are marked ``integration`` and are skipped otherwise.

Run with::

    pytest tests/integration/ -m integration -v

What is compared
----------------
* Fixed-effects coefficients β  (tolerance: 1%)
* Random-effects variance D[0,0]  (tolerance: 5%)
* Marginal log-likelihood  (tolerance: 0.1 units)
* Standard errors of β  (tolerance: 5%)
"""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def data_and_r_fit(r_env, sim_binary_data):
    """Return (data, r_results_dict) using the session-scoped rpy2 fixture."""
    from tests.conftest import r_fit_binary
    r_res = r_fit_binary(r_env, sim_binary_data)
    return sim_binary_data, r_res


@pytest.fixture(scope="module")
def py_result(data_and_r_fit):
    from glmmadaptive import MixedModel
    from glmmadaptive.families import Binomial

    data, _ = data_and_r_fit
    model = MixedModel(
        fixed="y ~ time",
        random="~ 1 | id",
        data=data,
        family=Binomial(),
        control={"iter_em": 50, "n_agh": 11, "verbose": False},
    )
    return model.fit()


class TestBinaryVsR:
    def test_betas_match_r(self, py_result, data_and_r_fit):
        _, r_res = data_and_r_fit
        np.testing.assert_allclose(
            py_result.params,
            r_res["betas"],
            rtol=0.01,
            err_msg="Fixed effects differ from R by >1%",
        )

    def test_D_matches_r(self, py_result, data_and_r_fit):
        _, r_res = data_and_r_fit
        np.testing.assert_allclose(
            py_result.D[0, 0],
            r_res["D"][0][0],
            rtol=0.05,
            err_msg="Random-effects variance differs from R by >5%",
        )

    def test_loglik_matches_r(self, py_result, data_and_r_fit):
        _, r_res = data_and_r_fit
        np.testing.assert_allclose(
            py_result.logLik,
            r_res["logLik"],
            atol=0.1,
            err_msg="logLik differs from R by >0.1",
        )

    def test_bse_matches_r(self, py_result, data_and_r_fit):
        _, r_res = data_and_r_fit
        np.testing.assert_allclose(
            py_result.bse,
            r_res["bse"],
            rtol=0.05,
            err_msg="Standard errors differ from R by >5%",
        )

    def test_converged(self, py_result):
        assert py_result.converged, "Python model should converge"
