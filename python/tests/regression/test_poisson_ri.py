"""
Regression tests for Poisson random-intercept GLMM.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_fixture(name):
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"Fixture '{name}.json' not found — run generate_r_fixtures.R")
    with open(path) as f:
        return json.load(f)


@pytest.mark.regression
class TestPoissonRI:
    @pytest.fixture(scope="class")
    def ref(self):
        return load_fixture("poisson_ri")

    @pytest.fixture(scope="class")
    def result(self, ref):
        from glmmadaptive import MixedModel
        from glmmadaptive.families import Poisson

        data = pd.DataFrame(ref["data"])
        model = MixedModel(
            fixed="y ~ x",
            random="~ 1 | id",
            data=data,
            family=Poisson(),
            control={"iter_em": 50, "verbose": False},
        )
        return model.fit()

    def test_loglik_close_to_r(self, result, ref):
        assert_allclose(result.logLik, ref["logLik"], atol=0.5)

    def test_betas_close_to_r(self, result, ref):
        r_betas = np.array(ref["betas"])
        assert_allclose(result.params, r_betas, rtol=0.05, atol=0.05)

    def test_converged(self, result):
        assert result.converged


@pytest.mark.regression
class TestNegBinomRI:
    @pytest.fixture(scope="class")
    def ref(self):
        return load_fixture("negbinom_ri")

    @pytest.fixture(scope="class")
    def result(self, ref):
        from glmmadaptive import MixedModel
        from glmmadaptive.families import NegativeBinomial

        data = pd.DataFrame(ref["data"])
        model = MixedModel(
            fixed="y ~ x",
            random="~ 1 | id",
            data=data,
            family=NegativeBinomial(),
            control={"iter_em": 50, "verbose": False},
        )
        return model.fit()

    def test_loglik_close_to_r(self, result, ref):
        assert_allclose(result.logLik, ref["logLik"], atol=1.0)

    def test_betas_close_to_r(self, result, ref):
        r_betas = np.array(ref["betas"])
        assert_allclose(result.params, r_betas, rtol=0.10, atol=0.10)
