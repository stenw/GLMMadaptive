"""
Regression tests for binary random-intercept GLMM.

These tests compare the Python fitting results against pre-saved R reference
outputs stored in ``tests/fixtures/binary_ri.json``.

To regenerate fixtures::

    Rscript tests/fixtures/generate_r_fixtures.R

The fixtures are checked into the repository so these tests run without R.
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
class TestBinaryRI:
    @pytest.fixture(scope="class")
    def ref(self):
        return load_fixture("binary_ri")

    @pytest.fixture(scope="class")
    def result(self, ref):
        from glmmadaptive import MixedModel
        from glmmadaptive.families import Binomial

        data = pd.DataFrame(ref["data"])
        model = MixedModel(
            fixed="y ~ time",
            random="~ 1 | id",
            data=data,
            family=Binomial(),
            control={"iter_em": 50, "verbose": False},
        )
        return model.fit()

    def test_loglik_close_to_r(self, result, ref):
        """logLik should match R within 0.5 units."""
        assert_allclose(result.logLik, ref["logLik"], atol=0.5)

    def test_betas_close_to_r(self, result, ref):
        """Fixed-effects coefficients should match R within 5%."""
        r_betas = np.array(ref["betas"])
        assert_allclose(result.params, r_betas, rtol=0.05, atol=0.05)

    def test_D_close_to_r(self, result, ref):
        """Random-effects variance should match R within 10%."""
        r_D_val = ref["D"][0][0]
        assert_allclose(result.D[0, 0], r_D_val, rtol=0.10)

    def test_converged(self, result):
        assert result.converged, "Model should have converged"

    def test_bse_positive(self, result):
        assert np.all(result.bse > 0)

    def test_confint_covers_betas(self, result, ref):
        """95% CI should cover the R point estimates."""
        ci = result.confint()
        r_betas = np.array(ref["betas"])
        lo = ci.iloc[:, 0].values
        hi = ci.iloc[:, 1].values
        assert np.all(lo < r_betas) and np.all(r_betas < hi)
