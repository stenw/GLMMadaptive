"""
Regression tests for penalized fixed-effects (Student's-t penalty).

These tests verify that:
1. Penalized fits converge and match the pre-saved Python reference fixture.
2. The penalty induces shrinkage relative to the unpenalized fit.
3. Custom penalty hyperparameters are accepted and produce valid fits.
4. Setting penalized=False leaves the fit unchanged from the baseline.
5. The MixedModel.__init__ argument parsing raises correctly on bad inputs.

Fixture: ``tests/fixtures/penalized_binary_ri.json`` (binary random-intercept,
default penalty pen_mu=0, pen_sigma=1, pen_df=3, same data as binary_ri.json).
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
        pytest.skip(f"Fixture '{name}.json' not found")
    with open(path) as f:
        return json.load(f)


@pytest.mark.regression
class TestPenalizedBinaryRI:
    """Penalized binary random-intercept model — regression against fixture."""

    @pytest.fixture(scope="class")
    def ref(self):
        return load_fixture("penalized_binary_ri")

    @pytest.fixture(scope="class")
    def data(self):
        raw = load_fixture("binary_ri")
        return pd.DataFrame(raw["data"])

    @pytest.fixture(scope="class")
    def result(self, data):
        from glmmadaptive import MixedModel
        from glmmadaptive.families import Binomial

        model = MixedModel(
            fixed="y ~ time",
            random="~ 1 | id",
            data=data,
            family=Binomial(),
            penalized=True,
            control={"iter_em": 50, "verbose": False},
        )
        return model.fit()

    def test_converged(self, result):
        assert result.converged, "Penalized model should converge"

    def test_loglik_matches_fixture(self, result, ref):
        assert_allclose(result.logLik, ref["logLik"], atol=0.5)

    def test_betas_match_fixture(self, result, ref):
        r_betas = np.array(ref["betas"])
        assert_allclose(result.params, r_betas, rtol=0.05, atol=0.05)

    def test_D_matches_fixture(self, result, ref):
        assert_allclose(result.D[0, 0], ref["D"][0][0], rtol=0.10)

    def test_bse_positive(self, result):
        assert np.all(result.bse > 0), "Standard errors should be positive"

    def test_penalized_stored_in_results(self, result):
        """The penalized dict should be stored on the results object."""
        assert result.penalized["penalized"] is True
        assert result.penalized["pen_mu"] == 0.0
        assert result.penalized["pen_sigma"] == 1.0
        assert result.penalized["pen_df"] == 3.0


@pytest.mark.regression
class TestPenaltyShrinkage:
    """The Student's-t penalty should shrink betas toward pen_mu=0."""

    @pytest.fixture(scope="class")
    def data(self):
        raw = load_fixture("binary_ri")
        return pd.DataFrame(raw["data"])

    @pytest.fixture(scope="class")
    def result_unpen(self, data):
        from glmmadaptive import MixedModel
        from glmmadaptive.families import Binomial

        return MixedModel(
            "y ~ time", "~ 1 | id", data=data, family=Binomial(),
            penalized=False, control={"iter_em": 50},
        ).fit()

    @pytest.fixture(scope="class")
    def result_pen(self, data):
        from glmmadaptive import MixedModel
        from glmmadaptive.families import Binomial

        return MixedModel(
            "y ~ time", "~ 1 | id", data=data, family=Binomial(),
            penalized=True, control={"iter_em": 50},
        ).fit()

    def test_penalty_shrinks_betas(self, result_unpen, result_pen):
        """Penalized betas should be closer to zero than unpenalized betas."""
        assert np.sum(result_pen.params ** 2) < np.sum(result_unpen.params ** 2), (
            "Penalized betas should have smaller L2 norm than unpenalized"
        )

    def test_penalty_lowers_loglik(self, result_unpen, result_pen):
        """Penalized logLik is the penalized objective, hence different from unpenalized."""
        # The penalized logLik includes the prior term, so it differs from the marginal ll
        assert result_pen.logLik != result_unpen.logLik

    def test_false_gives_same_as_baseline(self, result_unpen, data):
        """penalized=False should produce the same fit as the default (no penalty)."""
        from glmmadaptive import MixedModel
        from glmmadaptive.families import Binomial

        res2 = MixedModel(
            "y ~ time", "~ 1 | id", data=data, family=Binomial(),
            control={"iter_em": 50},
        ).fit()
        assert_allclose(result_unpen.logLik, res2.logLik, atol=1e-2)
        assert_allclose(result_unpen.params, res2.params, rtol=0.01)


@pytest.mark.regression
class TestPenalizedCustomHyperparams:
    """Verify that custom penalty hyperparameters work correctly."""

    @pytest.fixture(scope="class")
    def data(self):
        raw = load_fixture("binary_ri")
        return pd.DataFrame(raw["data"])

    def test_custom_dict_converges(self, data):
        from glmmadaptive import MixedModel
        from glmmadaptive.families import Binomial

        res = MixedModel(
            "y ~ time", "~ 1 | id", data=data, family=Binomial(),
            penalized={"pen_mu": 0.0, "pen_sigma": 2.0, "pen_df": 5.0},
            control={"iter_em": 50},
        ).fit()
        assert res.converged
        assert np.all(np.isfinite(res.params))

    def test_large_df_approaches_ridge(self, data):
        """pen_df=1e6 should behave like a normal (ridge) penalty — still shrinks."""
        from glmmadaptive import MixedModel
        from glmmadaptive.families import Binomial

        res_ridge = MixedModel(
            "y ~ time", "~ 1 | id", data=data, family=Binomial(),
            penalized={"pen_df": 1e6},
            control={"iter_em": 50},
        ).fit()
        res_unpen = MixedModel(
            "y ~ time", "~ 1 | id", data=data, family=Binomial(),
            penalized=False, control={"iter_em": 50},
        ).fit()
        # Ridge shrinks toward 0
        assert np.sum(res_ridge.params ** 2) < np.sum(res_unpen.params ** 2)

    def test_partial_dict_fills_defaults(self, data):
        """Providing only pen_df should keep pen_mu=0 and pen_sigma=1 as defaults."""
        from glmmadaptive import MixedModel
        from glmmadaptive.families import Binomial

        model = MixedModel(
            "y ~ time", "~ 1 | id", data=data, family=Binomial(),
            penalized={"pen_df": 10.0},
        )
        assert model._penalized["pen_mu"] == 0.0
        assert model._penalized["pen_sigma"] == 1.0
        assert model._penalized["pen_df"] == 10.0


@pytest.mark.regression
class TestPenalizedArgumentValidation:
    """MixedModel should raise on invalid penalized arguments."""

    @pytest.fixture(scope="class")
    def data(self):
        raw = load_fixture("binary_ri")
        return pd.DataFrame(raw["data"])

    def test_invalid_type_raises(self, data):
        from glmmadaptive import MixedModel
        from glmmadaptive.families import Binomial

        with pytest.raises(TypeError, match="bool or dict"):
            MixedModel("y ~ time", "~ 1 | id", data=data, family=Binomial(),
                       penalized="strong")

    def test_unknown_key_raises(self, data):
        from glmmadaptive import MixedModel
        from glmmadaptive.families import Binomial

        with pytest.raises(ValueError, match="Unknown 'penalized' keys"):
            MixedModel("y ~ time", "~ 1 | id", data=data, family=Binomial(),
                       penalized={"lambda": 1.0})
