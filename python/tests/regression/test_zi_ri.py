"""
Regression tests for zero-inflated random-intercept models.

These tests compare the Python implementation against pre-saved R reference
outputs in ``tests/fixtures/zi_poisson_ri.json`` and
``tests/fixtures/zi_negbinom_ri.json``.

To regenerate fixtures::

    Rscript tests/fixtures/generate_r_fixtures.R
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

def load_fixture(name: str) -> dict:
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"Fixture '{name}.json' not found — run generate_r_fixtures.R")
    with open(path) as f:
        return json.load(f)

def _fit_zip(data, control=None):
    from glmmadaptive import MixedModel
    from glmmadaptive.families import ZIPoisson
    ctrl = {"iter_em": 50, "verbose": False}
    if control:
        ctrl.update(control)
    return MixedModel(
        fixed="y ~ time",
        random="~ 1 | id",
        data=data,
        family=ZIPoisson(),
        zi_fixed="~ 1",
        control=ctrl,
    ).fit()

def _fit_zinb(data, control=None):
    from glmmadaptive import MixedModel
    from glmmadaptive.families import ZINegativeBinomial
    ctrl = {"iter_em": 50, "verbose": False}
    if control:
        ctrl.update(control)
    return MixedModel(
        fixed="y ~ time",
        random="~ 1 | id",
        data=data,
        family=ZINegativeBinomial(),
        zi_fixed="~ 1",
        control=ctrl,
    ).fit()

# ---------------------------------------------------------------------------
# Zero-inflated Poisson
# ---------------------------------------------------------------------------

@pytest.mark.regression
class TestZIPoissonRI:
    """Regression tests for ZI Poisson random-intercept model."""

    @pytest.fixture(scope="class")
    def ref(self):
        return load_fixture("zi_poisson_ri")

    
    def test_loglik_close_to_r(self, ref):
        data = pd.DataFrame(ref["data"])
        result = _fit_zip(data)
        assert_allclose(result.logLik, ref["logLik"], atol=0.5)

    
    def test_betas_close_to_r(self, ref):
        """Count-part fixed effects should match R within 5%."""
        data = pd.DataFrame(ref["data"])
        result = _fit_zip(data)
        r_betas = np.array(ref["betas"])
        assert_allclose(result.params, r_betas, rtol=0.05, atol=0.05)

    
    def test_gammas_close_to_r(self, ref):
        """Zero-part fixed effects (gammas) should match R within 10%."""
        data = pd.DataFrame(ref["data"])
        result = _fit_zip(data)
        r_gammas = np.array(ref["gammas"])
        assert_allclose(result.gammas, r_gammas, rtol=0.10, atol=0.10)

    
    def test_D_close_to_r(self, ref):
        """Random-effects variance should match R within 10%."""
        data = pd.DataFrame(ref["data"])
        result = _fit_zip(data)
        r_D = ref["D"][0][0]
        assert_allclose(result.D[0, 0], r_D, rtol=0.10)

    
    def test_converged(self, ref):
        data = pd.DataFrame(ref["data"])
        result = _fit_zip(data)
        assert result.converged

    
    def test_bse_positive(self, ref):
        data = pd.DataFrame(ref["data"])
        result = _fit_zip(data)
        assert np.all(result.bse > 0)


    def test_gamma_se_correct_hessian_block(self, ref):
        """SE in summary table must match vcov(zero_part), not D_chol block."""
        data = pd.DataFrame(ref["data"])
        result = _fit_zip(data)
        # vcov(zero_part) uses _param_slices() — known correct
        vcov_g = result.vcov(parm="zero_part")
        se_expected = np.sqrt(np.diag(vcov_g))
        # confint routes through vcov() too; cross-check via confint
        ci = result.confint(parm="zero_part", level=0.95)
        se_ci = (ci.iloc[:, 2].values - ci.iloc[:, 0].values) / (2 * 1.96)
        assert_allclose(se_ci, se_expected, rtol=1e-3)
        # The bug was: vcov_g used [n_betas : n_betas+n_g] which is the D block.
        # Verify the correct SE differs from the wrong block's SE.
        n_b = len(result.params)
        n_g = len(result.gammas)
        vcov_all = np.linalg.inv(result._Hessian)
        se_wrong = np.sqrt(np.maximum(np.diag(vcov_all[n_b : n_b + n_g, n_b : n_b + n_g]), 0.0))
        # They must NOT be the same (D_chol variance ≠ gamma variance)
        assert not np.allclose(se_wrong, se_expected, rtol=1e-3), (
            "Bug not triggered: wrong-block SEs match correct SEs — "
            "model may have no D params between betas and gammas."
        )


    def test_df_model_includes_gammas(self, ref):
        """df_model must count zero-part (gamma) parameters."""
        data = pd.DataFrame(ref["data"])
        result = _fit_zip(data)
        q = result.D.shape[0]
        n_D = q  # diagonal by default
        n_phis = len(result.phis) if result.phis is not None else 0
        n_gammas = len(result.gammas) if result.gammas is not None else 0
        assert n_gammas > 0, "ZI model should have gamma parameters"
        expected = len(result.params) + n_D + n_phis + n_gammas
        assert result.df_model == expected


    def test_aic_uses_correct_df(self, ref):
        """AIC = -2*logLik + 2*df_model (with gammas counted)."""
        data = pd.DataFrame(ref["data"])
        result = _fit_zip(data)
        expected_aic = -2.0 * result.logLik + 2.0 * result.df_model
        assert result.aic == pytest.approx(expected_aic)


    def test_anova_zip_with_zi_random(self, ref):
        """LRT between ZIP (no ZI RE) and ZIP with ZI random intercept."""
        from glmmadaptive import MixedModel
        from glmmadaptive.families import ZIPoisson
        from glmmadaptive.results import MixModResults

        data = pd.DataFrame(ref["data"])
        ctrl = {"iter_em": 50, "verbose": False}

        fm1 = MixedModel(
            fixed="y ~ time", random="~ 1 | id", data=data,
            family=ZIPoisson(), zi_fixed="~ 1", control=ctrl,
        ).fit()
        fm2 = MixedModel(
            fixed="y ~ time", random="~ 1 | id", data=data,
            family=ZIPoisson(), zi_fixed="~ 1", zi_random="~ 1 | id", control=ctrl,
        ).fit()

        anova_table = MixModResults.anova(fm1, fm2)
        assert anova_table.shape[0] == 2
        # fm2 (with ZI random effect) should fit at least as well as fm1
        assert fm2.logLik >= fm1.logLik - 1e-6

# ---------------------------------------------------------------------------
# Zero-inflated Negative Binomial
# ---------------------------------------------------------------------------

@pytest.mark.regression
class TestZINegBinomRI:
    """Regression tests for ZI Negative Binomial random-intercept model."""

    @pytest.fixture(scope="class")
    def ref(self):
        return load_fixture("zi_negbinom_ri")

    
    def test_loglik_close_to_r(self, ref):
        data = pd.DataFrame(ref["data"])
        result = _fit_zinb(data)
        assert_allclose(result.logLik, ref["logLik"], atol=1.0)

    
    def test_betas_close_to_r(self, ref):
        data = pd.DataFrame(ref["data"])
        result = _fit_zinb(data)
        r_betas = np.array(ref["betas"])
        assert_allclose(result.params, r_betas, rtol=0.10, atol=0.10)

    
    def test_gammas_close_to_r(self, ref):
        data = pd.DataFrame(ref["data"])
        result = _fit_zinb(data)
        r_gammas = np.array(ref["gammas"])
        assert_allclose(result.gammas, r_gammas, rtol=0.10, atol=0.10)

    
    def test_theta_close_to_r(self, ref):
        """Over-dispersion parameter theta = exp(phis[0]) should match R."""
        data = pd.DataFrame(ref["data"])
        result = _fit_zinb(data)
        r_theta = float(np.exp(ref["phis"][0]))
        py_theta = float(np.exp(result.phis[0]))
        assert_allclose(py_theta, r_theta, rtol=0.15)


    def test_converged(self, ref):
        data = pd.DataFrame(ref["data"])
        result = _fit_zinb(data)
        assert result.converged


    def test_gamma_se_correct_hessian_block(self, ref):
        """SE in summary table must match vcov(zero_part), not D_chol/phis block."""
        data = pd.DataFrame(ref["data"])
        result = _fit_zinb(data)
        vcov_g = result.vcov(parm="zero_part")
        se_expected = np.sqrt(np.diag(vcov_g))
        ci = result.confint(parm="zero_part", level=0.95)
        se_ci = (ci.iloc[:, 2].values - ci.iloc[:, 0].values) / (2 * 1.96)
        assert_allclose(se_ci, se_expected, rtol=1e-3)
        # Verify the correct block differs from the wrong one (n_betas offset, not full offset)
        n_b = len(result.params)
        n_g = len(result.gammas)
        vcov_all = np.linalg.inv(result._Hessian)
        se_wrong = np.sqrt(np.maximum(np.diag(vcov_all[n_b : n_b + n_g, n_b : n_b + n_g]), 0.0))
        assert not np.allclose(se_wrong, se_expected, rtol=1e-3), (
            "Bug not triggered: wrong-block SEs match correct SEs."
        )


    def test_df_model_includes_gammas(self, ref):
        """df_model must count zero-part (gamma) parameters."""
        data = pd.DataFrame(ref["data"])
        result = _fit_zinb(data)
        q = result.D.shape[0]
        n_D = q  # diagonal by default
        n_phis = len(result.phis) if result.phis is not None else 0
        n_gammas = len(result.gammas) if result.gammas is not None else 0
        assert n_gammas > 0, "ZI model should have gamma parameters"
        expected = len(result.params) + n_D + n_phis + n_gammas
        assert result.df_model == expected


    def test_aic_uses_correct_df(self, ref):
        """AIC = -2*logLik + 2*df_model (with gammas counted)."""
        data = pd.DataFrame(ref["data"])
        result = _fit_zinb(data)
        expected_aic = -2.0 * result.logLik + 2.0 * result.df_model
        assert result.aic == pytest.approx(expected_aic)
