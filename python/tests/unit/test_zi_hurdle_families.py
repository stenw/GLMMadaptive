"""
Unit tests for zero-inflated and hurdle family stubs.

All ZI/hurdle families are currently stubs: they expose the correct public
API (attributes ``has_zi``, ``family``, ``link``, ``n_phis``) but raise
``NotImplementedError`` for every computational method.

These tests also verify that ``MixedModel`` raises ``NotImplementedError``
when constructed with any of these families, since the fitting engine
does not yet support the zero-inflation component.
"""

import numpy as np
import pandas as pd
import pytest

from glmmadaptive.families.zero_inflated import ZIPoisson, ZINegativeBinomial, ZIBinomial
from glmmadaptive.families.hurdle import (
    HurdlePoisson,
    HurdleNegativeBinomial,
    HurdleBeta,
    HurdleLogNormal,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(0)
_ETA = _rng.normal(0, 1, 10)
_Y = _rng.poisson(2, 10).astype(float)


# ---------------------------------------------------------------------------
# Zero-inflated stubs
# ---------------------------------------------------------------------------

class TestZIFamilyAttributes:
    """Each ZI family must declare correct metadata attributes."""

    @pytest.mark.parametrize("cls, family_name, link, n_phis", [
        (ZIPoisson,          "zi_poisson",           "log",   0),
        (ZINegativeBinomial, "zi_negative_binomial", "log",   1),
        (ZIBinomial,         "zi_binomial",          "logit", 0),
    ])
    def test_has_zi_true(self, cls, family_name, link, n_phis):
        assert cls().has_zi is True

    @pytest.mark.parametrize("cls, family_name, link, n_phis", [
        (ZIPoisson,          "zi_poisson",           "log",   0),
        (ZINegativeBinomial, "zi_negative_binomial", "log",   1),
        (ZIBinomial,         "zi_binomial",          "logit", 0),
    ])
    def test_family_name(self, cls, family_name, link, n_phis):
        assert cls().family == family_name

    @pytest.mark.parametrize("cls, family_name, link, n_phis", [
        (ZIPoisson,          "zi_poisson",           "log",   0),
        (ZINegativeBinomial, "zi_negative_binomial", "log",   1),
        (ZIBinomial,         "zi_binomial",          "logit", 0),
    ])
    def test_link(self, cls, family_name, link, n_phis):
        assert cls().link == link

    @pytest.mark.parametrize("cls, family_name, link, n_phis", [
        (ZIPoisson,          "zi_poisson",           "log",   0),
        (ZINegativeBinomial, "zi_negative_binomial", "log",   1),
        (ZIBinomial,         "zi_binomial",          "logit", 0),
    ])
    def test_n_phis(self, cls, family_name, link, n_phis):
        assert cls().n_phis == n_phis


class TestZIFamilyNotImplemented:
    """Every computational method on a ZI stub must raise NotImplementedError."""

    @pytest.mark.parametrize("cls", [ZIPoisson, ZINegativeBinomial, ZIBinomial])
    def test_log_dens_raises(self, cls):
        with pytest.raises(NotImplementedError):
            cls().log_dens(_Y, _ETA)

    @pytest.mark.parametrize("cls", [ZIPoisson, ZINegativeBinomial, ZIBinomial])
    def test_linkinv_raises(self, cls):
        with pytest.raises(NotImplementedError):
            cls().linkinv(_ETA)

    @pytest.mark.parametrize("cls", [ZIPoisson, ZINegativeBinomial, ZIBinomial])
    def test_variance_raises(self, cls):
        mu = np.ones(10) * 0.5
        with pytest.raises(NotImplementedError):
            cls().variance(mu)

    @pytest.mark.parametrize("cls", [ZIPoisson, ZINegativeBinomial, ZIBinomial])
    def test_mu_eta_raises(self, cls):
        with pytest.raises(NotImplementedError):
            cls().mu_eta(_ETA)


# ---------------------------------------------------------------------------
# Hurdle stubs
# ---------------------------------------------------------------------------

class TestHurdleFamilyAttributes:
    """Each hurdle family must declare correct metadata attributes."""

    @pytest.mark.parametrize("cls, family_name, link, n_phis", [
        (HurdlePoisson,         "hurdle_poisson",          "log",   0),
        (HurdleNegativeBinomial,"hurdle_negative_binomial","log",   1),
        (HurdleBeta,            "hurdle_beta",             "logit", 1),
        (HurdleLogNormal,       "hurdle_lognormal",        "log",   1),
    ])
    def test_has_zi_true(self, cls, family_name, link, n_phis):
        assert cls().has_zi is True

    @pytest.mark.parametrize("cls, family_name, link, n_phis", [
        (HurdlePoisson,         "hurdle_poisson",          "log",   0),
        (HurdleNegativeBinomial,"hurdle_negative_binomial","log",   1),
        (HurdleBeta,            "hurdle_beta",             "logit", 1),
        (HurdleLogNormal,       "hurdle_lognormal",        "log",   1),
    ])
    def test_family_name(self, cls, family_name, link, n_phis):
        assert cls().family == family_name

    @pytest.mark.parametrize("cls, family_name, link, n_phis", [
        (HurdlePoisson,         "hurdle_poisson",          "log",   0),
        (HurdleNegativeBinomial,"hurdle_negative_binomial","log",   1),
        (HurdleBeta,            "hurdle_beta",             "logit", 1),
        (HurdleLogNormal,       "hurdle_lognormal",        "log",   1),
    ])
    def test_link(self, cls, family_name, link, n_phis):
        assert cls().link == link

    @pytest.mark.parametrize("cls, family_name, link, n_phis", [
        (HurdlePoisson,         "hurdle_poisson",          "log",   0),
        (HurdleNegativeBinomial,"hurdle_negative_binomial","log",   1),
        (HurdleBeta,            "hurdle_beta",             "logit", 1),
        (HurdleLogNormal,       "hurdle_lognormal",        "log",   1),
    ])
    def test_n_phis(self, cls, family_name, link, n_phis):
        assert cls().n_phis == n_phis


class TestHurdleFamilyNotImplemented:
    """Every computational method on a hurdle stub must raise NotImplementedError."""

    @pytest.mark.parametrize("cls", [
        HurdlePoisson, HurdleNegativeBinomial, HurdleBeta, HurdleLogNormal
    ])
    def test_log_dens_raises(self, cls):
        with pytest.raises(NotImplementedError):
            cls().log_dens(_Y, _ETA)

    @pytest.mark.parametrize("cls", [
        HurdlePoisson, HurdleNegativeBinomial, HurdleBeta, HurdleLogNormal
    ])
    def test_linkinv_raises(self, cls):
        with pytest.raises(NotImplementedError):
            cls().linkinv(_ETA)


# ---------------------------------------------------------------------------
# MixedModel construction with ZI/hurdle families raises NotImplementedError
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_data():
    """Tiny valid dataset for construction-only tests."""
    n, K = 10, 3
    rng = np.random.default_rng(1)
    ids = np.repeat(np.arange(n), K)
    time = np.tile(np.arange(K), n)
    y = rng.poisson(2, n * K)
    return pd.DataFrame({"id": ids, "time": time, "y": y})


class TestMixedModelRejectsZIFamilies:
    """
    MixedModel.__init__ must raise NotImplementedError for any family whose
    ``has_zi`` attribute is True, because the fitting engine does not yet
    support zero-inflation.
    """

    @pytest.mark.parametrize("cls", [
        ZIPoisson, ZINegativeBinomial, ZIBinomial,
        HurdlePoisson, HurdleNegativeBinomial, HurdleBeta, HurdleLogNormal,
    ])
    def test_raises_not_implemented(self, cls, minimal_data):
        from glmmadaptive import MixedModel
        with pytest.raises(NotImplementedError):
            MixedModel(
                fixed="y ~ time",
                random="~ 1 | id",
                data=minimal_data,
                family=cls(),
            )
