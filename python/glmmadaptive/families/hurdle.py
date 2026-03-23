"""
Hurdle (two-part) family objects.

TODO: Full implementation pending. These stubs define the public API so that
the rest of the codebase can import them without errors.

See ``R/Fit_Funs.R`` (functions ``hurdle.poisson``,
``hurdle.negative.binomial``, ``hurdle.beta.fam``, ``hurdle.lognormal``)
for the R reference implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from glmmadaptive.families.base import BaseFamily


class _HurdleStub(BaseFamily):
    """Common stub for hurdle families."""

    has_zi = True  # hurdle families also need a zero-part component

    def linkinv(self, eta: NDArray) -> NDArray:  # type: ignore[override]
        raise NotImplementedError(f"{self.__class__.__name__} is not yet implemented")

    def variance(self, mu: NDArray) -> NDArray:
        raise NotImplementedError

    def mu_eta(self, eta: NDArray) -> NDArray:
        raise NotImplementedError

    def log_dens(self, y, eta, phis=None, eta_zi=None) -> NDArray:
        raise NotImplementedError


class HurdlePoisson(_HurdleStub):
    """Hurdle Poisson (stub — TODO)."""
    family = "hurdle_poisson"
    link = "log"
    n_phis = 0


class HurdleNegativeBinomial(_HurdleStub):
    """Hurdle Negative Binomial (stub — TODO)."""
    family = "hurdle_negative_binomial"
    link = "log"
    n_phis = 1


class HurdleBeta(_HurdleStub):
    """Hurdle Beta (stub — TODO)."""
    family = "hurdle_beta"
    link = "logit"
    n_phis = 1


class HurdleLogNormal(_HurdleStub):
    """Hurdle log-Normal (stub — TODO)."""
    family = "hurdle_lognormal"
    link = "log"
    n_phis = 1
