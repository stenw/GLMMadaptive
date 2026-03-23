"""
Zero-inflated family objects.

TODO: Full implementation pending. These stubs define the public API so that
the rest of the codebase can import them without errors.

See ``R/Fit_Funs.R`` (functions ``zi.poisson``, ``zi.negative.binomial``,
``zi.binomial``) for the R reference implementation.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from glmmadaptive.families.base import BaseFamily


class _ZIStub(BaseFamily):
    """Common stub for all zero-inflated families."""

    has_zi = True

    def linkinv(self, eta: NDArray) -> NDArray:  # type: ignore[override]
        raise NotImplementedError(f"{self.__class__.__name__} is not yet implemented")

    def variance(self, mu: NDArray) -> NDArray:
        raise NotImplementedError

    def mu_eta(self, eta: NDArray) -> NDArray:
        raise NotImplementedError

    def log_dens(self, y, eta, phis=None, eta_zi=None) -> NDArray:
        raise NotImplementedError


class ZIPoisson(_ZIStub):
    """Zero-inflated Poisson (stub — TODO)."""
    family = "zi_poisson"
    link = "log"
    n_phis = 0


class ZINegativeBinomial(_ZIStub):
    """Zero-inflated Negative Binomial (stub — TODO)."""
    family = "zi_negative_binomial"
    link = "log"
    n_phis = 1


class ZIBinomial(_ZIStub):
    """Zero-inflated Binomial (stub — TODO)."""
    family = "zi_binomial"
    link = "logit"
    n_phis = 0
