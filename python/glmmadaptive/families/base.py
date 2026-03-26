"""
Abstract base class for GLMMadaptive family objects.

Mirrors the R family list structure used throughout GLMMadaptive.
Every family must implement ``log_dens``, ``linkinv``, ``variance``, and
``mu_eta``.  Score functions (``score_eta``, ``score_phis``) are optional —
if not overridden, the fitting engine falls back to finite-difference
approximations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class BaseFamily(ABC):
    """
    Abstract base class for response-distribution families.

    Attributes
    ----------
    family : str
        Distribution name (e.g. ``"binomial"``).
    link : str
        Link function name (e.g. ``"logit"``).
    n_phis : int
        Number of extra dispersion/shape parameters (0 for binomial/Poisson).
    has_zi : bool
        Whether the family supports a zero-inflation component.
    """

    family: str = "base"
    link: str = "identity"
    n_phis: int = 0
    has_zi: bool = False

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def log_dens(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Log p(y_j | η_j, phis) for each observation j.

        Parameters
        ----------
        y : ndarray of shape (n,)
            Response values.
        eta : ndarray of shape (n,)
            Linear predictor η = X β + Z b.
        phis : ndarray, optional
            Extra dispersion / shape parameters (log-scale if positive).
        eta_zi : ndarray, optional
            Linear predictor for the zero-inflation part.

        Returns
        -------
        ndarray of shape (n,)
        """

    @abstractmethod
    def linkinv(self, eta: NDArray) -> NDArray:
        """Inverse link g^{-1}(η) → μ."""

    @abstractmethod
    def variance(self, mu: NDArray) -> NDArray:
        """Variance function V(μ)."""

    @abstractmethod
    def mu_eta(self, eta: NDArray) -> NDArray:
        """Derivative dμ/dη = d g^{-1}(η) / dη."""

    # ------------------------------------------------------------------
    # Optional analytic score functions (default: None → finite diff)
    # ------------------------------------------------------------------

    def score_eta(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> Optional[NDArray]:
        """
        Analytic score ∂log p / ∂η.

        Return ``None`` to trigger automatic finite-difference fallback.
        """
        return None

    def score_phis(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> Optional[NDArray]:
        """
        Analytic score ∂log p / ∂phis.

        Return ``None`` to trigger automatic finite-difference fallback.
        """
        return None

    def score_eta_zi(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> Optional[NDArray]:
        """
        Analytic score ∂log p / ∂eta_zi (per observation).

        Only meaningful for zero-inflated families (``has_zi = True``).
        Return ``None`` to trigger automatic finite-difference fallback.
        """
        return None

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def fitted(self, eta: NDArray) -> NDArray:
        """μ̂ = linkinv(η) (alias for linkinv)."""
        return self.linkinv(eta)

    def link_fun(self, mu: NDArray) -> NDArray:
        """
        Forward link g(μ) → η.

        Used by ``marginal_coefs()`` to back-transform averaged predictions.
        Default: numerical inverse via bisection.  Override for efficiency.
        """
        from scipy.optimize import brentq
        result = np.empty_like(mu, dtype=float)
        for idx in np.ndindex(mu.shape):
            m = float(mu[idx])
            try:
                result[idx] = brentq(lambda eta: float(self.linkinv(np.array([eta]))) - m,
                                     -500, 500, xtol=1e-10)
            except ValueError:
                result[idx] = np.nan
        return result

    def simulate_response(
        self,
        mu: NDArray,
        phis: Optional[NDArray],
        eta_zi: Optional[NDArray],
        rng: "np.random.Generator",  # noqa: F821
    ) -> NDArray:
        """
        Draw one simulated response vector from the fitted distribution.

        Parameters
        ----------
        mu : ndarray of shape (n,)
            Fitted mean E[Y | b].
        phis : ndarray or None
            Dispersion parameters (log-scale).
        eta_zi : ndarray or None
            Linear predictor for the zero-inflation component.
        rng : np.random.Generator
            NumPy random generator for reproducibility.

        Returns
        -------
        ndarray of shape (n,)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.simulate_response() is not implemented"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(link='{self.link}')"
