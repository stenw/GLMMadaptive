"""
Zero-inflated family objects.

Mirrors ``zi.poisson`` and ``zi.negative.binomial`` in ``R/Fit_Funs.R``.

Both families use a logit parameterisation for the structural-zero
probability:  π = expit(η_zi),  where η_zi = X_zi γ + Z_zi b_zi.

The observed-data log-density is:

* y = 0:  log(π + (1-π) · p_0)          where p_0 = P(base_dist = 0)
* y > 0:  log(1-π) + log p_{base}(y)

All score functions are analytic.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, expit, digamma

from glmmadaptive.families.base import BaseFamily


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_sigmoid(x: NDArray) -> NDArray:
    """log(expit(x)) = -softplus(-x) — numerically stable."""
    return -np.logaddexp(0.0, -x)


def _log_one_minus_sigmoid(x: NDArray) -> NDArray:
    """log(1 - expit(x)) = -softplus(x) — numerically stable."""
    return -np.logaddexp(0.0, x)


# ---------------------------------------------------------------------------
# Zero-inflated Poisson
# ---------------------------------------------------------------------------

class ZIPoisson(BaseFamily):
    """
    Zero-inflated Poisson (ZIP) distribution with log link.

    P(Y=0) = π + (1-π) exp(-μ)
    P(Y=y) = (1-π) exp(-μ) μ^y / y!    for y > 0

    where π = expit(η_zi) is the structural-zero probability and
    μ = exp(η) is the Poisson mean.

    Mirrors ``zi.poisson()`` in ``R/Fit_Funs.R``.
    """

    family = "zi_poisson"
    link = "log"
    n_phis = 0
    has_zi = True

    def linkinv(self, eta: NDArray) -> NDArray:
        return np.exp(eta)

    def variance(self, mu: NDArray) -> NDArray:
        return mu

    def mu_eta(self, eta: NDArray) -> NDArray:
        return np.exp(eta)

    def log_dens(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        if eta_zi is None:
            raise ValueError("ZIPoisson requires eta_zi")
        mu = np.maximum(np.exp(eta), 1e-15)
        log_pi = _log_sigmoid(eta_zi)        # log π
        log_1mpi = _log_one_minus_sigmoid(eta_zi)  # log(1-π)

        # Poisson log-density
        log_pois = y * np.log(mu) - mu - gammaln(y + 1.0)
        log_p0_pois = -mu  # log P(Poisson = 0)

        return np.where(
            y == 0,
            np.logaddexp(log_pi, log_1mpi + log_p0_pois),
            log_1mpi + log_pois,
        )

    def score_eta(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        if eta_zi is None:
            raise ValueError("ZIPoisson requires eta_zi")
        mu = np.maximum(np.exp(eta), 1e-15)
        log_p0_pois = -mu
        log_denom = np.logaddexp(eta_zi, log_p0_pois)  # log(λ + exp(-μ))

        # y=0: d/dη log[λ + exp(-μ)] = -μ·exp(-μ)/(λ + exp(-μ))
        zi0_score = -mu * np.exp(log_p0_pois - log_denom)
        # y>0: d/dη log Poisson = y - μ
        return np.where(y == 0, zi0_score, y - mu)

    def score_eta_zi(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        if eta_zi is None:
            raise ValueError("ZIPoisson requires eta_zi")
        mu = np.maximum(np.exp(eta), 1e-15)
        pi = expit(eta_zi)
        log_p0_pois = -mu
        log_denom = np.logaddexp(eta_zi, log_p0_pois)

        # y=0:  λ/(λ + exp(-μ)) - π   where λ = exp(η_zi)
        zi0_score = np.exp(eta_zi - log_denom) - pi
        # y>0: -π
        return np.where(y == 0, zi0_score, -pi)


# ---------------------------------------------------------------------------
# Zero-inflated Negative Binomial
# ---------------------------------------------------------------------------

class ZINegativeBinomial(BaseFamily):
    """
    Zero-inflated Negative Binomial (ZINB, NB2) with log link.

    P(Y=0) = π + (1-π) · NB(0; μ, θ)
    P(Y=y) = (1-π) · NB(y; μ, θ)    for y > 0

    where π = expit(η_zi),  μ = exp(η),  θ = exp(phis[0]).

    Var(Y | b) = μ + μ²/θ   (NB2 parameterisation).

    Mirrors ``zi.negative.binomial()`` in ``R/Fit_Funs.R``.

    Parameters
    ----------
    theta : float or None
        If provided, θ is fixed and not estimated (n_phis becomes 0).
    """

    family = "zi_negative_binomial"
    link = "log"
    has_zi = True

    def __init__(self, theta: float | None = None):
        self.theta_fixed = theta
        self.n_phis = 0 if theta is not None else 1

    def _get_theta(self, phis: Optional[NDArray]) -> float:
        if self.theta_fixed is not None:
            return float(self.theta_fixed)
        if phis is None:
            raise ValueError("phis required for ZINegativeBinomial with free theta")
        return float(np.exp(phis[0]))

    def linkinv(self, eta: NDArray) -> NDArray:
        return np.exp(eta)

    def variance(self, mu: NDArray) -> NDArray:
        return mu  # placeholder; true variance needs theta

    def mu_eta(self, eta: NDArray) -> NDArray:
        return np.exp(eta)

    def log_dens(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        if eta_zi is None:
            raise ValueError("ZINegativeBinomial requires eta_zi")
        mu = np.maximum(np.exp(eta), 1e-15)
        theta = self._get_theta(phis)

        # NB2 log-density
        log_nb = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1.0)
            + theta * (np.log(theta) - np.log(theta + mu))
            + y * (np.log(mu) - np.log(theta + mu))
        )
        # log P(NB = 0) = θ · log(θ/(θ+μ))
        log_p0_nb = theta * (np.log(theta) - np.log(theta + mu))

        log_pi = _log_sigmoid(eta_zi)
        log_1mpi = _log_one_minus_sigmoid(eta_zi)

        return np.where(
            y == 0,
            np.logaddexp(log_pi, log_1mpi + log_p0_nb),
            log_1mpi + log_nb,
        )

    def score_eta(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        if eta_zi is None:
            raise ValueError("ZINegativeBinomial requires eta_zi")
        mu = np.maximum(np.exp(eta), 1e-15)
        theta = self._get_theta(phis)

        # NB score for y > 0:  θ(y - μ)/(μ + θ)
        nb_score = theta * (y - mu) / (mu + theta)

        # ZI adjustment for y = 0
        log_p0_nb = theta * (np.log(theta) - np.log(theta + mu))
        log_denom = np.logaddexp(eta_zi, log_p0_nb)  # log(λ + p_nb0)

        # d/dη log p0 = exp(log_p_nb0 - log_denom) · (-θμ/(θ+μ))
        zi0_score = np.exp(log_p0_nb - log_denom) * (-theta * mu / (theta + mu))

        return np.where(y == 0, zi0_score, nb_score)

    def score_eta_zi(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        if eta_zi is None:
            raise ValueError("ZINegativeBinomial requires eta_zi")
        mu = np.maximum(np.exp(eta), 1e-15)
        theta = self._get_theta(phis)
        pi = expit(eta_zi)

        log_p0_nb = theta * (np.log(theta) - np.log(theta + mu))
        log_denom = np.logaddexp(eta_zi, log_p0_nb)  # log(λ + p_nb0)

        # y=0:  λ/(λ + p_nb0) - π
        zi0_score = np.exp(eta_zi - log_denom) - pi
        # y>0:  -π
        return np.where(y == 0, zi0_score, -pi)

    def score_phis(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> Optional[NDArray]:
        if self.theta_fixed is not None:
            return None
        if eta_zi is None:
            raise ValueError("ZINegativeBinomial requires eta_zi")
        mu = np.maximum(np.exp(eta), 1e-15)
        theta = self._get_theta(phis)

        # y > 0: NB score w.r.t. θ
        d_log_nb_d_theta = (
            digamma(y + theta)
            - digamma(theta)
            + np.log(theta) + 1.0
            - np.log(theta + mu)
            - (y + theta) / (theta + mu)
        )

        # y = 0: d log p0 / d θ = exp(log_p_nb0 - log_denom) · d log_p_nb0/d θ
        log_p0_nb = theta * (np.log(theta) - np.log(theta + mu))
        log_denom = np.logaddexp(eta_zi, log_p0_nb)
        # d log p_nb0 / d θ = log(θ/(θ+μ)) + μ/(θ+μ)
        d_log_p0_nb_d_theta = np.log(theta) - np.log(theta + mu) + mu / (theta + mu)
        zi0_d_theta = np.exp(log_p0_nb - log_denom) * d_log_p0_nb_d_theta

        d_log_p_d_theta = np.where(y == 0, zi0_d_theta, d_log_nb_d_theta)

        # Chain rule: d/d phis[0] = d/d θ · θ  (θ = exp(phis[0]))
        return np.array([float(np.sum(d_log_p_d_theta * theta))])


# ---------------------------------------------------------------------------
# Zero-inflated Binomial  (stub — less common, implement on demand)
# ---------------------------------------------------------------------------

class ZIBinomial(BaseFamily):
    """Zero-inflated Binomial (stub)."""

    family = "zi_binomial"
    link = "logit"
    n_phis = 0
    has_zi = True

    def linkinv(self, eta: NDArray) -> NDArray:
        return expit(eta)

    def variance(self, mu: NDArray) -> NDArray:
        return mu * (1.0 - mu)

    def mu_eta(self, eta: NDArray) -> NDArray:
        mu = expit(eta)
        return mu * (1.0 - mu)

    def log_dens(self, y, eta, phis=None, eta_zi=None) -> NDArray:
        raise NotImplementedError("ZIBinomial.log_dens not yet implemented")
