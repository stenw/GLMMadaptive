"""
Standard GLM family objects.

Implements: Binomial, Poisson, NegativeBinomial, Gamma, Beta.

Each mirrors the corresponding R family and its ``log_dens`` / ``score_eta``
implementations from ``R/Fit_Funs.R`` and ``R/mixed_model.R``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, betaln, logit, expit
from scipy.stats import nbinom

from glmmadaptive.families.base import BaseFamily


# ---------------------------------------------------------------------------
# Helper: numerically stable log(1 + exp(x))
# ---------------------------------------------------------------------------

def _softplus(x: NDArray) -> NDArray:
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))


def _log1mexp(a: NDArray) -> NDArray:
    """log(1 - exp(-a)) for a > 0 (numerically stable)."""
    # See Mächler (2012) "Accurately Computing log(1 – exp(– |a|))"
    return np.where(a < np.log(2), np.log(-np.expm1(-a)), np.log1p(-np.exp(-a)))


# ---------------------------------------------------------------------------
# Binomial
# ---------------------------------------------------------------------------

class Binomial(BaseFamily):
    """
    Binomial family for binary and proportion responses.

    Mirrors ``binomial()`` in R.

    Parameters
    ----------
    link : str
        Link function: ``"logit"`` (default), ``"probit"``, ``"cloglog"``.
    """

    family = "binomial"
    n_phis = 0
    has_zi = False

    def __init__(self, link: str = "logit"):
        if link not in ("logit", "probit", "cloglog"):
            raise ValueError(f"Unknown link '{link}' for Binomial family")
        self.link = link

    # --- inverse link -------------------------------------------------------

    def linkinv(self, eta: NDArray) -> NDArray:
        if self.link == "logit":
            return expit(eta)
        if self.link == "probit":
            from scipy.stats import norm
            return norm.cdf(eta)
        # cloglog
        return -np.expm1(-np.exp(eta))

    # --- variance ------------------------------------------------------------

    def variance(self, mu: NDArray) -> NDArray:
        return mu * (1.0 - mu)

    # --- dμ/dη --------------------------------------------------------------

    def mu_eta(self, eta: NDArray) -> NDArray:
        if self.link == "logit":
            mu = expit(eta)
            return mu * (1.0 - mu)
        if self.link == "probit":
            from scipy.stats import norm
            return norm.pdf(eta)
        # cloglog: dμ/dη = exp(η - exp(η)) = exp(η) * exp(-exp(η))
        return np.exp(eta) * np.exp(-np.exp(eta))

    # --- log density (supports counts: y in 0..n, or binary y in {0,1}) -----

    def log_dens(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        """
        log p(y | η) for binary (y ∈ {0,1}) or binomial (y = successes/trials).

        For binary responses, *y* should be in {0, 1}.
        For proportion/count responses pass a 2-column array [successes, trials]
        as the model response via the formula; internally *y* may be a float.
        """
        mu = self.linkinv(eta)
        mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
        # Standard binary log-likelihood
        return y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu)

    # --- analytic score ∂log p / ∂η -----------------------------------------

    def score_eta(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        mu = self.linkinv(eta)
        mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
        # Canonical logit: score = y - mu; chain rule for others
        if self.link == "logit":
            return y - mu
        dmu_deta = self.mu_eta(eta)
        return (y - mu) / self.variance(mu) * dmu_deta


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------

class Poisson(BaseFamily):
    """
    Poisson family for count responses.

    Parameters
    ----------
    link : str
        Link function: ``"log"`` (default), ``"sqrt"``, ``"identity"``.
    """

    family = "poisson"
    n_phis = 0
    has_zi = False

    def __init__(self, link: str = "log"):
        if link not in ("log", "sqrt", "identity"):
            raise ValueError(f"Unknown link '{link}' for Poisson family")
        self.link = link

    def linkinv(self, eta: NDArray) -> NDArray:
        if self.link == "log":
            return np.exp(eta)
        if self.link == "sqrt":
            return eta ** 2
        return eta  # identity

    def variance(self, mu: NDArray) -> NDArray:
        return mu

    def mu_eta(self, eta: NDArray) -> NDArray:
        if self.link == "log":
            return np.exp(eta)
        if self.link == "sqrt":
            return 2.0 * eta
        return np.ones_like(eta)

    def log_dens(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        mu = self.linkinv(eta)
        mu = np.maximum(mu, 1e-15)
        return y * np.log(mu) - mu - gammaln(y + 1.0)

    def score_eta(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        mu = self.linkinv(eta)
        mu = np.maximum(mu, 1e-15)
        if self.link == "log":
            return y - mu
        dmu = self.mu_eta(eta)
        return (y - mu) / mu * dmu


# ---------------------------------------------------------------------------
# Negative Binomial
# ---------------------------------------------------------------------------

class NegativeBinomial(BaseFamily):
    """
    Negative Binomial family (NB2 parameterisation: Var = μ + μ²/θ).

    The dispersion parameter θ is estimated as ``exp(phis[0])`` so that the
    unconstrained scalar ``phis[0]`` is passed to the optimiser.

    Mirrors ``negative.binomial()`` in ``R/Fit_Funs.R``.

    Parameters
    ----------
    theta : float or None
        If provided, θ is fixed and not estimated.
    """

    family = "negative_binomial"
    link = "log"
    has_zi = False

    def __init__(self, theta: float | None = None):
        self.theta_fixed = theta
        self.n_phis = 0 if theta is not None else 1

    def _get_theta(self, phis: Optional[NDArray]) -> NDArray:
        if self.theta_fixed is not None:
            return np.asarray(self.theta_fixed)
        if phis is None:
            raise ValueError("phis required for NegativeBinomial with free theta")
        return np.exp(phis[0])

    def linkinv(self, eta: NDArray) -> NDArray:
        return np.exp(eta)

    def variance(self, mu: NDArray) -> NDArray:
        # Requires theta — use placeholder; actual variance computed with phis
        return mu  # fallback

    def mu_eta(self, eta: NDArray) -> NDArray:
        return np.exp(eta)

    def log_dens(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        """
        NB2 log-density:
            log p(y|μ,θ) = log Γ(y+θ) - log Γ(θ) - log Γ(y+1)
                          + θ log(θ/(θ+μ)) + y log(μ/(θ+μ))
        """
        mu = np.maximum(self.linkinv(eta), 1e-15)
        theta = self._get_theta(phis)
        ld = (
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1.0)
            + theta * (np.log(theta) - np.log(theta + mu))
            + y * (np.log(mu) - np.log(theta + mu))
        )
        return ld

    def score_eta(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        """∂log p / ∂η (log link → canonical chain rule)."""
        mu = np.maximum(self.linkinv(eta), 1e-15)
        theta = self._get_theta(phis)
        # ∂log p / ∂μ = (y - mu) / (mu * (1 + mu/theta))
        # ∂μ/∂η = mu  (log link)
        return (y - mu) / (mu + mu**2 / theta) * mu

    def score_phis(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> Optional[NDArray]:
        """
        Analytic score ∂log p / ∂phis[0].

        Uses chain rule: ∂/∂phis[0] = ∂/∂θ * ∂θ/∂phis[0]
        where phis[0] = log θ → ∂θ/∂phis[0] = θ.
        """
        if self.theta_fixed is not None:
            return None
        mu = np.maximum(self.linkinv(eta), 1e-15)
        theta = self._get_theta(phis)
        # ∂log p / ∂θ
        from scipy.special import digamma
        d_log_p_d_theta = (
            digamma(y + theta)
            - digamma(theta)
            + np.log(theta)
            + 1.0
            - np.log(theta + mu)
            - (y + theta) / (theta + mu)
        )
        # ∂θ/∂phis[0] = theta (since theta = exp(phis[0]))
        return np.array([np.sum(d_log_p_d_theta * theta)])


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------

class Gamma(BaseFamily):
    """
    Gamma family (log link by default).

    The shape parameter α is estimated as ``exp(phis[0])``.
    Parameterisation: Y ~ Gamma(shape=α, rate=α/μ) so that E[Y]=μ, Var[Y]=μ²/α.

    Mirrors ``Gamma.fam()`` in ``R/Fit_Funs.R``.

    Parameters
    ----------
    link : str
        ``"log"`` (default), ``"inverse"``, or ``"identity"``.
    """

    family = "gamma"
    n_phis = 1
    has_zi = False

    def __init__(self, link: str = "log"):
        if link not in ("log", "inverse", "identity"):
            raise ValueError(f"Unknown link '{link}' for Gamma family")
        self.link = link

    def linkinv(self, eta: NDArray) -> NDArray:
        if self.link == "log":
            return np.exp(eta)
        if self.link == "inverse":
            return 1.0 / eta
        return eta

    def variance(self, mu: NDArray) -> NDArray:
        return mu ** 2

    def mu_eta(self, eta: NDArray) -> NDArray:
        if self.link == "log":
            return np.exp(eta)
        if self.link == "inverse":
            return -1.0 / (eta ** 2)
        return np.ones_like(eta)

    def log_dens(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        """
        log p(y|μ,α) = α log(α/μ) + (α-1) log(y) - α y/μ - log Γ(α)
        """
        if phis is None:
            raise ValueError("Gamma family requires phis (shape parameter)")
        alpha = np.exp(phis[0])
        mu = np.maximum(self.linkinv(eta), 1e-15)
        y = np.maximum(y, 1e-15)
        return (
            alpha * np.log(alpha / mu)
            + (alpha - 1.0) * np.log(y)
            - alpha * y / mu
            - gammaln(alpha)
        )


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------

class Beta(BaseFamily):
    """
    Beta family for proportion responses on the open interval (0, 1).

    Parameterisation: Y ~ Beta(μφ, (1-μ)φ) where φ = exp(phis[0]).
    E[Y] = μ, Var[Y] = μ(1-μ)/(1+φ).

    Mirrors ``beta.fam()`` in ``R/Fit_Funs.R``.

    Parameters
    ----------
    link : str
        ``"logit"`` (default) or ``"log-log"``.
    """

    family = "beta"
    n_phis = 1
    has_zi = False

    def __init__(self, link: str = "logit"):
        if link not in ("logit", "log-log"):
            raise ValueError(f"Unknown link '{link}' for Beta family")
        self.link = link

    def linkinv(self, eta: NDArray) -> NDArray:
        if self.link == "logit":
            return expit(eta)
        return np.exp(-np.exp(-eta))  # log-log

    def variance(self, mu: NDArray) -> NDArray:
        return mu * (1.0 - mu)

    def mu_eta(self, eta: NDArray) -> NDArray:
        if self.link == "logit":
            mu = expit(eta)
            return mu * (1.0 - mu)
        return np.exp(-np.exp(-eta)) * np.exp(-eta)

    def log_dens(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        """
        log p(y|μ,φ) = log B(μφ, (1-μ)φ)^{-1} + (μφ-1) log y + ((1-μ)φ-1) log(1-y)
        """
        if phis is None:
            raise ValueError("Beta family requires phis (precision parameter)")
        phi = np.exp(phis[0])
        mu = np.clip(self.linkinv(eta), 1e-15, 1.0 - 1e-15)
        y = np.clip(y, 1e-15, 1.0 - 1e-15)
        a = mu * phi
        b = (1.0 - mu) * phi
        return (
            gammaln(phi)
            - gammaln(a)
            - gammaln(b)
            + (a - 1.0) * np.log(y)
            + (b - 1.0) * np.log(1.0 - y)
        )
