"""
Standard GLM family objects.

Implements: Binomial, Poisson, NegativeBinomial, Gamma, Beta, Gaussian,
StudentsT.

Each mirrors the corresponding R family and its ``log_dens`` / ``score_eta``
implementations from ``R/Fit_Funs.R`` and ``R/mixed_model.R``.

Notes
-----
The R package explicitly rejects ``gaussian()`` and redirects users to
``lme()`` / ``lmer()``.  The Python port implements :class:`Gaussian` here
because there is no equivalent Python GLMM package to redirect to, and the
adaptive Gauss-Hermite quadrature approach works perfectly well for normal
responses.  For the same reason :class:`StudentsT` (R: ``students.t()``) is
also fully implemented.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, betaln, logit, expit
from scipy.stats import nbinom, norm as sp_norm, t as sp_t

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
        if np.ndim(y) == 2:
            # Grouped binomial: y = [successes, failures]
            k = y[:, 0]
            N = y[:, 0] + y[:, 1]
            return (gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)
                    + k * np.log(mu) + (N - k) * np.log(1.0 - mu))
        # Standard binary log-likelihood
        return y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu)

    # --- forward link g(μ) → η -----------------------------------------------

    def link_fun(self, mu: NDArray) -> NDArray:
        mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
        if self.link == "logit":
            return logit(mu)
        if self.link == "probit":
            from scipy.stats import norm
            return norm.ppf(mu)
        # cloglog: η = log(-log(1-μ))
        return np.log(-np.log1p(-mu))

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
        if np.ndim(y) == 2:
            k = y[:, 0]
            N = y[:, 0] + y[:, 1]
            if self.link == "logit":
                return k - N * mu          # canonical: ∂logL/∂η = k − N·μ
            dmu_deta = self.mu_eta(eta)
            return (k - N * mu) / (N * self.variance(mu)) * dmu_deta
        # Canonical logit: score = y - mu; chain rule for others
        if self.link == "logit":
            return y - mu
        dmu_deta = self.mu_eta(eta)
        return (y - mu) / self.variance(mu) * dmu_deta

    # --- simulate response ---------------------------------------------------

    def simulate_response(self, mu, phis, eta_zi, rng):
        mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
        return rng.binomial(1, mu).astype(float)


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

    def link_fun(self, mu: NDArray) -> NDArray:
        mu = np.maximum(mu, 1e-15)
        if self.link == "log":
            return np.log(mu)
        if self.link == "sqrt":
            return np.sqrt(mu)
        return mu  # identity

    def simulate_response(self, mu, phis, eta_zi, rng):
        mu = np.maximum(mu, 1e-15)
        return rng.poisson(mu).astype(float)


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

    def link_fun(self, mu: NDArray) -> NDArray:
        return np.log(np.maximum(mu, 1e-15))

    def simulate_response(self, mu, phis, eta_zi, rng):
        mu = np.maximum(mu, 1e-15)
        theta = self._get_theta(phis)
        # NB2: size=theta, p = theta/(theta+mu)
        p = theta / (theta + mu)
        return rng.negative_binomial(theta, p).astype(float)


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

    def link_fun(self, mu: NDArray) -> NDArray:
        mu = np.maximum(mu, 1e-15)
        if self.link == "log":
            return np.log(mu)
        if self.link == "inverse":
            return 1.0 / mu
        return mu  # identity

    def simulate_response(self, mu, phis, eta_zi, rng):
        if phis is None:
            raise ValueError("Gamma requires phis")
        alpha = np.exp(phis[0])
        mu = np.maximum(mu, 1e-15)
        # Gamma(shape=alpha, scale=mu/alpha)
        return rng.gamma(alpha, mu / alpha)


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


# ---------------------------------------------------------------------------
# Gaussian (Normal)
# ---------------------------------------------------------------------------

class Gaussian(BaseFamily):
    """
    Gaussian (normal) family for continuous responses.

    Parameterisation: Y ~ N(η, σ²) where σ = exp(phis[0]).
    The residual standard deviation σ is estimated on the log scale so that
    the optimiser works over the unconstrained real line.

    Mirrors the ``students.t()`` R family in the df → ∞ limit.  The R package
    itself rejects ``gaussian()`` and redirects to ``lme()`` / ``lmer()``, but
    the Python port implements this family directly via adaptive GHQ.

    Parameters
    ----------
    link : str
        Link function: ``"identity"`` (default), ``"log"``, or ``"inverse"``.
    """

    family = "gaussian"
    n_phis = 1
    has_zi = False

    def __init__(self, link: str = "identity"):
        if link not in ("identity", "log", "inverse"):
            raise ValueError(f"Unknown link '{link}' for Gaussian family")
        self.link = link

    # --- inverse link -------------------------------------------------------

    def linkinv(self, eta: NDArray) -> NDArray:
        if self.link == "log":
            return np.exp(eta)
        if self.link == "inverse":
            return 1.0 / eta
        return eta  # identity

    # --- variance ------------------------------------------------------------

    def variance(self, mu: NDArray) -> NDArray:
        # Constant variance (dispersion carried in phis)
        return np.ones_like(mu)

    # --- dμ/dη --------------------------------------------------------------

    def mu_eta(self, eta: NDArray) -> NDArray:
        if self.link == "log":
            return np.exp(eta)
        if self.link == "inverse":
            return -1.0 / (eta ** 2)
        return np.ones_like(eta)  # identity

    # --- log density --------------------------------------------------------

    def log_dens(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        """
        log p(y|η, σ) = -½ log(2π) - log σ - ½ (y - μ)² / σ²

        where μ = linkinv(η) and σ = exp(phis[0]).
        """
        if phis is None:
            raise ValueError("Gaussian family requires phis (log residual SD)")
        sigma = np.exp(phis[0])
        mu = self.linkinv(eta)
        return sp_norm.logpdf(y, loc=mu, scale=sigma)

    # --- analytic score ∂log p / ∂η ----------------------------------------

    def score_eta(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        """
        ∂ log p / ∂η = (y - μ) / σ²  ×  dμ/dη
        """
        if phis is None:
            raise ValueError("Gaussian family requires phis")
        sigma2 = np.exp(2.0 * phis[0])
        mu = self.linkinv(eta)
        return (y - mu) / sigma2 * self.mu_eta(eta)

    # --- analytic score ∂log p / ∂phis[0] -----------------------------------

    def score_phis(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> Optional[NDArray]:
        """
        ∂ log p / ∂ phis[0]  where phis[0] = log σ.

        d/d(log σ) [-log σ - ½(y-μ)²/σ²]  =  (y-μ)²/σ² - 1
        Summed over all observations.
        """
        if phis is None:
            raise ValueError("Gaussian family requires phis")
        sigma2 = np.exp(2.0 * phis[0])
        mu = self.linkinv(eta)
        return np.array([np.sum((y - mu) ** 2 / sigma2 - 1.0)])

    def link_fun(self, mu: NDArray) -> NDArray:
        mu = np.asarray(mu, dtype=float)
        if self.link == "log":
            return np.log(np.maximum(mu, 1e-15))
        if self.link == "inverse":
            return 1.0 / mu
        return mu  # identity

    def simulate_response(self, mu, phis, eta_zi, rng):
        if phis is None:
            raise ValueError("Gaussian requires phis")
        sigma = np.exp(phis[0])
        return rng.normal(mu, sigma)


# ---------------------------------------------------------------------------
# StudentsT
# ---------------------------------------------------------------------------

class StudentsT(BaseFamily):
    """
    Location-scale Student's-t family for continuous responses.

    Parameterisation: Y ~ t(η, σ², df) where σ = exp(phis[0]) and df is
    fixed (not estimated).  As df → ∞ this converges to :class:`Gaussian`.

    Mirrors ``students.t()`` in ``R/Fit_Funs.R``.

    Parameters
    ----------
    df : float
        Degrees of freedom (required; must be positive).
    link : str
        Link function: ``"identity"`` (default), ``"log"``, or ``"inverse"``.
    """

    family = "students_t"
    n_phis = 1
    has_zi = False

    def __init__(self, df: float, link: str = "identity"):
        if df <= 0:
            raise ValueError("df must be positive")
        if link not in ("identity", "log", "inverse"):
            raise ValueError(f"Unknown link '{link}' for StudentsT family")
        self.df = float(df)
        self.link = link

    # --- inverse link -------------------------------------------------------

    def linkinv(self, eta: NDArray) -> NDArray:
        if self.link == "log":
            return np.exp(eta)
        if self.link == "inverse":
            return 1.0 / eta
        return eta

    # --- variance ------------------------------------------------------------

    def variance(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    # --- dμ/dη --------------------------------------------------------------

    def mu_eta(self, eta: NDArray) -> NDArray:
        if self.link == "log":
            return np.exp(eta)
        if self.link == "inverse":
            return -1.0 / (eta ** 2)
        return np.ones_like(eta)

    # --- log density --------------------------------------------------------

    def log_dens(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        """
        log p(y|η, σ, df) = log dt((y-μ)/σ, df) - log σ

        Mirrors ``dt(x = (y - eta) / sigma, df = .df, log = TRUE) - log(sigma)``
        in R's ``students.t`` implementation.
        """
        if phis is None:
            raise ValueError("StudentsT family requires phis (log scale parameter)")
        sigma = np.exp(phis[0])
        mu = self.linkinv(eta)
        return sp_t.logpdf((y - mu) / sigma, df=self.df) - phis[0]

    # --- analytic score ∂log p / ∂η ----------------------------------------

    def score_eta(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Mirrors R's ``score_eta_fun`` for ``students.t``:
            (y-μ) * (df+1) / (df*σ²) / (1 + (y-μ)²/(df*σ²))  ×  dμ/dη
        """
        if phis is None:
            raise ValueError("StudentsT family requires phis")
        sigma2 = np.exp(2.0 * phis[0])
        mu = self.linkinv(eta)
        d = y - mu
        score_mu = d * (self.df + 1.0) / (self.df * sigma2) / (
            1.0 + d ** 2 / (self.df * sigma2)
        )
        return score_mu * self.mu_eta(eta)

    # --- analytic score ∂log p / ∂phis[0] -----------------------------------

    def score_phis(
        self,
        y: NDArray,
        eta: NDArray,
        phis: Optional[NDArray] = None,
        eta_zi: Optional[NDArray] = None,
    ) -> Optional[NDArray]:
        """
        Mirrors R's ``score_phis_fun`` for ``students.t``:
            (df+1) * (y-μ)²/(df*σ²) / (1 + (y-μ)²/(df*σ²)) - 1
        summed over observations (chain rule: ∂σ/∂phis[0] = σ, so ×σ/σ = 1
        after substituting through the identity-link score formula).
        """
        if phis is None:
            raise ValueError("StudentsT family requires phis")
        sigma2 = np.exp(2.0 * phis[0])
        mu = self.linkinv(eta)
        d2_df = (y - mu) ** 2 / self.df
        score_per_obs = (self.df + 1.0) * d2_df / sigma2 / (
            1.0 + d2_df / sigma2
        ) - 1.0
        return np.array([np.sum(score_per_obs)])

    def link_fun(self, mu: NDArray) -> NDArray:
        mu = np.asarray(mu, dtype=float)
        if self.link == "log":
            return np.log(np.maximum(mu, 1e-15))
        if self.link == "inverse":
            return 1.0 / mu
        return mu  # identity

    def simulate_response(self, mu, phis, eta_zi, rng):
        if phis is None:
            raise ValueError("StudentsT requires phis")
        sigma = np.exp(phis[0])
        return sp_t.rvs(df=self.df, loc=mu, scale=sigma, random_state=rng)
