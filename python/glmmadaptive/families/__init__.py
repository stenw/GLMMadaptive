"""
Family objects for GLMMadaptive.

Each family object specifies the response distribution and carries the
functions required by the fitting engine:

* ``log_dens(y, eta, phis)``  — log p(y | eta, phis) per observation
* ``linkinv(eta)``            — inverse link function μ = g^{-1}(η)
* ``variance(mu)``            — variance function V(μ)
* ``mu_eta(eta)``             — dμ/dη

Implemented
-----------
* :class:`Binomial`        — logit / probit / cloglog link
* :class:`Poisson`         — log / sqrt / identity link
* :class:`NegativeBinomial` — log link, over-dispersion parameter θ
* :class:`Gamma`           — log / inverse / identity link
* :class:`Beta`            — logit / log-log link

Stubs (raise ``NotImplementedError``)
--------------------------------------
* :class:`ZIPoisson`, :class:`ZINegativeBinomial`, :class:`ZIBinomial`
* :class:`HurdlePoisson`, :class:`HurdleNegativeBinomial`,
  :class:`HurdleBeta`, :class:`HurdleLogNormal`
* :class:`CensoredNormal`, :class:`StudentsT`, :class:`UnitLindley`,
  :class:`BetaBinomial`
"""

from glmmadaptive.families.base import BaseFamily
from glmmadaptive.families.standard import Binomial, Poisson, NegativeBinomial, Gamma, Beta
from glmmadaptive.families.zero_inflated import ZIPoisson, ZINegativeBinomial, ZIBinomial
from glmmadaptive.families.hurdle import (
    HurdlePoisson,
    HurdleNegativeBinomial,
    HurdleBeta,
    HurdleLogNormal,
)

__all__ = [
    "BaseFamily",
    "Binomial",
    "Poisson",
    "NegativeBinomial",
    "Gamma",
    "Beta",
    "ZIPoisson",
    "ZINegativeBinomial",
    "ZIBinomial",
    "HurdlePoisson",
    "HurdleNegativeBinomial",
    "HurdleBeta",
    "HurdleLogNormal",
]
