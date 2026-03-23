"""
GLMMadaptive — Generalized Linear Mixed Models via Adaptive Gauss-Hermite Quadrature.

Python port of the R package GLMMadaptive (Rizopoulos, 2023).

Quick start
-----------
>>> import pandas as pd
>>> import numpy as np
>>> from glmmadaptive import MixedModel
>>> from glmmadaptive.families import Binomial
>>>
>>> model = MixedModel(
...     fixed="y ~ x1 + x2",
...     random="~ 1 | id",
...     data=df,
...     family=Binomial(),
... )
>>> res = model.fit()
>>> print(res.summary())

Conventions
-----------
* The public API follows :mod:`statsmodels` conventions: models are constructed
  from formula strings and a ``pandas.DataFrame``, then fitted with ``.fit()``
  which returns a results object.
* Optimisation is delegated to :func:`statsmodels.base.optimizer` wrappers and
  :func:`scipy.optimize.minimize`.
* Numerical derivatives use central differences by default (identical to the
  R implementation's ``numeric_deriv="cd"`` setting).

Future work — JAX integration
------------------------------
Replacing the finite-difference Hessians with JAX ``jax.jacfwd``/``jax.hessian``
will give exact gradients and significant speed-ups on large datasets.  A
design note and migration path are documented in ``docs/jax_roadmap.md``.

Porting status
--------------
Implemented
~~~~~~~~~~~
* Core fitting engine (hybrid EM + quasi-Newton, adaptive GH quadrature)
* Families: :class:`~glmmadaptive.families.Binomial`,
  :class:`~glmmadaptive.families.Poisson`,
  :class:`~glmmadaptive.families.NegativeBinomial`,
  :class:`~glmmadaptive.families.Gamma`,
  :class:`~glmmadaptive.families.Beta`
* Results: ``summary``, ``coef``/``fixef``/``ranef``, ``vcov``, ``confint``,
  ``logLik``, ``predict``, ``fitted``, ``residuals``, ``anova``

Stubs / TODO
~~~~~~~~~~~~
* Zero-inflated families (``ZIPoisson``, ``ZINegativeBinomial``, ``ZIBinomial``)
* Hurdle families (``HurdlePoisson``, ``HurdleNegativeBinomial``,
  ``HurdleBeta``, ``HurdleLogNormal``)
* Ordinal / continuation-ratio models
* Censored-normal, Student-t, unit-Lindley, COM-Poisson families
* ``marginal_coefs`` (Hedeker et al. 2017 marginalisation)
* ``cooks_distance`` (leave-one-group-out diagnostics)
* ``scoring_rules`` (proper scoring rules)
* ``VIF`` (generalised variance inflation factors)
* ``effectPlotData``
"""

from glmmadaptive.core.mixed_model import MixedModel  # noqa: F401
from glmmadaptive.results.mixmod_results import MixModResults  # noqa: F401
from glmmadaptive import families  # noqa: F401

__version__ = "0.1.0"
__all__ = ["MixedModel", "MixModResults", "families"]
