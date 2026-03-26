"""
Optional ``marginaleffects`` integration for GLMMadaptive.

Requires the ``marginaleffects`` package (``pip install marginaleffects`` or
``pip install glmmadaptive[marginaleffects]``).

Usage
-----
>>> from glmmadaptive.marginaleffects import wrap_mixmod
>>> from marginaleffects import predictions, comparisons, datagrid, hypotheses
>>>
>>> mfit = wrap_mixmod(fit)          # fit is a MixModResults object
>>> predictions(mfit)
>>> comparisons(mfit, variables={"time": "pairwise"})
"""

from .adapter import ModelGLMMadaptive, wrap_mixmod

__all__ = ["ModelGLMMadaptive", "wrap_mixmod"]
