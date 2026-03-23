"""
Numerical differentiation utilities.

Mirrors ``fd()``, ``cd()``, ``fd_vec()``, ``cd_vec()`` from
``R/Functions.R`` in the original GLMMadaptive package.

Convention
----------
All functions operate on *scalar-output* callables unless noted.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable


_MACHINE_EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# Gradient estimators
# ---------------------------------------------------------------------------

def fd_grad(
    f: Callable[[NDArray], float],
    x: NDArray,
    eps: float | None = None,
) -> NDArray:
    """
    Forward-difference gradient of *f* at *x*.

    Mirrors ``fd()`` in ``R/Functions.R``.
    Step size: ``h_i = eps * (|x_i| + eps)``  where ``eps = machine_eps^0.25``.

    Parameters
    ----------
    f : callable
        Scalar-valued function.
    x : ndarray of shape (p,)
        Evaluation point.
    eps : float, optional
        Base step size (default: ``machine_eps^0.25 ≈ 1.22e-4``).

    Returns
    -------
    ndarray of shape (p,)
    """
    if eps is None:
        eps = _MACHINE_EPS ** 0.25
    x = np.asarray(x, dtype=float)
    p = x.size
    f0 = f(x)
    grad = np.empty(p)
    for i in range(p):
        h = eps * (abs(x[i]) + eps)
        x_fwd = x.copy()
        x_fwd[i] += h
        grad[i] = (f(x_fwd) - f0) / h
    return grad


def cd_grad(
    f: Callable[[NDArray], float],
    x: NDArray,
    eps: float = 1e-4,
) -> NDArray:
    """
    Central-difference gradient of *f* at *x*.

    Mirrors ``cd()`` in ``R/Functions.R``.
    Step size: ``h_i = eps * max(|x_i|, 1)``.

    Parameters
    ----------
    f : callable
        Scalar-valued function.
    x : ndarray of shape (p,)
        Evaluation point.
    eps : float
        Base step size (default: 1e-4).

    Returns
    -------
    ndarray of shape (p,)
    """
    x = np.asarray(x, dtype=float)
    p = x.size
    grad = np.empty(p)
    for i in range(p):
        h = eps * max(abs(x[i]), 1.0)
        x_fwd = x.copy()
        x_bwd = x.copy()
        x_fwd[i] += h
        x_bwd[i] -= h
        grad[i] = (f(x_fwd) - f(x_bwd)) / (2.0 * h)
    return grad


# ---------------------------------------------------------------------------
# Hessian estimators
# ---------------------------------------------------------------------------

def fd_hess(
    f: Callable[[NDArray], float],
    x: NDArray,
    eps: float | None = None,
) -> NDArray:
    """
    Forward-difference Hessian of *f* at *x*.

    Mirrors ``fd_vec()`` in ``R/Functions.R``.

    Returns a symmetrised approximation ``0.5 * (J + J^T)``.
    """
    if eps is None:
        eps = _MACHINE_EPS ** 0.25
    x = np.asarray(x, dtype=float)
    p = x.size
    grad0 = fd_grad(f, x, eps=eps)
    H = np.empty((p, p))
    for i in range(p):
        h = eps * (abs(x[i]) + eps)
        x_fwd = x.copy()
        x_fwd[i] += h
        grad_fwd = fd_grad(f, x_fwd, eps=eps)
        H[i, :] = (grad_fwd - grad0) / h
    return 0.5 * (H + H.T)


def cd_hess(
    f: Callable[[NDArray], float],
    x: NDArray,
    eps: float = 1e-3,
) -> NDArray:
    """
    Central-difference Hessian of *f* at *x*.

    Mirrors ``cd_vec()`` in ``R/Functions.R``.
    Step size: ``h_i = eps * max(|x_i|, 1)``.

    Returns a symmetrised approximation ``0.5 * (J + J^T)``.
    """
    x = np.asarray(x, dtype=float)
    p = x.size
    H = np.empty((p, p))
    grad_fwd = np.empty(p)
    grad_bwd = np.empty(p)
    for i in range(p):
        h = eps * max(abs(x[i]), 1.0)
        x_fwd = x.copy()
        x_bwd = x.copy()
        x_fwd[i] += h
        x_bwd[i] -= h
        grad_fwd[i] = cd_grad(f, x_fwd, eps=eps)[i]
        grad_bwd[i] = cd_grad(f, x_bwd, eps=eps)[i]
        H[i, :] = (cd_grad(f, x_fwd, eps=eps) - cd_grad(f, x_bwd, eps=eps)) / (2.0 * h)
    return 0.5 * (H + H.T)
