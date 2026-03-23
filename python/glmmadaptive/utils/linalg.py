"""
Linear-algebra utilities.

Mirrors functions from ``R/Functions.R``:
* ``nearPD()``          — nearest positive-definite matrix (Higham 2002)
* ``chol_transf()``     — Cholesky ↔ covariance parameterisation
* ``dmvnorm()``         — multivariate normal log-density
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky, solve_triangular


# ---------------------------------------------------------------------------
# Nearest positive-definite matrix  (Higham 2002 algorithm)
# ---------------------------------------------------------------------------

def nearPD(
    X: NDArray,
    eig_tol: float = 1e-6,
    conv_tol: float = 1e-7,
    posd_tol: float = 1e-8,
    max_iter: int = 100,
) -> NDArray:
    """
    Project *X* onto the cone of positive-definite symmetric matrices.

    Implements the Higham (2002) algorithm exactly as in ``nearPD()`` from
    ``R/Functions.R`` (which itself follows the ``Matrix::nearPD`` R
    implementation).

    Parameters
    ----------
    X : ndarray of shape (p, p)
        Symmetric matrix to project.
    eig_tol : float
        Minimum eigenvalue as fraction of the largest eigenvalue.
    conv_tol : float
        Convergence threshold on the relative Frobenius change.
    posd_tol : float
        Final eigenvalue threshold to ensure strict positive-definiteness.
    max_iter : int
        Maximum Dykstra iterations.

    Returns
    -------
    ndarray of shape (p, p)
        Nearest positive-definite matrix (symmetric).
    """
    n = X.shape[0]
    D_S = np.zeros_like(X)  # Dykstra correction
    X_hat = X.copy()

    for _ in range(max_iter):
        Y = X_hat
        # Apply Dykstra correction
        R = Y - D_S
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(R)
        # Clip eigenvalues
        lam_max = eigvals.max()
        eigvals_clip = np.maximum(eigvals, eig_tol * lam_max)
        X_hat = eigvecs @ np.diag(eigvals_clip) @ eigvecs.T
        # Update Dykstra correction
        D_S = X_hat - R
        # Symmetrise
        X_hat = 0.5 * (X_hat + X_hat.T)
        # Convergence check
        diff = np.max(np.abs(Y - X_hat))
        norm_Y = np.max(np.abs(Y))
        if norm_Y > 0 and diff / norm_Y < conv_tol:
            break

    # Final clipping to ensure strict PD
    eigvals, eigvecs = np.linalg.eigh(X_hat)
    lam_max = eigvals.max()
    if eigvals.min() < posd_tol * lam_max:
        eigvals = np.maximum(eigvals, posd_tol * lam_max)
        X_hat = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return 0.5 * (X_hat + X_hat.T)


# ---------------------------------------------------------------------------
# Cholesky ↔ covariance reparameterisation  (port of chol_transf() in R)
# ---------------------------------------------------------------------------

def cov_to_chol(D: NDArray, diagonal: bool = False) -> NDArray:
    """
    Unconstrained parameterisation of a positive-definite matrix.

    For a full matrix: returns the lower-triangular Cholesky factor with
    log-diagonal, packed as a 1-D vector of length ``q*(q+1)//2``.
    For a diagonal matrix: returns ``log(diag(D))``.

    Parameters
    ----------
    D : ndarray of shape (q, q)
        Positive-definite covariance matrix.
    diagonal : bool
        If True, only the diagonal of *D* is used.

    Returns
    -------
    ndarray
        Unconstrained parameter vector.
    """
    q = D.shape[0]
    if diagonal:
        return np.log(np.diag(D))
    L = cholesky(D, lower=True)
    # log-transform diagonal so the full vector is unconstrained
    L[np.arange(q), np.arange(q)] = np.log(L[np.arange(q), np.arange(q)])
    idx = np.tril_indices(q)
    return L[idx]


def chol_to_cov(params: NDArray, q: int, diagonal: bool = False) -> NDArray:
    """
    Inverse of :func:`cov_to_chol`: recover a PD covariance matrix.

    Parameters
    ----------
    params : ndarray
        Unconstrained parameter vector.
    q : int
        Dimension of the covariance matrix.
    diagonal : bool
        If True, *params* is interpreted as ``log(diag(D))``.

    Returns
    -------
    ndarray of shape (q, q)
        Positive-definite covariance matrix.
    """
    if diagonal:
        return np.diag(np.exp(params))
    L = np.zeros((q, q))
    idx = np.tril_indices(q)
    L[idx] = params
    L[np.arange(q), np.arange(q)] = np.exp(L[np.arange(q), np.arange(q)])
    return L @ L.T


# ---------------------------------------------------------------------------
# Multivariate normal log-density  (port of dmvnorm() in R)
# ---------------------------------------------------------------------------

def log_dmvnorm(
    x: NDArray,
    mean: NDArray | None = None,
    cov: NDArray | None = None,
    cov_inv: NDArray | None = None,
    log_det_cov: float | None = None,
) -> float | NDArray:
    """
    Log-density of the multivariate normal distribution.

    Mirrors ``dmvnorm()`` in ``R/Functions.R``.

    Parameters
    ----------
    x : ndarray of shape (q,) or (n, q)
        Evaluation point(s).
    mean : ndarray of shape (q,), optional
        Mean vector (default: zeros).
    cov : ndarray of shape (q, q), optional
        Covariance matrix.  Either *cov* or both *cov_inv* and
        *log_det_cov* must be provided.
    cov_inv : ndarray of shape (q, q), optional
        Pre-computed inverse of *cov* (avoids re-computation in hot loops).
    log_det_cov : float, optional
        Pre-computed ``log(det(cov))``.

    Returns
    -------
    float or ndarray
        Log-density value(s).
    """
    x = np.atleast_2d(x)
    q = x.shape[-1]

    if mean is not None:
        x = x - mean

    if cov_inv is None or log_det_cov is None:
        if cov is None:
            raise ValueError("Provide either cov or (cov_inv, log_det_cov)")
        sign, log_det_cov = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Covariance matrix is not positive definite")
        cov_inv = np.linalg.inv(cov)

    maha = np.einsum("...i,ij,...j->...", x, cov_inv, x)
    log_dens = -0.5 * (q * np.log(2 * np.pi) + log_det_cov + maha)
    return float(log_dens) if log_dens.ndim == 0 else log_dens


def dmvnorm(
    x: NDArray,
    mean: NDArray | None = None,
    cov: NDArray | None = None,
    **kwargs,
) -> float | NDArray:
    """Multivariate normal density (not log)."""
    return np.exp(log_dmvnorm(x, mean=mean, cov=cov, **kwargs))
