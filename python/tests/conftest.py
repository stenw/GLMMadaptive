"""
Pytest configuration and shared fixtures.

The ``r_glmm`` fixture provides a helper that fits a model in R via rpy2 and
returns the relevant numerical results for comparison with the Python port.

The ``sim_binary_data`` and ``sim_count_data`` fixtures generate small
synthetic datasets that are also used to pre-compute R reference outputs
stored in ``tests/fixtures/``.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Synthetic dataset generators
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def sim_binary_data(rng):
    """
    Small binary longitudinal dataset.

    200 subjects, 5 observations each.  Random intercept model.
    True parameters: beta = [−1.0, 0.5],  D = [[0.5]].
    """
    n_subjects = 200
    n_obs = 5
    N = n_subjects * n_obs

    id_ = np.repeat(np.arange(n_subjects), n_obs)
    time = np.tile(np.arange(n_obs), n_subjects)
    b = rng.normal(0, np.sqrt(0.5), n_subjects)
    eta = -1.0 + 0.5 * time + b[id_]
    from scipy.special import expit
    p = expit(eta)
    y = rng.binomial(1, p)

    return pd.DataFrame({"id": id_, "time": time, "y": y})


@pytest.fixture(scope="session")
def sim_poisson_data(rng):
    """
    Small Poisson count dataset with random intercept.

    150 subjects, 4 observations each.
    True params: beta = [0.5, 0.3],  D = [[0.4]].
    """
    n_subjects = 150
    n_obs = 4
    id_ = np.repeat(np.arange(n_subjects), n_obs)
    x = rng.normal(0, 1, n_subjects * n_obs)
    b = rng.normal(0, np.sqrt(0.4), n_subjects)
    eta = 0.5 + 0.3 * x + b[id_]
    lam = np.exp(eta)
    y = rng.poisson(lam)
    return pd.DataFrame({"id": id_, "x": x, "y": y})


@pytest.fixture(scope="session")
def sim_negbinom_data(rng):
    """Negative Binomial data; theta (size) = 2."""
    n_subjects = 100
    n_obs = 4
    id_ = np.repeat(np.arange(n_subjects), n_obs)
    x = rng.normal(0, 1, n_subjects * n_obs)
    b = rng.normal(0, np.sqrt(0.3), n_subjects)
    eta = 0.4 + 0.2 * x + b[id_]
    mu = np.exp(eta)
    theta = 2.0
    p_nb = theta / (theta + mu)
    y = rng.negative_binomial(theta, p_nb)
    return pd.DataFrame({"id": id_, "x": x, "y": y})


# ---------------------------------------------------------------------------
# Reference fixture loader
# ---------------------------------------------------------------------------

def load_fixture(name: str) -> dict:
    """Load a JSON reference fixture produced by ``generate_r_fixtures.R``."""
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"Reference fixture '{name}.json' not found. "
                    f"Run tests/fixtures/generate_r_fixtures.R first.")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# rpy2 helper (skips if rpy2 / R not available)
# ---------------------------------------------------------------------------

def _rpy2_available() -> bool:
    try:
        import rpy2.robjects as ro  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def r_env():
    """Return an rpy2 R environment, or skip if not available."""
    if not _rpy2_available():
        pytest.skip("rpy2 not installed — skipping R comparison tests")
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    try:
        importr("GLMMadaptive")
    except Exception:
        pytest.skip("R package GLMMadaptive not installed")
    return ro


def r_fit_binary(r_env, data: pd.DataFrame) -> dict:
    """
    Fit the binary random-intercept model in R and return numeric results.

    Returns dict with keys: betas, D, logLik, bse.
    """
    ro = r_env
    import rpy2.robjects as ro_mod
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    r_data = pandas2ri.py2rpy(data)
    ro_mod.globalenv["py_data"] = r_data
    ro_mod.r("""
        library(GLMMadaptive)
        fm <- mixed_model(
            fixed  = y ~ time,
            random = ~ 1 | id,
            data   = py_data,
            family = binomial()
        )
        r_betas   <- as.numeric(fixef(fm))
        r_D       <- as.numeric(fm$D)
        r_loglik  <- as.numeric(logLik(fm))
        r_bse     <- sqrt(diag(vcov(fm)))
    """)
    return {
        "betas": np.array(ro_mod.r("r_betas")),
        "D": np.array([[ro_mod.r("r_D")[0]]]),
        "logLik": float(ro_mod.r("r_loglik")[0]),
        "bse": np.array(ro_mod.r("r_bse")),
    }
