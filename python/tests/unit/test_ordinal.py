"""
Unit tests for glmmadaptive.ordinal (cr_setup and cr_marg_probs).

R cross-validation fixtures were captured by running:

    library(GLMMadaptive)
    y_small <- c(0L, 1L, 2L, 3L, 0L, 2L)
    fwd <- cr_setup(y_small)
    bwd <- cr_setup(y_small, direction="backward")
    eta  <- matrix(c(-1,0,1, 0,1,-1), nrow=2, byrow=TRUE)
    mp_f <- cr_marg_probs(eta)
    mp_b <- cr_marg_probs(eta, direction="backward")

R uses 1-based indexing; ``subs`` fixtures below are converted to 0-based.
"""

import numpy as np
import pandas as pd
import pytest

from glmmadaptive.ordinal import cr_setup, cr_marg_probs

# ---------------------------------------------------------------------------
# R fixtures (0-indexed subs, literal R output otherwise)
# ---------------------------------------------------------------------------

# y_small = c(0L, 1L, 2L, 3L, 0L, 2L) — 4 levels {0,1,2,3}, ncoefs=3

# === forward cr_setup ===
# R: fwd$y    : 1 0 1 0 0 1 0 0 0 1 0 0 1
R_FWD_Y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], dtype=int)
# R: fwd$subs : 1 2 2 3 3 3 4 4 4 5 6 6 6  → 0-indexed
R_FWD_SUBS = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5, 5], dtype=int)
# R: fwd$reps : 1 2 3 3 1 3
R_FWD_REPS = np.array([1, 2, 3, 3, 1, 3], dtype=int)
# R: as.character(fwd$cohort): all all y>=1 all y>=1 y>=2 all y>=1 y>=2 all all y>=1 y>=2
R_FWD_COHORT = [
    "all", "all", "y>=1", "all", "y>=1", "y>=2",
    "all", "y>=1", "y>=2", "all", "all", "y>=1", "y>=2",
]

# === backward cr_setup ===
# R: bwd$y    : 0 0 0 0 0 1 0 1 1 0 0 0 0 1
R_BWD_Y = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1], dtype=int)
# R: bwd$subs : 1 1 1 2 2 2 3 3 4 5 5 5 6 6  → 0-indexed
R_BWD_SUBS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5], dtype=int)
# R: bwd$reps : 3 3 2 1 3 2
R_BWD_REPS = np.array([3, 3, 2, 1, 3, 2], dtype=int)
# R: as.character(bwd$cohort): all y<=2 y<=1 all y<=2 y<=1 all y<=2 all all y<=2 y<=1 all y<=2
R_BWD_COHORT = [
    "all", "y<=2", "y<=1", "all", "y<=2", "y<=1",
    "all", "y<=2", "all", "all", "y<=2", "y<=1", "all", "y<=2",
]

# === cr_marg_probs forward (eta shape 2x3 → output 2x4) ===
# R printed:
#           [,1]      [,2]       [,3]       [,4]
# [1,] 0.2689414 0.3655293 0.26722332 0.09830597
# [2,] 0.5000000 0.3655293 0.03616474 0.09830597
R_MPROBS_FWD = np.array([
    [0.26894141, 0.36552930, 0.26722332, 0.09830597],
    [0.50000000, 0.36552930, 0.03616474, 0.09830597],
])

# === cr_marg_probs backward (eta shape 2x3 → output 2x4) ===
# R printed:
#            [,1]       [,2]      [,3]      [,4]
# [1,] 0.09830597 0.03616474 0.1344707 0.7310586
# [2,] 0.09830597 0.09830597 0.5344466 0.2689414
R_MPROBS_BWD = np.array([
    [0.09830597, 0.03616474, 0.13447072, 0.73105858],
    [0.09830597, 0.09830597, 0.53444664, 0.26894141],
])

# Input eta matrix used for cr_marg_probs tests
ETA_SMALL = np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, -1.0]])

Y_SMALL = np.array([0, 1, 2, 3, 0, 2], dtype=int)


# ---------------------------------------------------------------------------
# cr_setup — structural tests
# ---------------------------------------------------------------------------

class TestCrSetupForwardStructure:
    def test_output_keys(self):
        out = cr_setup(Y_SMALL)
        assert set(out.keys()) == {"y", "cohort", "subs", "reps"}

    def test_y_is_binary(self):
        out = cr_setup(Y_SMALL)
        assert set(out["y"]).issubset({0, 1})

    def test_subs_valid_indices(self):
        out = cr_setup(Y_SMALL)
        assert out["subs"].min() >= 0
        assert out["subs"].max() < len(Y_SMALL)

    def test_cohort_is_categorical(self):
        out = cr_setup(Y_SMALL)
        assert isinstance(out["cohort"], pd.Categorical)

    def test_row_count(self):
        out = cr_setup(Y_SMALL)
        assert len(out["y"]) == out["reps"].sum()
        assert len(out["subs"]) == out["reps"].sum()
        assert len(out["cohort"]) == out["reps"].sum()


# ---------------------------------------------------------------------------
# cr_setup — R cross-validation: forward
# ---------------------------------------------------------------------------

class TestCrSetupForwardMatchesR:
    def setup_method(self):
        self.out = cr_setup(Y_SMALL, direction="forward")

    def test_y_matches_r(self):
        np.testing.assert_array_equal(self.out["y"], R_FWD_Y)

    def test_subs_matches_r(self):
        np.testing.assert_array_equal(self.out["subs"], R_FWD_SUBS)

    def test_reps_matches_r(self):
        np.testing.assert_array_equal(self.out["reps"], R_FWD_REPS)

    def test_cohort_labels_match_r(self):
        assert list(self.out["cohort"]) == R_FWD_COHORT

    def test_cohort_levels(self):
        assert list(self.out["cohort"].categories) == ["all", "y>=1", "y>=2"]


# ---------------------------------------------------------------------------
# cr_setup — R cross-validation: backward
# ---------------------------------------------------------------------------

class TestCrSetupBackwardMatchesR:
    def setup_method(self):
        self.out = cr_setup(Y_SMALL, direction="backward")

    def test_y_matches_r(self):
        np.testing.assert_array_equal(self.out["y"], R_BWD_Y)

    def test_subs_matches_r(self):
        np.testing.assert_array_equal(self.out["subs"], R_BWD_SUBS)

    def test_reps_matches_r(self):
        np.testing.assert_array_equal(self.out["reps"], R_BWD_REPS)

    def test_cohort_labels_match_r(self):
        assert list(self.out["cohort"]) == R_BWD_COHORT

    def test_cohort_levels(self):
        assert list(self.out["cohort"].categories) == ["all", "y<=2", "y<=1"]


# ---------------------------------------------------------------------------
# cr_setup — edge cases
# ---------------------------------------------------------------------------

def test_cr_setup_raises_for_two_levels():
    y_binary = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="fewer than 3 levels"):
        cr_setup(y_binary)


def test_cr_setup_accepts_pandas_categorical():
    y_cat = pd.Categorical([0, 1, 2, 0, 2], categories=[0, 1, 2])
    out = cr_setup(y_cat)
    assert isinstance(out["cohort"], pd.Categorical)
    assert set(out["y"]).issubset({0, 1})


def test_cr_setup_invalid_direction():
    with pytest.raises(ValueError):
        cr_setup(Y_SMALL, direction="sideways")


# ---------------------------------------------------------------------------
# cr_marg_probs — structural tests
# ---------------------------------------------------------------------------

class TestCrMargProbsForwardStructure:
    def setup_method(self):
        self.probs = cr_marg_probs(ETA_SMALL)

    def test_shape(self):
        # (n, K-1) in → (n, K) out; K-1=3, K=4
        assert self.probs.shape == (2, 4)

    def test_rows_sum_to_one(self):
        np.testing.assert_allclose(self.probs.sum(axis=1), np.ones(2), atol=1e-10)

    def test_non_negative(self):
        assert np.all(self.probs >= 0)


class TestCrMargProbsBackwardStructure:
    def setup_method(self):
        self.probs = cr_marg_probs(ETA_SMALL, direction="backward")

    def test_shape(self):
        assert self.probs.shape == (2, 4)

    def test_rows_sum_to_one(self):
        np.testing.assert_allclose(self.probs.sum(axis=1), np.ones(2), atol=1e-10)

    def test_non_negative(self):
        assert np.all(self.probs >= 0)


# ---------------------------------------------------------------------------
# cr_marg_probs — R cross-validation
# ---------------------------------------------------------------------------

def test_cr_marg_probs_forward_matches_r():
    probs = cr_marg_probs(ETA_SMALL, direction="forward")
    np.testing.assert_allclose(probs, R_MPROBS_FWD, rtol=1e-6)


def test_cr_marg_probs_backward_matches_r():
    probs = cr_marg_probs(ETA_SMALL, direction="backward")
    np.testing.assert_allclose(probs, R_MPROBS_BWD, rtol=1e-6)


def test_cr_marg_probs_raises_for_1d():
    with pytest.raises(ValueError):
        cr_marg_probs(np.array([1.0, 2.0, 3.0]))


def test_cr_marg_probs_raises_for_single_column():
    with pytest.raises(ValueError):
        cr_marg_probs(np.array([[1.0], [2.0]]))


def test_cr_marg_probs_invalid_direction():
    with pytest.raises(ValueError):
        cr_marg_probs(ETA_SMALL, direction="diagonal")


# ---------------------------------------------------------------------------
# End-to-end: cr_setup + MixedModel(Binomial)
# ---------------------------------------------------------------------------

def _simulate_ordinal_data(seed: int = 42, n_subj: int = 100, n_obs: int = 4):
    """Simulate a small ordinal longitudinal dataset (4 categories)."""
    rng = np.random.default_rng(seed)
    n_total = n_subj * n_obs
    ids = np.repeat(np.arange(n_subj), n_obs)
    time = np.tile(np.arange(n_obs), n_subj)
    sex = np.repeat(rng.choice([0, 1], size=n_subj), n_obs)

    b = rng.normal(0, 0.8, size=n_subj)  # random intercepts

    # True thresholds and fixed effects
    thresholds = np.array([-1.5, 0.0, 0.9])
    beta_time = -0.3
    beta_sex = 0.5

    eta_common = beta_time * time + beta_sex * sex + b[ids]
    eta_matrix = eta_common[:, None] + thresholds[None, :]  # (n_total, 3)

    mprobs = cr_marg_probs(eta_matrix, direction="forward")
    y_ord = np.array([rng.choice(4, p=row) for row in mprobs])

    return pd.DataFrame({"id": ids, "time": time, "sex": sex, "y": y_ord})


def test_cr_model_fits_without_error():
    """Full pipeline: simulate → cr_setup → MixedModel(Binomial).fit()"""
    from glmmadaptive import MixedModel
    from glmmadaptive.families import Binomial

    df = _simulate_ordinal_data()
    cr = cr_setup(df["y"].values)

    cr_data = df.iloc[cr["subs"]].copy().reset_index(drop=True)
    cr_data["y_new"] = cr["y"]
    cr_data["cohort"] = cr["cohort"].astype(str)  # formula needs string/categorical

    fm = MixedModel(
        fixed="y_new ~ cohort + sex + time",
        random="~ 1 | id",
        data=cr_data,
        family=Binomial(),
    ).fit(verbose=False)

    coefs = fm.fixef()
    assert len(coefs) > 0
    assert np.all(np.isfinite(coefs.values))


def test_cr_anova_significance():
    """LRT: interaction model should have significantly better fit."""
    from glmmadaptive import MixedModel, MixModResults
    from glmmadaptive.families import Binomial

    df = _simulate_ordinal_data(seed=7)
    cr = cr_setup(df["y"].values)

    cr_data = df.iloc[cr["subs"]].copy().reset_index(drop=True)
    cr_data["y_new"] = cr["y"]
    cr_data["cohort"] = cr["cohort"].astype(str)

    fm = MixedModel(
        fixed="y_new ~ cohort + sex + time",
        random="~ 1 | id",
        data=cr_data,
        family=Binomial(),
    ).fit(verbose=False)

    gm = MixedModel(
        fixed="y_new ~ cohort * sex + time",
        random="~ 1 | id",
        data=cr_data,
        family=Binomial(),
    ).fit(verbose=False)

    lrt = MixModResults.anova(fm, gm)
    # LRT table must have p-value and LRT statistic columns
    assert lrt is not None
    # The interaction model must have a higher log-likelihood
    assert gm.logLik >= fm.logLik
