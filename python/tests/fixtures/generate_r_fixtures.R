#!/usr/bin/env Rscript
# =============================================================================
# generate_r_fixtures.R
#
# Run this script once to generate JSON reference fixtures used by the Python
# regression tests.  Requires the GLMMadaptive package to be installed in R.
#
# Usage:
#   cd python/tests/fixtures
#   Rscript generate_r_fixtures.R
#
# Output:
#   binary_ri.json            — binary random-intercept model
#   poisson_ri.json           — Poisson random-intercept model
#   negbinom_ri.json          — Negative Binomial random-intercept model
#   binary_ri_predictions.json  — predict() and ranef() for binary RI
#   poisson_ri_predictions.json — predict() and ranef() for Poisson RI
#   zi_poisson_ri.json        — zero-inflated Poisson RI model
#   zi_negbinom_ri.json       — zero-inflated Negative Binomial RI model
#   gaussian_ri.json          — Gaussian RI (fitted via students.t df=1e6)
# =============================================================================

suppressPackageStartupMessages({
  library(GLMMadaptive)
  library(jsonlite)
})

set.seed(42)

# ---------------------------------------------------------------------------
# Helper: convert R MixMod fit → list suitable for JSON
# ---------------------------------------------------------------------------
extract_fit <- function(fm, data) {
  list(
    betas  = as.numeric(fixef(fm)),
    D      = lapply(as.data.frame(fm$D), as.numeric),
    logLik = as.numeric(logLik(fm)),
    bse    = as.numeric(sqrt(diag(vcov(fm)))),
    data   = as.list(data)
  )
}

# ---------------------------------------------------------------------------
# 1. Binary random-intercept model
# ---------------------------------------------------------------------------
cat("Fitting binary random-intercept model...\n")

n_subjects <- 200
n_obs      <- 5
id         <- rep(seq_len(n_subjects), each = n_obs)
time       <- rep(0:(n_obs - 1), times = n_subjects)
b          <- rnorm(n_subjects, sd = sqrt(0.5))
eta        <- -1.0 + 0.5 * time + b[id]
p          <- plogis(eta)
y          <- rbinom(n_subjects * n_obs, 1, p)
df_bin     <- data.frame(id = id, time = time, y = y)

fm_bin <- mixed_model(
  fixed  = y ~ time,
  random = ~ 1 | id,
  data   = df_bin,
  family = binomial()
)

bin_fixture        <- extract_fit(fm_bin, df_bin)
bin_fixture$D      <- as.list(as.data.frame(fm_bin$D))  # named list per column

write_json(bin_fixture, "binary_ri.json", digits = 10, auto_unbox = FALSE)
cat("  Saved binary_ri.json\n")
cat("  betas:", paste(round(bin_fixture$betas, 4), collapse = ", "), "\n")
cat("  D[1,1]:", round(fm_bin$D[1,1], 4), "\n")
cat("  logLik:", round(bin_fixture$logLik, 4), "\n\n")

# ---------------------------------------------------------------------------
# 2. Poisson random-intercept model
# ---------------------------------------------------------------------------
cat("Fitting Poisson random-intercept model...\n")

n_subjects <- 150
n_obs      <- 4
id         <- rep(seq_len(n_subjects), each = n_obs)
x          <- rnorm(n_subjects * n_obs)
b          <- rnorm(n_subjects, sd = sqrt(0.4))
eta        <- 0.5 + 0.3 * x + b[id]
y          <- rpois(n_subjects * n_obs, lambda = exp(eta))
df_pois    <- data.frame(id = id, x = x, y = y)

fm_pois <- mixed_model(
  fixed  = y ~ x,
  random = ~ 1 | id,
  data   = df_pois,
  family = poisson()
)

pois_fixture <- extract_fit(fm_pois, df_pois)
write_json(pois_fixture, "poisson_ri.json", digits = 10, auto_unbox = FALSE)
cat("  Saved poisson_ri.json\n")
cat("  betas:", paste(round(pois_fixture$betas, 4), collapse = ", "), "\n")
cat("  D[1,1]:", round(fm_pois$D[1,1], 4), "\n")
cat("  logLik:", round(pois_fixture$logLik, 4), "\n\n")

# ---------------------------------------------------------------------------
# 3. Negative Binomial random-intercept model
# ---------------------------------------------------------------------------
cat("Fitting Negative Binomial random-intercept model...\n")

n_subjects <- 100
n_obs      <- 4
id         <- rep(seq_len(n_subjects), each = n_obs)
x          <- rnorm(n_subjects * n_obs)
b          <- rnorm(n_subjects, sd = sqrt(0.3))
eta        <- 0.4 + 0.2 * x + b[id]
mu         <- exp(eta)
theta_true <- 2.0
y          <- rnbinom(n_subjects * n_obs, size = theta_true,
                      mu = mu)
df_nb      <- data.frame(id = id, x = x, y = y)

fm_nb <- mixed_model(
  fixed  = y ~ x,
  random = ~ 1 | id,
  data   = df_nb,
  family = negative.binomial()
)

nb_fixture       <- extract_fit(fm_nb, df_nb)
nb_fixture$phis  <- as.numeric(fm_nb$phis)
write_json(nb_fixture, "negbinom_ri.json", digits = 10, auto_unbox = FALSE)
cat("  Saved negbinom_ri.json\n")
cat("  betas:", paste(round(nb_fixture$betas, 4), collapse = ", "), "\n")
cat("  theta (exp(phis)):", round(exp(fm_nb$phis), 4), "\n")
cat("  logLik:", round(nb_fixture$logLik, 4), "\n\n")

# ---------------------------------------------------------------------------
# 4. Prediction fixtures for binary and Poisson models
#    (predict type="mean_subject", type="subject_specific", and ranef)
# ---------------------------------------------------------------------------
cat("Generating prediction fixtures for binary model...\n")

preds_ms_bin  <- as.numeric(predict(fm_bin, type = "mean_subject"))
preds_ss_bin  <- as.numeric(predict(fm_bin, type = "subject_specific"))
ranef_bin     <- as.numeric(ranef(fm_bin)[, 1])

bin_pred_fixture <- list(
  data                         = as.list(df_bin),
  betas                        = as.numeric(fixef(fm_bin)),
  D                            = as.list(as.data.frame(fm_bin$D)),
  logLik                       = as.numeric(logLik(fm_bin)),
  predictions_mean_subject     = preds_ms_bin,
  predictions_subject_specific = preds_ss_bin,
  ranef                        = ranef_bin
)
write_json(bin_pred_fixture, "binary_ri_predictions.json",
           digits = 10, auto_unbox = FALSE)
cat("  Saved binary_ri_predictions.json\n\n")

cat("Generating prediction fixtures for Poisson model...\n")

preds_ms_pois  <- as.numeric(predict(fm_pois, type = "mean_subject"))
preds_ss_pois  <- as.numeric(predict(fm_pois, type = "subject_specific"))
ranef_pois     <- as.numeric(ranef(fm_pois)[, 1])

pois_pred_fixture <- list(
  data                         = as.list(df_pois),
  betas                        = as.numeric(fixef(fm_pois)),
  D                            = as.list(as.data.frame(fm_pois$D)),
  logLik                       = as.numeric(logLik(fm_pois)),
  predictions_mean_subject     = preds_ms_pois,
  predictions_subject_specific = preds_ss_pois,
  ranef                        = ranef_pois
)
write_json(pois_pred_fixture, "poisson_ri_predictions.json",
           digits = 10, auto_unbox = FALSE)
cat("  Saved poisson_ri_predictions.json\n\n")

# ---------------------------------------------------------------------------
# 5. Zero-inflated Poisson random-intercept model
# ---------------------------------------------------------------------------
cat("Fitting zero-inflated Poisson model...\n")

n_subjects <- 80
n_obs      <- 6
id         <- rep(seq_len(n_subjects), each = n_obs)
time_zi    <- rep(0:(n_obs - 1), times = n_subjects)
b_zi       <- rnorm(n_subjects, sd = sqrt(0.5))
eta_zi_y   <- 1.2 + 0.15 * time_zi + b_zi[id]
mu_zi      <- exp(eta_zi_y)
shape_zi   <- 2.0
y_zi       <- rnbinom(n_subjects * n_obs, size = shape_zi, mu = mu_zi)
# Structural zeros (gamma0 = -1.5, ~36% zeros)
zi_mask    <- as.logical(rbinom(n_subjects * n_obs, 1, plogis(-1.5)))
y_zi[zi_mask] <- 0L
df_zi      <- data.frame(id = id, time = time_zi, y = y_zi)

fm_zip <- mixed_model(
  fixed    = y ~ time,
  random   = ~ 1 | id,
  data     = df_zi,
  family   = zi.poisson(),
  zi_fixed = ~ 1
)

zip_fixture <- list(
  betas  = as.numeric(fixef(fm_zip)),
  gammas = as.numeric(fm_zip$gammas),
  D      = as.list(as.data.frame(fm_zip$D)),
  logLik = as.numeric(logLik(fm_zip)),
  bse    = as.numeric(sqrt(diag(vcov(fm_zip)))),
  data   = as.list(df_zi)
)
write_json(zip_fixture, "zi_poisson_ri.json", digits = 10, auto_unbox = FALSE)
cat("  Saved zi_poisson_ri.json\n")
cat("  betas:",  paste(round(zip_fixture$betas,  4), collapse = ", "), "\n")
cat("  gammas:", paste(round(zip_fixture$gammas, 4), collapse = ", "), "\n")
cat("  D[1,1]:", round(fm_zip$D[1,1], 4), "\n")
cat("  logLik:", round(zip_fixture$logLik, 4), "\n\n")

# ---------------------------------------------------------------------------
# 6. Zero-inflated Negative Binomial random-intercept model
#    (same data as section 5)
# ---------------------------------------------------------------------------
cat("Fitting zero-inflated Negative Binomial model...\n")

fm_zinb <- mixed_model(
  fixed    = y ~ time,
  random   = ~ 1 | id,
  data     = df_zi,
  family   = zi.negative.binomial(),
  zi_fixed = ~ 1
)

zinb_fixture <- list(
  betas  = as.numeric(fixef(fm_zinb)),
  gammas = as.numeric(fm_zinb$gammas),
  D      = as.list(as.data.frame(fm_zinb$D)),
  phis   = as.numeric(fm_zinb$phis),
  logLik = as.numeric(logLik(fm_zinb)),
  bse    = as.numeric(sqrt(diag(vcov(fm_zinb)))),
  data   = as.list(df_zi)
)
write_json(zinb_fixture, "zi_negbinom_ri.json", digits = 10, auto_unbox = FALSE)
cat("  Saved zi_negbinom_ri.json\n")
cat("  betas:",  paste(round(zinb_fixture$betas,  4), collapse = ", "), "\n")
cat("  gammas:", paste(round(zinb_fixture$gammas, 4), collapse = ", "), "\n")
cat("  theta (exp(phis)):", round(exp(fm_zinb$phis), 4), "\n")
cat("  logLik:", round(zinb_fixture$logLik, 4), "\n\n")

# ---------------------------------------------------------------------------
# 7. Gaussian (Normal) random-intercept model
#    R rejects gaussian() in mixed_model(), so we use students.t(df=1e6)
#    which is numerically indistinguishable from Gaussian.
# ---------------------------------------------------------------------------
cat("Fitting Gaussian (students.t df=1e6) random-intercept model...\n")

n_subjects <- 150
n_obs      <- 5
set.seed(42)
id_g       <- rep(seq_len(n_subjects), each = n_obs)
time_g     <- rep(0:(n_obs - 1), times = n_subjects)
b_g        <- rnorm(n_subjects, sd = sqrt(0.8))
sigma_eps  <- 1.2
y_g        <- 2.0 + 0.5 * time_g + b_g[id_g] + rnorm(n_subjects * n_obs, sd = sigma_eps)
df_gauss   <- data.frame(id = id_g, time = time_g, y = y_g)

fm_gauss <- mixed_model(
  fixed  = y ~ time,
  random = ~ 1 | id,
  data   = df_gauss,
  family = students.t(df = 1e6),   # numerically identical to Gaussian
  n_phis = 1
)

gauss_fixture <- list(
  betas  = as.numeric(fixef(fm_gauss)),
  phis   = as.numeric(fm_gauss$phis),   # log(sigma_residual)
  D      = as.list(as.data.frame(fm_gauss$D)),
  logLik = as.numeric(logLik(fm_gauss)),
  bse    = as.numeric(sqrt(diag(vcov(fm_gauss)))),
  data   = as.list(df_gauss)
)
write_json(gauss_fixture, "gaussian_ri.json", digits = 10, auto_unbox = FALSE)
cat("  Saved gaussian_ri.json\n")
cat("  betas:", paste(round(gauss_fixture$betas, 4), collapse = ", "), "\n")
cat("  sigma = exp(phis):", round(exp(fm_gauss$phis), 4), "\n")
cat("  D[1,1]:", round(fm_gauss$D[1, 1], 4), "\n")
cat("  logLik:", round(gauss_fixture$logLik, 4), "\n\n")

cat("Done.  All fixtures written to:", getwd(), "\n")
