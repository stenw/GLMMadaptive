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
# Output:  binary_ri.json, poisson_ri.json, negbinom_ri.json
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

cat("Done.  All fixtures written to:", getwd(), "\n")
