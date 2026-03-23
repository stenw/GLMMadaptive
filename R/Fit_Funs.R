# logLik_mixed
# Computes the negative marginal log-likelihood of the mixed model using adaptive
# Gauss-Hermite quadrature. The marginal likelihood integrates out the random effects:
#   log p(y) = sum_i log integral[ p(y_i | b_i) * p(b_i) ] db_i
# where p(b_i) = N(0, D). The integral is approximated by the pre-computed AGH quadrature
# (provided in GH), using the log-sum-exp trick for numerical stability.
#
# The function negates the log-likelihood to allow minimization by standard optimizers.
# If penalized = TRUE, a multivariate t penalty on betas is added.
#
# Arguments:
#   thetas:       numeric vector of all unconstrained parameters (betas, Chol-D, phis, gammas)
#   id:           integer vector of group indices for each observation
#   y:            response vector or matrix
#   N:            binomial totals (NULL if not applicable)
#   X:            fixed-effects design matrix (n_obs x ncx)
#   Z:            random-effects design matrix (n_obs x ncz)
#   offset:       offset for the linear predictor (NULL if not applicable)
#   X_zi:         zero-inflation fixed-effects design matrix (NULL if not ZI)
#   Z_zi:         zero-inflation random-effects design matrix (NULL if not ZI)
#   offset_zi:    offset for the ZI linear predictor (NULL if not applicable)
#   GH:           list from GHfun() containing quadrature nodes, weights, and log-determinants
#   canonical:    logical; TRUE for canonical links (binomial/logit, poisson/log)
#   user_defined: logical; TRUE if a user-defined family is used
#   Xty:          precomputed X'y cross-product (used with canonical link for efficiency)
#   Xty_weights:  weighted version of Xty (NULL if no weights)
#   log_dens:     function(y, eta, mu_fun, phis, eta_zi) returning log-density
#   mu_fun:       inverse-link function
#   var_fun:      variance function of the family
#   mu.eta_fun:   derivative of the mean w.r.t. the linear predictor
#   score_eta_fun, score_eta_zi_fun, score_phis_fun: analytic score functions (or NULL)
#   list_thetas:  list skeleton for relist() to parse thetas into betas/D/phis/gammas
#   diag_D:       logical; TRUE if D is constrained to be diagonal
#   penalized:    logical; TRUE to add the Student's-t penalty on betas
#   pen_mu, pen_invSigma, pen_df: penalty hyperparameters (used if penalized = TRUE)
#   weights:      numeric vector of group weights (NULL if unweighted)
#   i_contributions: logical; if TRUE, return per-group contributions instead of total
#
# Returns:
#   If i_contributions = FALSE: the negative total marginal log-likelihood (scalar).
#   If i_contributions = TRUE: a numeric vector of per-group negative log-likelihood contributions.
logLik_mixed <- function (thetas, id, y, N, X, Z, offset, X_zi, Z_zi, offset_zi, GH,
                          canonical, user_defined, Xty, Xty_weights, log_dens, mu_fun, var_fun,
                          mu.eta_fun, score_eta_fun, score_eta_zi_fun, score_phis_fun,
                          list_thetas, diag_D, penalized, pen_mu, pen_invSigma, pen_df,
                          weights, i_contributions = FALSE) {
    thetas <- relist(thetas, skeleton = list_thetas)
    betas <- thetas$betas
    phis <- thetas$phis
    gammas <- thetas$gammas
    D <- if (diag_D) diag(exp(thetas$D), length(thetas$D)) else chol_transf(thetas$D)
    nRE <- ncol(D)
    ##
    b <- GH$b
    Ztb <- GH$Ztb
    Z_zitb <- GH$Z_zitb
    wGH <- GH$wGH
    log_wGH <- rep(log(wGH), each = length(unique(id)))
    #dets <- GH$dets
    log_dets <- GH$log_dets
    ##
    eta_y <- as.vector(X %*% betas) + Ztb
    if (!is.null(offset))
        eta_y <- eta_y + offset
    eta_zi <- if (!is.null(X_zi)) as.vector(X_zi %*% gammas)
    if (!is.null(Z_zi))
        eta_zi <- eta_zi + Z_zitb
    if (!is.null(offset_zi))
        eta_zi <- eta_zi + offset_zi
    log_Lik <- log_dens(y, eta_y, mu_fun, phis, eta_zi)
    log_p_yb <- unname(rowsum(log_Lik, id, reorder = FALSE))
    log_p_b <- matrix(dmvnorm(b, rep(0, nRE), D, TRUE),
                      nrow(log_p_yb), ncol(log_p_yb), byrow = TRUE)
    log_p_y <- rowLogSumExps(log_p_yb + log_p_b + log_wGH) + log_dets
    out <- if (i_contributions) {
        - if (is.null(weights)) log_p_y else weights * log_p_y
    } else {
        - sum(if (is.null(weights)) log_p_y else weights * log_p_y, na.rm = TRUE)
    }
    if (penalized)
        out <- out - dmvt(betas, mu = pen_mu, invSigma = pen_invSigma, df = pen_df)
    out
}

# score_mixed
# Computes the gradient (score vector) of the negative marginal log-likelihood with
# respect to all unconstrained parameters (betas, Chol-D, phis, gammas).
#
# The score is computed by differentiating through the AGH approximation of the marginal
# log-likelihood. For each parameter group:
#   - score.betas: uses the expected score with respect to the linear predictor eta,
#     weighted by the posterior probability p(b|y). For canonical links this simplifies
#     to X'(E[mu] - y); for non-canonical links uses the chain rule.
#   - score.D: uses the EM M-step formula for D:
#     - Diagonal D: 0.5 * (n/D_kk - E[b_k^2] / D_kk^2) re-parameterized for log(D_kk)
#     - Full D: uses deriv_D() and jacobian2() for the chain rule through the Cholesky param
#   - score.phis: uses numerical central differences if no analytic score_phis_fun provided
#   - score.gammas: uses the expected score w.r.t. eta_zi, weighted by posterior
#
# The score is negated (to match the negated log-likelihood) so that both logLik_mixed
# and score_mixed can be passed directly to minimization routines.
#
# Arguments: (same as logLik_mixed, plus i_contributions)
#   i_contributions: if TRUE, returns per-observation (or per-group for score.D) contributions
#                    as a list; used for the sandwich variance estimator in vcov.MixMod()
#
# Returns:
#   If i_contributions = FALSE: a numeric vector (the gradient of the negative log-likelihood)
#   If i_contributions = TRUE: a named list with components score.betas, score.D,
#                               score.phis, score.gammas
score_mixed <- function (thetas, id, y, N, X, Z, offset, X_zi, Z_zi, offset_zi, GH,
                         canonical, user_defined, Xty, Xty_weights, log_dens, mu_fun, var_fun,
                         mu.eta_fun, score_eta_fun, score_eta_zi_fun, score_phis_fun,
                         list_thetas, diag_D, penalized, pen_mu, pen_invSigma, pen_df,
                         i_contributions = FALSE, weights) {
    thetas <- relist(thetas, skeleton = list_thetas)
    betas <- thetas$betas
    phis <- thetas$phis
    gammas <- thetas$gammas
    D <- if (diag_D) diag(exp(thetas$D), length(thetas$D)) else chol_transf(thetas$D)
    nRE <- ncol(D)
    ##
    b <- GH$b
    b2 <- GH$b2
    Ztb <- GH$Ztb
    Z_zitb <- GH$Z_zitb
    wGH <- GH$wGH
    log_wGH <- rep(log(wGH), each = length(unique(id)))
    ##
    eta_y <- as.vector(X %*% betas) + Ztb
    if (!is.null(offset))
        eta_y <- eta_y + offset
    eta_zi <- if (!is.null(X_zi)) as.vector(X_zi %*% gammas)
    if (!is.null(Z_zi))
        eta_zi <- eta_zi + Z_zitb
    if (!is.null(offset_zi))
        eta_zi <- eta_zi + offset_zi
    log_Lik <- log_dens(y, eta_y, mu_fun, phis, eta_zi)
    log_p_yb <- unname(rowsum(log_Lik, id, reorder = FALSE))
    log_p_b <- matrix(dmvnorm(b, rep(0, nRE), D, TRUE),
                      nrow(log_p_yb), ncol(log_p_yb), byrow = TRUE)
    log_p_yb_b <- log_p_yb + log_p_b
    log_p_y <- rowLogSumExps(log_p_yb_b + log_wGH)
    p_by <- exp(log_p_yb_b - log_p_y)
    t_p_by <- t(p_by)
    n <- length(log_p_y)
    NN <- if (NCOL(y) == 2) nrow(y) else length(y)
    post_b <- apply(b, 2, function (b_k) colSums(t_p_by * matrix(b_k, ncol(Ztb), n) * wGH))
    post_b2 <- apply(b2, 2, function (b_k) colSums(t_p_by * matrix(b_k, ncol(Ztb), n) * wGH))
    if (!is.null(weights)) {
        post_b <- weights * post_b
        post_b2 <- weights * post_b2
    }
    post_vb <- post_b2 - if (nRE > 1) t(apply(post_b, 1, function (x) x %o% x)) else
        as.matrix(apply(post_b, 1, function (x) x %o% x))
    ###
    mu_y <- if (!is.null(attr(log_Lik, "mu_y"))) attr(log_Lik, "mu_y") else mu_fun(eta_y)
    score.betas <- if (user_defined) {
        ncx <- ncol(X)
        sc <- if (i_contributions) matrix(0.0, NN, ncx) else numeric(ncx)
        if (!is.null(score_eta_fun)) {
            z <- score_eta_fun(y, mu_y, phis, eta_zi)
            for (l in seq_len(ncx)) {
                cc <- if (is.null(weights)) {
                    drop(rowsum(X[, l] * z, id, reorder = FALSE))
                } else {
                    weights * drop(rowsum(X[, l] * z, id, reorder = FALSE))
                } 
                if (i_contributions) {
                    sc[, l] <- c((X[, l] * z * p_by[id, , drop = FALSE]) %*% wGH)
                } else {
                    sc[l] <- sum(c((cc * p_by) %*% wGH), na.rm = TRUE)
                }
            }
            - sc
        } else {
            l1 <- log_dens(y, eta_y + 1e-04, mu_fun, phis, eta_zi)
            l2 <- log_dens(y, eta_y - 1e-04, mu_fun, phis, eta_zi)
            z <- (l1 - l2) / (2 * 1e-04)
            for (l in seq_len(ncx)) {
                cc <- if (is.null(weights)) {
                    drop(rowsum(X[, l] * z, id, reorder = FALSE))
                } else {
                    weights * drop(rowsum(X[, l] * z, id, reorder = FALSE))
                }
                if (i_contributions) {
                    sc[, l] <- c((X[, l] * z * p_by[id, , drop = FALSE]) %*% wGH)
                } else {
                    sc[l] <- sum(c((cc * p_by) %*% wGH), na.rm = TRUE)
                }
            }
            - sc
        }
    } else {
        ncx <- ncol(X)
        sc <- if (i_contributions) matrix(0.0, NN, ncx) else numeric(ncx)
        if (canonical) {
            if (!is.null(N))
                mu_y <- mu_y * N
            for (l in seq_len(ncx)) {
                cc <- if (is.null(weights)) {
                    drop(rowsum(X[, l] * mu_y, id, reorder = FALSE))
                } else {
                    weights * drop(rowsum(X[, l] * mu_y, id, reorder = FALSE))
                }
                if (i_contributions) {
                    sc[, l] <- c((X[, l] * mu_y * p_by[id, , drop = FALSE]) %*% wGH)
                } else {
                    sc[l] <- sum(c((cc * p_by) %*% wGH), na.rm = TRUE)
                }
           }
            if (i_contributions) {
                - (X * if (NCOL(y) == 2) y[, 1] else y) + sc
            } else {
                if (is.null(weights)) - Xty + sc else - Xty_weights + sc
            }
        } else {
            var <- var_fun(mu_y)
            deriv <- mu.eta_fun(eta_y)
            z <- if (!is.null(N)) (y[, 1] - N * mu_y) * deriv / var else (y - mu_y) * deriv / var
            for (l in seq_len(ncx)) {
                cc <- if (is.null(weights)) {
                    drop(rowsum(X[, l] * z, id, reorder = FALSE))
                } else {
                    weights * drop(rowsum(X[, l] * z, id, reorder = FALSE))
                }
                if (i_contributions) {
                    sc[, l] <- c((X[, l] * z * p_by[id, , drop = FALSE]) %*% wGH)
                } else {
                    sc[l] <- sum(c((cc * p_by) %*% wGH), na.rm = TRUE)
                }
            }
            - sc
        }
    }
    if (penalized) {
        pen_invSigma_betas <- betas * diag(pen_invSigma) / pen_df
        fact <- (pen_df + ncx) / c(1 + crossprod(betas, pen_invSigma_betas))
        score.betas <- score.betas + pen_invSigma_betas * fact
    }
    ###
    score.phis <- if (!is.null(phis)) {
        if (is.null(score_phis_fun)) {
            n_phis <- length(phis)
            sc <- if (i_contributions) matrix(0.0, NN, n_phis) else numeric(n_phis)
            for (i in seq_len(n_phis)) {
                phis1 <- phis2 <- phis
                phis1[i] <- phis[i] + 1e-03
                phis2[i] <- phis[i] - 1e-03
                l1 <- log_dens(y, eta_y, mu_fun, phis1, eta_zi)
                l2 <- log_dens(y, eta_y, mu_fun, phis2, eta_zi)
                z <- (l1 - l2) / (phis1[i] - phis2[i])
                if (i_contributions) {
                    sc[, i] <- c((z * p_by[id, , drop = FALSE]) %*% wGH)
                } else {
                    cc <- if (is.null(weights)) {
                        c((rowsum(z, id, reorder = FALSE) * p_by) %*% wGH)
                    } else {
                        weights * c((rowsum(z, id, reorder = FALSE) * p_by) %*% wGH)
                    }
                    sc[i] <- sum(cc, na.rm = TRUE)
                }
            }
            - sc
        } else {
            z <- score_phis_fun(y, mu_y, phis, eta_zi)
            if (i_contributions) {
                -c((z * p_by[id, , drop = FALSE]) %*% wGH)
            } else {
                cc <- if (is.null(weights)) {
                    c((rowsum(z, id, reorder = FALSE) * p_by) %*% wGH)
                } else {
                    weights * c((rowsum(z, id, reorder = FALSE) * p_by) %*% wGH)
                }
                -sum(cc, na.rm = TRUE)
            }
        }
    }
    score.gammas <- if (!is.null(X_zi)) {
        z <- if (!is.null(score_eta_zi_fun)) {
            score_eta_zi_fun(y, mu_y, phis, eta_zi)
        } else {
            l1 <- log_dens(y, eta_y, mu_fun, phis, eta_zi + 1e-03)
            l2 <- log_dens(y, eta_y, mu_fun, phis, eta_zi - 1e-03)
            (l1 - l2) / (2 * 1e-03)
        }
        ncx_zi <- ncol(X_zi)
        sc <- if (i_contributions) matrix(0.0, NN, ncx_zi) else numeric(ncx_zi)
        for (l in seq_len(ncx_zi)) {
            cc <- if (is.null(weights)) {
                drop(rowsum(X_zi[, l] * drop(z), id, reorder = FALSE))
            } else {
                weights * drop(rowsum(X_zi[, l] * drop(z), id, reorder = FALSE))
            }
            if (i_contributions) {
                sc[, l] <- c((X_zi[, l] * drop(z) * p_by[id, , drop = FALSE]) %*% wGH)
            } else {
                sc[l] <- sum(c((cc * p_by) %*% wGH), na.rm = TRUE)
            }
        }
        - sc
    }
    ###
    score.D <- if (diag_D) {
        D <- diag(D)
        svD <- 1/D
        svD2 <- svD^2
        if (i_contributions) {
            NA
        } else {
            cS.postVB <- colSums(as.matrix(post_vb), na.rm = TRUE)
            dim(cS.postVB) <- c(nRE, nRE)
            D * 0.5 * (n * svD - diag(cS.postVB) * svD2 - 
                           colSums(as.matrix(post_b^2), na.rm = TRUE) * svD2)
        }
    } else {
        svD <- solve(D)
        dD <- deriv_D(D)
        ndD <- length(dD)
        D1 <- sapply(dD, function (x) sum(svD * x))
        D2 <- t(sapply(dD, function (x) c(svD %*% x %*% svD)))
        if (i_contributions) {
            rr <- matrix(0.0, n, ndD)
            for (j in seq_len(n)) {
                cS.postVB <- colSums(as.matrix(post_vb)[j, , drop = FALSE], na.rm = TRUE)
                out <- numeric(ndD)
                for (i in seq_along(dD)) {
                    D.mat <- D2[i, ]
                    dim(D.mat) <- c(nRE, nRE)
                    out[i] <- sum(D2[i, ] * cS.postVB, na.rm = TRUE) +
                        sum((post_b[j, , drop = FALSE] %*% D.mat) * post_b[j, , drop = FALSE], na.rm = TRUE)
                }
                J <- jacobian2(attr(D, "L"), nRE)
                rr[j, ] <- drop(0.5 * (D1 - out) %*% J)
            }
            rr
        } else {
            cS.postVB <- colSums(as.matrix(post_vb), na.rm = TRUE)
            out <- numeric(ndD)
            for (i in seq_along(dD)) {
                D.mat <- D2[i, ]
                dim(D.mat) <- c(nRE, nRE)
                out[i] <- sum(D2[i, ] * cS.postVB, na.rm = TRUE) +
                    sum((post_b %*% D.mat) * post_b, na.rm = TRUE)
            }
            J <- jacobian2(attr(D, "L"), nRE)
            if (is.null(weights)) {
                drop(0.5 * (n * D1 - out) %*% J)
            } else {
                drop(0.5 * (sum(weights) * D1 - out) %*% J)
            }
        }
    }
    ###
    if (i_contributions)
        list(score.betas = score.betas, score.D = score.D, score.phis = score.phis, 
             score.gammas = score.gammas)
    else
        c(score.betas, score.D, score.phis, score.gammas)
}

# score_betas
# Computes the score (gradient) of the negative log-likelihood with respect to the fixed-
# effects coefficients (betas). Used during the EM Newton-Raphson update for betas.
#
# The score for betas is:
#   - For canonical links: sum_i E_{b|y}[X_i' (mu_i(b) - y_i)] (efficient form using X'y)
#   - For non-canonical links: -sum_i E_{b|y}[X_i' (y_i - mu_i(b)) * dmu/deta / var(mu_i)]
#   - For user-defined families: uses the provided score_eta_fun or numerical differences
# The expectation is approximated using the posterior weights p_by from the EM E-step.
#
# Arguments:
#   betas:         current fixed-effects coefficient vector
#   y:             response vector or matrix
#   N:             binomial totals (NULL if not applicable)
#   X:             fixed-effects design matrix
#   id:            integer vector of group indices
#   offset:        offset vector (NULL if not applicable)
#   weights:       group weights (NULL if unweighted)
#   phis:          current dispersion parameters (NULL if not applicable)
#   Ztb:           pre-computed Z * b values (from GHfun output)
#   eta_zi:        zero-inflation linear predictor (NULL if not ZI)
#   p_by:          n x k^q matrix of posterior weights from EM E-step
#   wGH:           quadrature weights vector
#   canonical:     logical; TRUE for canonical link
#   user_defined:  logical; TRUE for user-defined family
#   Xty, Xty_weights: precomputed X'y and weighted X'y (for canonical link efficiency)
#   log_dens, mu_fun, var_fun, mu.eta_fun: family functions
#   score_eta_fun, score_phis_fun: analytic score functions (NULL triggers numeric diff)
#   penalized, pen_mu, pen_invSigma, pen_df: penalty settings
#
# Returns:
#   A numeric vector of length ncx (the gradient of the negative log-likelihood w.r.t. betas).
score_betas <- function (betas, y, N, X, id, offset, weights, phis, Ztb, eta_zi, p_by, wGH, canonical,
                         user_defined, Xty, Xty_weights, log_dens, mu_fun, var_fun, mu.eta_fun,
                         score_eta_fun, score_phis_fun, penalized, pen_mu, pen_invSigma,
                         pen_df) {
    eta_y <- as.vector(X %*% betas) + Ztb
    if (!is.null(offset))
        eta_y <- eta_y + offset
    mu_y <- mu_fun(eta_y)
    ncx <- ncol(X)
    sc <- numeric(ncx)
    out <- if (user_defined) {
        if (!is.null(score_eta_fun)) {
            z <- score_eta_fun(y, mu_y, phis, eta_zi)
            for (l in seq_len(ncx)) {
                cc <- if (is.null(weights)) {
                    drop(rowsum(X[, l] * z, id, reorder = FALSE))
                } else {
                    weights * drop(rowsum(X[, l] * z, id, reorder = FALSE))
                }
                sc[l] <- sum(c((cc * p_by) %*% wGH), na.rm = TRUE)
            }
            - sc
        } else {
            l1 <- log_dens(y, eta_y + 1e-05, mu_fun, phis, eta_zi)
            l2 <- log_dens(y, eta_y - 1e-05, mu_fun, phis, eta_zi)
            z <- (l1 - l2) / (2 * 1e-05)
            for (l in seq_len(ncx)) {
                cc <- if (is.null(weights)) {
                    drop(rowsum(X[, l] * z, id, reorder = FALSE))
                } else {
                    weights * drop(rowsum(X[, l] * z, id, reorder = FALSE))
                }
                sc[l] <- sum(c((cc * p_by) %*% wGH), na.rm = TRUE)
            }
            - sc
        }
    } else {
        if (canonical) {
            if (!is.null(N))
                mu_y <- N * mu_y
            sc <- numeric(ncx)
            for (l in seq_len(ncx)) {
                cc <- if (is.null(weights)) {
                    rowsum(X[, l] * mu_y, id, reorder = FALSE)
                } else {
                    weights * rowsum(X[, l] * mu_y, id, reorder = FALSE)
                }
                sc[l] <- sum(c((cc * p_by) %*% wGH), na.rm = TRUE)
            }
            if (is.null(weights)) - Xty + sc else - Xty_weights + sc
        } else {
            var <- var_fun(mu_y)
            deriv <- mu.eta_fun(eta_y)
            z <- if (!is.null(N)) (y[, 1] - N * mu_y) * deriv / var else (y - mu_y) * deriv / var
            for (l in seq_len(ncx)) {
                cc <- if (is.null(weights)) {
                    rowsum(X[, l] * z, id, reorder = FALSE)
                } else {
                    weights * rowsum(X[, l] * z, id, reorder = FALSE)
                }
                sc[l] <- sum(c((cc * p_by) %*% wGH), na.rm = TRUE)
            }
            - sc
        }
    }
    if (penalized) {
        pen_invSigma_betas <- betas * diag(pen_invSigma) / pen_df
        fact <- (pen_df + ncx) / c(1 + crossprod(betas, pen_invSigma_betas))
        out <- out + pen_invSigma_betas * fact
    }
    out
}

# score_phis
# Computes the score (gradient) of the negative log-likelihood with respect to the
# dispersion/shape parameters phis. Used during the EM Newton-Raphson update for phis.
#
# If score_phis_fun is provided (analytic), it is used directly; otherwise, central
# differences on log_dens are used to approximate the score numerically.
# The expectation over the posterior is computed using p_by and wGH.
#
# Arguments:
#   phis:           current dispersion parameter vector
#   y:              response vector or matrix
#   X:              fixed-effects design matrix
#   betas:          current fixed-effects coefficients
#   Ztb:            pre-computed Z * b (from GHfun)
#   offset:         offset vector (NULL if not applicable)
#   weights:        group weights (NULL if unweighted)
#   eta_zi:         zero-inflation linear predictor (NULL if not ZI)
#   id:             integer vector of group indices
#   p_by:           n x k^q matrix of posterior weights from EM E-step
#   log_dens:       log-density function of the family
#   mu_fun:         inverse-link function
#   wGH:            quadrature weights vector
#   score_phis_fun: analytic score function w.r.t. phis, or NULL for numerical diff
#
# Returns:
#   A numeric vector of length n_phis (gradient of negative log-likelihood w.r.t. phis).
score_phis <- function (phis, y, X, betas, Ztb, offset, weights, eta_zi, id, p_by,
                        log_dens, mu_fun, wGH, score_phis_fun) {
    eta_y <- as.vector(X %*% betas) + Ztb
    if (!is.null(offset))
        eta_y <- eta_y + offset
    if (is.null(score_phis_fun)) {
        n_phis <- length(phis)
        sc <- numeric(n_phis)
        for (i in seq_len(n_phis)) {
            phis1 <- phis2 <- phis
            phis1[i] <- phis1[i] + 1e-03
            phis2[i] <- phis2[i] - 1e-03
            l1 <- log_dens(y, eta_y, mu_fun, phis1, eta_zi)
            l2 <- log_dens(y, eta_y, mu_fun, phis2, eta_zi)
            z <- (l1 - l2) / (phis1[i] - phis2[i])
            cc <- if (is.null(weights)) {
                c((rowsum(z, id, reorder = FALSE) * p_by) %*% wGH)
            } else {
                weights * c((rowsum(z, id, reorder = FALSE) * p_by) %*% wGH)
            }
            sc[i] <- sum(cc, na.rm = TRUE)
        }
        - sc
    } else {
        mu_y <- mu_fun(eta_y)
        z <- score_phis_fun(y, mu_y, phis, eta_zi)
        -sum(c((rowsum(z, id, reorder = FALSE) * p_by) %*% wGH), na.rm = TRUE)
    }
}

# score_gammas
# Computes the score (gradient) of the negative log-likelihood with respect to the
# zero-part (zero-inflation) fixed-effects coefficients gammas. Used during the EM
# Newton-Raphson update for gammas.
#
# Recomputes eta_zi from the current gammas (since gammas are being updated), then uses
# either the analytic score_eta_zi_fun or central differences on log_dens w.r.t. eta_zi.
# The posterior expectation is computed using p_by and wGH.
#
# Arguments:
#   gammas:           current zero-part fixed-effects coefficient vector
#   y:                response vector or matrix
#   X:                main fixed-effects design matrix
#   betas:            current main fixed-effects coefficients
#   Ztb:              pre-computed Z * b (from GHfun)
#   offset:           main offset vector (NULL if not applicable)
#   weights:          group weights (NULL if unweighted)
#   X_zi:             zero-inflation fixed-effects design matrix
#   Z_zi:             zero-inflation random-effects design matrix (NULL if no ZI random effects)
#   Z_zitb:           pre-computed Z_zi * b (from GHfun, NULL if no ZI random effects)
#   offset_zi:        zero-inflation offset vector (NULL if not applicable)
#   log_dens:         log-density function of the family
#   score_eta_zi_fun: analytic score w.r.t. eta_zi, or NULL for numerical diff
#   phis:             current dispersion parameters (NULL if not applicable)
#   mu_fun:           inverse-link function
#   p_by:             n x k^q matrix of posterior weights from EM E-step
#   wGH:              quadrature weights vector
#   id:               integer vector of group indices
#
# Returns:
#   A numeric vector of length ncx_zi (gradient of negative log-lik. w.r.t. gammas).
score_gammas <- function (gammas, y, X, betas, Ztb, offset, weights, X_zi, Z_zi, Z_zitb, offset_zi,
                          log_dens, score_eta_zi_fun, phis, mu_fun, p_by, wGH, id) {
    eta_y <- as.vector(X %*% betas) + Ztb
    if (!is.null(offset))
        eta_y <- eta_y + offset
    eta_zi <- as.vector(X_zi %*% gammas)
    if (!is.null(Z_zi))
        eta_zi <- eta_zi + Z_zitb
    if (!is.null(offset_zi))
        eta_zi <- eta_zi + offset_zi
    mu_y <- mu_fun(eta_y)
    z <- if (!is.null(score_eta_zi_fun)) {
        score_eta_zi_fun(y, mu_y, phis, eta_zi)
    } else {
        l1 <- log_dens(y, eta_y, mu_fun, phis, eta_zi + 1e-03)
        l2 <- log_dens(y, eta_y, mu_fun, phis, eta_zi - 1e-03)
        (l1 - l2) / (2 * 1e-03)
    }
    ncx_zi <- ncol(X_zi)
    sc <- numeric(ncx_zi)
    for (l in seq_len(ncx_zi)) {
        cc <- if (is.null(weights)) {
            drop(rowsum(X_zi[, l] * z, id, reorder = FALSE))
        } else {
            weights * drop(rowsum(X_zi[, l] * z, id, reorder = FALSE))
        }
        sc[l] <- sum(c((cc * p_by) %*% wGH), na.rm = TRUE)
    }
    - sc
}

# binomial_log_dens
# Computes the log-density of the binomial distribution for GLMMadaptive's internal use.
# Handles both binary outcomes (y is a 0/1 vector) and binomial counts (y is a 2-column
# matrix with successes and failures). Attaches mu_y as an attribute for reuse in
# score computations.
#
# Arguments:
#   y:      response: numeric vector (binary) or 2-column matrix (successes, failures)
#   eta:    linear predictor vector
#   mu_fun: inverse-link function (e.g., plogis for logit link)
#   phis:   ignored (no dispersion parameter for binomial)
#   eta_zi: ignored (no zero-inflation for this function)
#
# Returns:
#   Numeric vector of log-density values; attribute "mu_y" contains the fitted means.
binomial_log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
    mu_y <- mu_fun(eta)
    out <- if (NCOL(y) == 2L) {
        dbinom(y[, 1L], y[, 1L] + y[, 2L], mu_y, TRUE)
    } else {
        dbinom(y, 1L, mu_y, TRUE)
    }
    attr(out, "mu_y") <- mu_y
    out
}

# poisson_log_dens
# Computes the log-density of the Poisson distribution for GLMMadaptive's internal use.
# Implements log p(y | mu) = y*log(mu) - mu - log(y!) directly (equivalent to dpois but
# vectorized over quadrature points). Attaches mu_y as an attribute for reuse.
#
# Arguments:
#   y:      non-negative integer response vector
#   eta:    linear predictor vector (log-scale for Poisson)
#   mu_fun: inverse-link function (exp for log link)
#   phis:   ignored
#   eta_zi: ignored
#
# Returns:
#   Numeric vector of log-density values; attribute "mu_y" contains the fitted means.
poisson_log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
    mu_y <- mu_fun(eta)
    out <- y * log(mu_y) - mu_y - lgamma(y + 1)
    attr(out, "mu_y") <- mu_y
    out
}

# gamma_log_dens
# NOTE: This is a standalone log-density function separate from Gamma.fam(). It uses a
# different parameterization: shape = mu/scale, scale = exp(phis). This is a legacy
# function; the preferred approach for Gamma models uses Gamma.fam() which parameterizes
# as shape = phi, scale = mu/phi (where phi = exp(phis)), providing a cleaner interpretation
# of phi as the shape/concentration parameter.
#
# Arguments:
#   y:      positive real response vector
#   eta:    linear predictor (log-scale, so mu = exp(eta))
#   mu_fun: inverse-link function
#   phis:   log of the scale parameter
#   eta_zi: ignored
#
# Returns:
#   Numeric vector of log-density values; attribute "mu_y" contains the fitted means.
gamma_log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
    mu_y <- mu_fun(eta)
    scale <- exp(phis)
    out <- dgamma(y, shape = mu_y / scale, scale = scale, log = TRUE)
    attr(out, "mu_y") <- mu_y
    out
}

# negative.binomial_log_dens
# Computes the log-density of the negative binomial distribution (NB2 parameterization)
# where the variance is mu + mu^2/size and size = exp(phis). This standalone function
# is used for computing initial values; the full family object (negative.binomial()) also
# defines analytic score functions.
#
# log p(y | mu, size) = lgamma(y+size) - lgamma(size) - lgamma(y+1)
#                     + size*log(size/(mu+size)) + y*log(mu/(mu+size))
#
# Arguments:
#   y:      non-negative integer response vector
#   eta:    log-scale linear predictor
#   mu_fun: inverse-link function (exp for log link)
#   phis:   log of the size (overdispersion) parameter
#   eta_zi: ignored
#
# Returns:
#   Numeric vector of log-density values; attribute "mu_y" contains the fitted means.
negative.binomial_log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
    phis <- exp(phis)
    mu <- mu_fun(eta)
    log_mu_phis <- log(mu + phis)
    comp1 <- lgamma(y + phis) - lgamma(phis) - lgamma(y + 1)
    comp2 <- phis * log(phis) - phis * log_mu_phis
    comp3 <- y * (log(mu) - log_mu_phis)
    out <- comp1 + comp2 + comp3
    attr(out, "mu_y") <- mu
    out
}

# negative.binomial
# Creates a family object for the negative binomial (NB2) distribution with log link.
# The NB2 parameterization has variance = mu + mu^2/size where size = exp(phis).
# Provides analytic score_eta_fun and score_phis_fun for efficient optimization.
# Note: the first make.link() call appears to be a duplicate.
#
# Returns:
#   A list of class "family" with components: family, link, linkfun, linkinv, log_dens,
#   variance, score_eta_fun, score_phis_fun.
negative.binomial <- function () {
    stats <- make.link("log")
    stats <- make.link(link = "log")
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        # the log density function
        phis <- exp(phis)
        mu <- mu_fun(eta)
        log_mu_phis <- log(mu + phis)
        comp1 <- lgamma(y + phis) - lgamma(phis) - lgamma(y + 1)
        comp2 <- phis * log(phis) - phis * log_mu_phis
        comp3 <- y * (log(mu) - log_mu_phis)
        out <- comp1 + comp2 + comp3
        attr(out, "mu_y") <- mu
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        # the derivative of the log density w.r.t. mu
        phis <- exp(phis)
        #mu_phis <- mu + phis
        #comp2 <- - phis / mu_phis
        #comp3 <- y / mu - y / mu_phis
        ## the derivative of mu w.r.t. eta (this depends on the chosen link function)
        #mu.eta <- mu
        #(comp2 + comp3) * mu.eta
        mu.mu_phis <- mu / (mu + phis)
        - phis * mu.mu_phis + y * (1 - mu.mu_phis) 
    }
    score_phis_fun <- function (y, mu, phis, eta_zi) {
        # the derivative of the log density w.r.t. phis
        phis <- exp(phis)
        mu_phis <- mu + phis
        #comp1 <- digamma(y + phis) - digamma(phis)
        #comp2 <- log(phis) + 1 - log(mu_phis) - phis / mu_phis
        #comp3 <- - y / mu_phis
        #(comp1 + comp2 + comp3) * phis
        y_phis <- y + phis
        comp1 <- log(phis) + 1 - digamma(phis)
        comp2 <- digamma(y_phis)
        comp3 <- - log(mu_phis) - y_phis / mu_phis
        (comp1 + comp2 + comp3) * phis
    }
    structure(list(family = "negative binomial", link = stats$name, 
                   linkfun = stats$linkfun, linkinv = stats$linkinv, log_dens = log_dens,
                   variance = function (mu, theta) mu + mu^2 / theta,
                   score_eta_fun = score_eta_fun, score_phis_fun = score_phis_fun),
              class = "family")
}

# zi.poisson
# Creates a family object for the zero-inflated Poisson (ZIP) distribution with log link.
# The ZIP mixes a point mass at zero with a Poisson distribution:
#   P(Y=0) = pi + (1-pi)*exp(-mu)
#   P(Y=y) = (1-pi)*exp(-mu)*mu^y/y!  for y > 0
# where pi = plogis(eta_zi) is the probability of a structural zero.
# Provides analytic score_eta_fun and score_eta_zi_fun.
#
# Returns:
#   A list of class "family" with components: family, link, linkfun, linkinv, log_dens,
#   score_eta_fun, score_eta_zi_fun.
zi.poisson <- function () {
    stats <- make.link(link = "log")
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        # the log density function
        ind_y0 <- y == 0
        ind_y1 <- y > 0
        mu <- as.matrix(mu_fun(eta))
        lambda <- as.matrix(exp(eta_zi))
        mu0 <- mu[ind_y0, ]
        lambda0 <- lambda[ind_y0, ]
        mu1 <- mu[ind_y1, ]
        out <- as.matrix(eta)
        out[ind_y0, ] <- log(lambda0 + exp(-mu0))
        out[ind_y1, ] <- y[ind_y1] * log(mu1) - mu1 - lgamma(y[ind_y1] + 1)
        out <- out - log(1 + drop(lambda))
        attr(out, "mu_y") <- mu
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        ind_y0 <- y == 0
        ind_y1 <- y > 0
        mu <- as.matrix(mu)
        lambda <- as.matrix(exp(eta_zi))
        mu0 <- mu[ind_y0, ]
        lambda0 <- lambda[ind_y0, ]
        mu1 <- mu[ind_y1, ]
        out <- mu
        out[ind_y0, ] <- - mu0 / (lambda0 * exp(mu0) + 1)
        out[ind_y1, ] <- y[ind_y1] - mu1
        out
    }
    score_eta_zi_fun <- function (y, mu, phis, eta_zi) {
        ind_y0 <- y == 0
        mu <- as.matrix(mu)
        eta_zi <- as.matrix(eta_zi)
        lambda <- exp(eta_zi)
        lambda0_exp_mu0 <- exp(eta_zi[ind_y0, ] + mu[ind_y0, ])
        lambda0_exp_mu0[lambda0_exp_mu0 == Inf] <- 1e200
        out <- matrix(- lambda / (1 + lambda), nrow = nrow(mu), ncol = ncol(mu))
        out[ind_y0, ] <- out[ind_y0, ] + 
            drop(lambda0_exp_mu0 / (lambda0_exp_mu0 + 1))
        out
    }
    structure(list(family = "zero-inflated poisson", link = stats$name, 
                   linkfun = stats$linkfun, linkinv = stats$linkinv, log_dens = log_dens,
                   score_eta_fun = score_eta_fun,
                   score_eta_zi_fun = score_eta_zi_fun),
              class = "family")
}

# zi.negative.binomial
# Creates a family object for the zero-inflated negative binomial (ZINB) distribution
# with log link. Mixes a point mass at zero with a NB2 distribution:
#   P(Y=0) = pi + (1-pi) * NB(0; mu, size)
#   P(Y=y) = (1-pi) * NB(y; mu, size)  for y > 0
# where pi = plogis(eta_zi) is the structural zero probability and size = exp(phis).
# Provides analytic score_eta_fun, score_eta_zi_fun, and score_phis_fun.
#
# Returns:
#   A list of class "family" with the standard components plus all three score functions.
zi.negative.binomial <- function () {
    stats <- make.link(link = "log")
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        # the log density function
        # NB part
        phis <- exp(phis)
        mu <- mu_fun(eta)
        log_mu_phis <- log(mu + phis)
        comp1 <- lgamma(y + phis) - lgamma(phis) - lgamma(y + 1)
        comp2 <- phis * log(phis) - phis * log_mu_phis
        comp3 <- y * (log(mu) - log_mu_phis)
        out <- as.matrix(comp1 + comp2 + comp3)
        # ZI part
        ind_y0 <- y == 0
        ind_y1 <- y > 0
        pis <- as.matrix(plogis(eta_zi))
        # combined
        out[ind_y0, ] <- log(pis[ind_y0, ] + (1 - pis[ind_y0, ]) * exp(out[ind_y0, ]))
        out[ind_y1, ] <- log(1 - pis[ind_y1, ]) + out[ind_y1, ]
        attr(out, "mu_y") <- mu
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        # NB part
        phis <- exp(phis)
        mu <- as.matrix(mu)
        mu.mu_phis <- mu / (mu + phis)
        out <- - phis * mu.mu_phis + y * (1 - mu.mu_phis) 
        # ZI part
        ind_y0 <- y == 0
        lambda <- exp(as.matrix(eta_zi)[ind_y0, ])
        mu0 <- mu[ind_y0, ]
        t <- phis / (phis + mu0)
        den <- (lambda + t^phis) * (mu0 + phis)^2
        out[ind_y0, ] <- - phis^2 * t^(phis - 1) * mu0 / den
        out
    }
    score_eta_zi_fun <- function (y, mu, phis, eta_zi) {
        phis <- exp(phis)
        ind_y0 <- y == 0
        ind_y1 <- y > 0
        # NB part
        mu <- as.matrix(mu)
        lambda <- as.matrix(exp(eta_zi))
        out <- mu
        out[ind_y1, ] <- - lambda[ind_y1, ] / (1 + lambda[ind_y1, ])
        # ZI part
        t <- phis / (phis + mu[ind_y0, ])
        lambda0 <- lambda[ind_y0, ]
        out[ind_y0, ] <- lambda0 / (lambda0 + t^phis) - lambda0 / (1 + lambda0)
        out
    }
    score_phis_fun <- function (y, mu, phis, eta_zi) {
        # NB part
        phis <- exp(phis)
        mu <- as.matrix(mu)
        mu_phis <- mu + phis
        comp1 <- digamma(y + phis) - digamma(phis)
        comp2 <- log(phis) + 1 - log(mu_phis) - phis / mu_phis
        comp3 <- - y / mu_phis
        out <- (comp1 + comp2 + comp3) * phis
        # ZI part
        ind_y0 <- y == 0
        lambda <- as.matrix(exp(eta_zi))
        t <- phis / (phis + mu[ind_y0, ])
        t_phis <- t^phis
        out[ind_y0, ] <- t_phis * (log(t) + 1 - t) * phis / (lambda[ind_y0, ] + t_phis)
        out
    }
    structure(list(family = "zero-inflated negative binomial", link = stats$name, 
                   linkfun = stats$linkfun, linkinv = stats$linkinv, log_dens = log_dens,
                   score_eta_fun = score_eta_fun,
                   score_eta_zi_fun = score_eta_zi_fun,
                   score_phis_fun = score_phis_fun),
              class = "family")
}

# hurdle.poisson
# Creates a family object for the hurdle Poisson distribution with log link.
# A hurdle model treats zeros and positives as two separate processes:
#   P(Y=0) = plogis(eta_zi)    [zero part: logistic probability of a zero]
#   P(Y=y | Y>0) ~ truncated Poisson(mu) / (1 - exp(-mu))  for y > 0
# Unlike zero-inflation, there are no "structural zeros" from the count process.
# Also provides a simulate() function for generating data from the fitted model.
#
# Returns:
#   A list of class "family" with components including variance, score_eta_fun,
#   score_eta_zi_fun, and simulate.
hurdle.poisson <- function () {
    stats <- make.link("log")
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        # binary indicator for y > 0
        ind <- y > 0
        # non-zero part
        eta <- as.matrix(eta)
        mu <- mu_fun(eta)
        eta_zi <- as.matrix(eta_zi)
        out <- eta
        out[ind, ] <- plogis(eta_zi[ind, ], lower.tail = FALSE, log.p = TRUE) - 
            mu[ind, ] + y[ind] * eta[ind, ] - log(- expm1(-mu[ind, ])) - lgamma(y[ind] + 1)
        # zero part
        out[!ind, ] <- plogis(eta_zi[!ind, ], log.p = TRUE)
        attr(out, "mu_y") <- mu
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        # binary indicator for y > 0
        ind <- y > 0
        # non-zero part
        mu <- as.matrix(mu)
        mu_ind <- mu[ind, ]
        out <- mu
        out[!ind, ] <- 0
        out[ind, ] <- - mu_ind + y[ind] + (exp(-mu_ind) * mu_ind) / expm1(-mu_ind) 
        out
    }
    score_eta_zi_fun <- function (y, mu, phis, eta_zi) {
        ind <- y > 0
        probs <- plogis(as.matrix(eta_zi))
        out <- 1 - probs
        out[ind, ] <- - probs[ind, ]
        out
    }
    simulate <- function (n, mu, phis, eta_zi) {
        y <- qpois(runif(n, ppois(0, mu), 1), mu)
        y[as.logical(rbinom(n, 1, plogis(eta_zi)))] <- 0
        y
    }
    structure(list(family = "hurdle poisson", link = stats$name, 
                   linkfun = stats$linkfun, linkinv = stats$linkinv, log_dens = log_dens,
                   variance = function (mu) (mu + mu^2)/(1 - exp(-mu)) - mu^2/((1 - exp(-mu))^2),
                   score_eta_fun = score_eta_fun, score_eta_zi_fun = score_eta_zi_fun,
                   simulate = simulate),
              class = "family")
}

# hurdle.negative.binomial
# Creates a family object for the hurdle negative binomial distribution with log link.
# Combines a logistic model for zero/non-zero with a zero-truncated NB2 for positive counts:
#   P(Y=0) = plogis(eta_zi)
#   P(Y=y | Y>0) ~ truncated NB(mu, size) / (1 - NB(0; mu, size))  for y > 0
# size = exp(phis). Provides analytic score functions and simulate().
#
# Returns:
#   A list of class "family" with score_eta_fun, score_eta_zi_fun, score_phis_fun, simulate.
hurdle.negative.binomial <- function () {
    stats <- make.link("log")
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        phis <- exp(phis)
        # binary indicator for y > 0
        ind <- y > 0
        # non-zero part
        eta <- as.matrix(eta)
        mu <- mu_fun(eta)
        log_mu_phis <- log(mu + phis)
        eta_zi <- as.matrix(eta_zi)
        out <- eta
        comp1 <- lgamma(y + phis) - lgamma(phis) - lgamma(y + 1)
        comp2 <- phis * log(phis) - phis * log_mu_phis
        comp3 <- y * log(mu) - y * log_mu_phis
        log_g <- comp1 + comp2 + comp3
        comp4 <- log(1 - (1 + mu / phis)^(-phis))
        out[ind, ] <- plogis(eta_zi[ind, ], lower.tail = FALSE, log.p = TRUE) + 
            log_g[ind, ] - comp4[ind, ]
        # zero part
        out[!ind, ] <- plogis(eta_zi[!ind, ], log.p = TRUE)
        attr(out, "mu_y") <- mu
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        phis <- exp(phis)
        # binary indicator for y > 0
        ind <- y > 0
        # non-zero part
        mu <- as.matrix(mu)
        mu_phis <- mu + phis
        comp2 <- - phis / mu_phis
        comp3 <- y / mu - y / mu_phis
        k <- (1 + mu / phis)
        comp4 <- k^(- phis - 1) / (1 - k^(-phis))
        mu.eta <- mu
        out <- (comp2 + comp3 - comp4) * mu.eta
        out[!ind, ] <- 0
        out
    }
    score_eta_zi_fun <- function (y, mu, phis, eta_zi) {
        ind <- y > 0
        probs <- plogis(as.matrix(eta_zi))
        out <- 1 - probs
        out[ind, ] <- - probs[ind, ]
        out
    }
    score_phis_fun <- function (y, mu, phis, eta_zi) {
        ind_y0 <- y == 0
        phis <- exp(phis)
        mu <- as.matrix(mu)
        mu_phis <- mu + phis
        comp1 <- digamma(y + phis) - digamma(phis)
        comp2 <- log(phis) + 1 - log(mu_phis) - phis / mu_phis
        comp3 <- - y / mu_phis
        k <- mu / phis
        k1 <- 1 + k
        comp4 <- k1^(-phis) * (k / k1 - log(k1)) / (1 - k1^(-phis))
        out <- (comp1 + comp2 + comp3 + comp4) * phis
        out[ind_y0, ] <- 0
        out
    }
    simulate <- function (n, mu, phis, eta_zi) {
        y <- qnbinom(runif(n, pnbinom(0, mu = mu, size = exp(phis)), 1), 
                     mu = mu, size = exp(phis))
        y[as.logical(rbinom(n, 1, plogis(eta_zi)))] <- 0
        y
    }
    structure(list(family = "hurdle negative binomial", link = stats$name, 
                   linkfun = stats$linkfun, linkinv = stats$linkinv, log_dens = log_dens,
                   score_eta_fun = score_eta_fun, score_eta_zi_fun = score_eta_zi_fun,
                   score_phis_fun = score_phis_fun,
                   simulate = simulate),
              class = "family")
}

# zi.binomial
# Creates a family object for the zero-inflated binomial distribution with logit link.
# Mixes a point mass at zero with a binomial distribution:
#   P(Y=0) = pi + (1-pi) * Binomial(0; N, mu)
#   P(Y=y | Y>0) = (1-pi) * Binomial(y; N, mu)
# where pi = plogis(eta_zi). Supports both binary (N=1) and grouped binomial (N>1) responses.
# Provides analytic score_eta_fun and score_eta_zi_fun; no dispersion parameter.
#
# Returns:
#   A list of class "family" with score_eta_fun, score_eta_zi_fun, score_phis_fun = NULL.
zi.binomial <- function () {
    stats <- make.link(link = "logit")
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        # the log density function
        # Binomial part
        mu <- mu_fun(eta)
        y <- as.matrix(y)
        N <- if (ncol(y) == 2L) y[, 1L] + y[, 2L] else rep(1L, nrow(y))
        out <- as.matrix(dbinom(y[, 1L], N, mu, TRUE))
        # ZI part
        ind_y0 <- y[, 1L] == 0
        ind_y1 <- y[, 1L] > 0
        pis <- as.matrix(plogis(eta_zi))
        # combined
        out[ind_y0, ] <- log(pis[ind_y0, ] + (1 - pis[ind_y0, ]) * exp(out[ind_y0, ]))
        out[ind_y1, ] <- log(1 - pis[ind_y1, ]) + out[ind_y1, ]
        attr(out, "mu_y") <- mu
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        # Binomial part
        mu <- as.matrix(mu)
        y <- as.matrix(y)
        N <- if (ncol(y) == 2L) y[, 1L] + y[, 2L] else rep(1L, nrow(y))
        out <- y[, 1L] * (1 - mu) - (N - y[, 1L]) * mu
        # ZI part
        ind_y0 <- y[, 1L] == 0
        eta_zi <- as.matrix(eta_zi)
        pis <- plogis(eta_zi[ind_y0, ])
        mu0 <- mu[ind_y0, ]
        N0 <- N[ind_y0]
        pis1 <- 1 - pis
        den <- pis + pis1 * (1 - mu0)^N0
        out[ind_y0, ] <- - (N0 * mu0 * pis1 * (1 - mu0)^N0) / den
        out
    }
    score_eta_zi_fun <- function (y, mu, phis, eta_zi) {
        y <- as.matrix(y)
        N <- if (ncol(y) == 2L) y[, 1L] + y[, 2L] else rep(1L, nrow(y))
        ind_y0 <- y[, 1L] == 0
        ind_y1 <- y[, 1L] > 0
        pis <- as.matrix(plogis(eta_zi))
        mu <- as.matrix(mu)
        # Binomial part
        out <- mu
        out[ind_y1, ] <- - pis[ind_y1, ]
        # ZI part
        mu0 <- mu[ind_y0, ]
        N0 <- N[ind_y0]
        pis1 <- 1 - pis[ind_y0, ]
        FF <- (1 - mu0)^N0
        den <- pis[ind_y0, ] + pis1 * FF
        out[ind_y0, ] <- pis[ind_y0, ] * pis1 * (1 - FF) / den
        out
    }
    structure(list(family = "zero-inflated binomial", link = stats$name, 
                   linkfun = stats$linkfun, linkinv = stats$linkinv, log_dens = log_dens,
                   score_eta_fun = score_eta_fun,
                   score_eta_zi_fun = score_eta_zi_fun,
                   score_phis_fun = NULL),
              class = "family")
}

# hurdle.lognormal
# Creates a family object for the hurdle log-normal distribution with identity link
# (on the log-scale). Combines a logistic model for structural zeros with a log-normal
# model for positive values:
#   P(Y=0) = plogis(eta_zi)
#   Y | Y>0 ~ LogNormal(eta, sigma^2)  where sigma = exp(phis)
# Note: the identity link means eta = E[log(Y) | Y>0] (not E[Y]).
# Provides analytic score functions and simulate().
#
# Returns:
#   A list of class "family" with score_eta_fun, score_eta_zi_fun, score_phis_fun, simulate.
hurdle.lognormal <- function () {
    stats <- make.link("identity")
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        sigma <- exp(phis)
        # binary indicator for y > 0
        ind <- y > 0
        # non-zero part
        eta <- as.matrix(eta)
        eta_zi <- as.matrix(eta_zi)
        out <- eta
        out[ind, ] <- plogis(eta_zi[ind, ], lower.tail = FALSE, log.p = TRUE) + 
            dnorm(x = log(y[ind]), mean = eta[ind, ], sd = sigma, log = TRUE)
        # zero part
        out[!ind, ] <- plogis(eta_zi[!ind, ], log.p = TRUE)
        attr(out, "mu_y") <- eta
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        sigma <- exp(phis)
        # binary indicator for y > 0
        ind <- y > 0
        # non-zero part
        eta <- as.matrix(mu)
        out <- eta
        out[!ind, ] <- 0
        out[ind, ] <- (log(y[ind]) - eta[ind, ]) / sigma^2
        out
    }
    score_eta_zi_fun <- function (y, mu, phis, eta_zi) {
        ind <- y > 0
        probs <- plogis(as.matrix(eta_zi))
        out <- 1 - probs
        out[ind, ] <- - probs[ind, ]
        out
    }
    score_phis_fun <- function (y, mu, phis, eta_zi) {
        sigma <- exp(phis)
        # binary indicator for y > 0
        ind <- y > 0
        # non-zero part
        eta <- as.matrix(mu)
        out <- eta
        out[!ind, ] <- 0
        out[ind, ] <- - 1 + (log(y[ind]) - eta[ind, ])^2 / sigma^2
        out
    }
    simulate <- function (n, mu, phis, eta_zi) {
        y <- rlnorm(n = n, meanlog = mu, sdlog = exp(phis))
        y[as.logical(rbinom(n, 1, plogis(eta_zi)))] <- 0
        y
    }
    structure(list(family = "hurdle log-normal", link = stats$name, 
                   linkfun = stats$linkfun, linkinv = stats$linkinv, log_dens = log_dens,
                   score_eta_fun = score_eta_fun, score_eta_zi_fun = score_eta_zi_fun,
                   score_phis_fun = score_phis_fun, simulate = simulate),
              class = "family")
}

# beta.fam
# Creates a family object for the beta distribution with logit link, using the
# mean-dispersion parameterization: Y ~ Beta(mu*phi, (1-mu)*phi) where phi = exp(phis)
# is the precision (concentration) parameter. The mean is mu = plogis(eta) and
# variance = mu*(1-mu)/(phi+1).
# Provides analytic score_eta_fun, score_phis_fun, and simulate().
#
# Returns:
#   A list of class "family" with variance, log_dens, simulate, score_eta_fun, score_phis_fun.
beta.fam <- function () {
    stats <- make.link("logit")
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        # the log density function
        phi <- exp(phis)
        mu <- mu_fun(eta)
        mu_phi <- mu * phi
        comp1 <- lgamma(phi) - lgamma(mu_phi)
        comp2 <- (mu_phi - 1) * log(y) - lgamma(phi - mu_phi)
        comp3 <- (phi - mu_phi - 1) * log(1 - y)
        out <- comp1 + comp2 + comp3
        attr(out, "mu_y") <- mu
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        # the derivative of the log density w.r.t. mu
        phi <- exp(phis)
        mu_phi <- mu * phi
        comp1 <- - digamma(mu_phi) * phi
        comp2 <- phi * (log(y) + digamma(phi - mu_phi))
        comp3 <- - phi * log(1 - y)
        # the derivative of mu w.r.t. eta (this depends on the chosen link function)
        mu.eta <- mu - mu * mu
        (comp1 + comp2 + comp3) * mu.eta
    }
    score_phis_fun <- function (y, mu, phis, eta_zi) {
        phi <- exp(phis)
        mu_phi <- mu * phi
        mu1 <- 1 - mu
        comp1 <- digamma(phi) - digamma(mu_phi) * mu
        comp2 <- mu * log(y) - digamma(phi - mu_phi) * mu1
        comp3 <- log(1 - y) * mu1
        (comp1 + comp2 + comp3) * phi
    }
    simulate <- function (n, mu, phis, eta_zi) {
        phi <- exp(phis)
        rbeta(n, shape1 = mu * phi, shape2 = phi * (1 - mu))
    }
    structure(list(family = "beta", link = stats$name, linkfun = stats$linkfun,
                   linkinv = stats$linkinv, variance = function (mu) mu * (1 - mu), 
                   log_dens = log_dens, simulate = simulate,
                   score_eta_fun = score_eta_fun, score_phis_fun = score_phis_fun),
              class = "family")
}

# hurdle.beta.fam
# Creates a family object for the hurdle beta distribution with logit link.
# Combines a logistic model for structural zeros with a beta distribution for positive values:
#   P(Y=0) = plogis(eta_zi)
#   Y | Y>0 ~ Beta(mu*phi, (1-mu)*phi)  where phi = exp(phis)
# Note: in this family mu_fun is applied to eta but the stored "mu_y" is eta itself
# (the linear predictor), not the mean mu. This is because the mean of the hurdle beta
# is only meaningful conditional on Y>0.
# Provides analytic score functions and simulate().
#
# Returns:
#   A list of class "family" with score_eta_fun, score_eta_zi_fun, score_phis_fun, simulate.
hurdle.beta.fam <- function () {
    stats <- make.link("logit")
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        phi <- exp(phis)
        # binary indicator for y > 0
        ind <- y > 0
        # non-zero part
        eta <- as.matrix(eta)
        eta_zi <- as.matrix(eta_zi)
        out <- eta
        mu <- mu_fun(eta)
        mu_phi <- mu * phi
        comp1 <- lgamma(phi) - lgamma(mu_phi)
        comp2 <- (mu_phi - 1) * log(y) - lgamma(phi - mu_phi)
        comp3 <- (phi - mu_phi - 1) * log(1 - y)
        out[ind, ] <- plogis(eta_zi[ind, ], lower.tail = FALSE, log.p = TRUE) + 
            comp1[ind, ] + comp2[ind, ] + comp3[ind, ]
        # zero part
        out[!ind, ] <- plogis(eta_zi[!ind, ], log.p = TRUE)
        attr(out, "mu_y") <- eta
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        phi <- exp(phis)
        # binary indicator for y > 0
        ind <- y > 0
        # non-zero part
        mu <- as.matrix(mu)
        mu_phi <- mu * phi
        out <- mu
        comp1 <- - digamma(mu_phi) * phi
        comp2 <- phi * (log(y) + digamma(phi - mu_phi))
        comp3 <- - phi * log(1 - y)
        mu.eta <- mu - mu * mu
        out[!ind, ] <- 0
        out[ind, ] <- (comp1[ind, ] + comp2[ind, ] + comp3[ind]) * mu.eta[ind, ]
        out
    }
    score_eta_zi_fun <- function (y, mu, phis, eta_zi) {
        ind <- y > 0
        probs <- plogis(as.matrix(eta_zi))
        out <- 1 - probs
        out[ind, ] <- - probs[ind, ]
        out
    }
    score_phis_fun <- function (y, mu, phis, eta_zi) {
        phi <- exp(phis)
        # binary indicator for y > 0
        ind <- y > 0
        # non-zero part
        mu <- as.matrix(mu)
        mu_phi <- mu * phi
        mu1 <- 1 - mu
        out <- mu
        comp1 <- digamma(phi) - digamma(mu_phi) * mu
        comp2 <- mu * log(y) - digamma(phi - mu_phi) * mu1
        comp3 <- log(1 - y) * mu1
        out[ind, ] <- (comp1[ind, ] + comp2[ind, ] + comp3[ind, ]) * phi
        out[!ind, ] <- 0
        out
    }
    simulate <- function (n, mu, phis, eta_zi) {
        phi <- exp(phis)
        y <- rbeta(n, shape1 = mu * phi, shape2 = phi * (1 - mu))
        y[as.logical(rbinom(n, 1, plogis(eta_zi)))] <- 0
        y
    }
    structure(list(family = "hurdle beta", link = stats$name, 
                   linkfun = stats$linkfun, linkinv = stats$linkinv, log_dens = log_dens,
                   score_eta_fun = score_eta_fun, score_eta_zi_fun = score_eta_zi_fun,
                   score_phis_fun = score_phis_fun, simulate = simulate),
              class = "family")
}

# students.t
# Creates a family object for the Student's-t distribution with user-specified link and
# fixed degrees of freedom df. The t distribution is parameterized as a location-scale t:
#   Y ~ t(eta, sigma^2, df) where sigma = exp(phis)
# This is a heavier-tailed alternative to the normal (Gaussian) for continuous outcomes.
# The df parameter is fixed (not estimated) and captured in the function's closure.
# Provides analytic score_eta_fun, score_phis_fun, and simulate().
#
# Arguments:
#   df:   degrees of freedom (required, must be a positive number)
#   link: link function name (default: "identity")
#
# Returns:
#   A list of class "family" with log_dens, variance, score_eta_fun, score_phis_fun,
#   simulate, and a df component.
students.t <- function (df = stop("'df' must be specified"), link = "identity") {
    .df <- df
    env <- new.env(parent = .GlobalEnv)
    assign(".df", df, envir = env)
    stats <- make.link(link)
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        # the log density function
        sigma <- exp(phis)
        out <- dt(x = (y - eta) / sigma, df = .df, log = TRUE) - log(sigma)
        attr(out, "mu_y") <- eta
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        # the derivative of the log density w.r.t. mu
        sigma2 <- exp(phis)^2
        y_mu <- y - mu
        (y_mu * (.df + 1) / (.df * sigma2)) / (1 + y_mu^2 / (.df * sigma2))
    }
    score_phis_fun <- function (y, mu, phis, eta_zi) {
        sigma <- exp(phis)
        y_mu2_df <- (y - mu)^2 / .df
        (.df + 1) * y_mu2_df * sigma^{-2} / (1 + y_mu2_df / sigma^2) - 1
    }
    simulate <- function (n, mu, phis, eta_zi) {
        phi <- exp(phis)
        mu + phi * rt(n, df = .df)
    }
    environment(log_dens) <- environment(score_eta_fun) <- env
    environment(score_phis_fun) <- environment(simulate) <- env
    structure(list(family = "Student's-t", link = stats$name, linkfun = stats$linkfun,
                   linkinv = stats$linkinv, log_dens = log_dens, 
                   variance = function (mu) rep.int(1, length(mu)),
                   score_eta_fun = score_eta_fun, score_phis_fun = score_phis_fun,
                   simulate = simulate, df = df),
              class = "family")
}

# compoisson
# Creates a family object for the Conway-Maxwell-Poisson (CMP) distribution with log link.
# The CMP generalizes the Poisson by adding a dispersion parameter nu = exp(phis):
#   P(Y=y) = (lambda^y / (y!)^nu) / Z(lambda, nu)
# where Z(lambda, nu) = sum_{j=0}^{max} lambda^j / (j!)^nu is the normalizing constant.
# When nu=1 this reduces to Poisson; nu>1 gives under-dispersion; nu<1 gives over-dispersion.
# The max argument truncates the infinite normalizing constant sum.
# NOTE: compoisson() and compoisson2() appear to have identical implementations. Their
# difference (if any) may have been intended but is not currently reflected in the code.
#
# Arguments:
#   max: truncation point for the normalizing constant Z (default: 100)
#
# Returns:
#   A list of class "family" with log_dens, score_eta_fun, score_phis_fun.
compoisson <- function (max = 100) {
    stats <- make.link("log")
    .max <- max
    env <- new.env(parent = .GlobalEnv)
    assign(".max", max, envir = env)
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        # the log density function
        phis <- exp(phis)
        mu <- mu_fun(eta)
        Z <- function (lambda, nu, sumTo) {
            out <- lambda
            j <- seq(1, sumTo)
            log_lambda <- log(lambda)
            nu_log_factorial <- nu * cumsum(log(j))
            for (i in seq_along(out)) {
                out[i] <- 1 + sum(exp(j * log_lambda[i] - nu_log_factorial))
            }
            out
        }
        out <- y * log(mu) - phis * lgamma(y + 1) - log(Z(mu, phis, .max))
        attr(out, "mu_y") <- mu
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        phi <- exp(phis)
        Y <- function (lambda, nu, sumTo) {
            out <- lambda
            j <- seq(1, sumTo)
            log_lambda <- log(lambda)
            log_j <- log(j)
            nu_log_factorial <- nu * cumsum(log_j)
            for (i in seq_along(out)) {
                F1 <- j * log_lambda[i] - nu_log_factorial
                num <- sum(exp(log_j + F1))
                den <- 1 + sum(exp(F1))
                out[i] <- num / den
            }
            out
        }
        y - Y(mu, phi, .max)
    }
    score_phis_fun <- function (y, mu, phis, eta_zi) {
        phi <- exp(phis)
        W <- function (lambda, nu, sumTo) {
            out <- lambda
            j <- seq(1, sumTo)
            log_lambda <- log(lambda)
            log_factorial <- cumsum(log(j))
            log_log_factorial <- log(log_factorial)
            nu_log_factorial <- nu * log_factorial
            for (i in seq_along(out)) {
                num <- sum(exp(j * log_lambda[i] + log_log_factorial - nu_log_factorial))
                den <- 1 + sum(exp(j * log_lambda[i] - nu_log_factorial))
                out[i] <- num / den
            }
            out
        }
        (- lgamma(y + 1) + W(mu, phi, .max)) * phi
    }
    simulate <- function (n, mu, phis, eta_zi) {
        phi <- exp(phis)
    }
    structure(list(family = "Conway Maxwell Poisson", link = stats$name, 
                   linkfun = stats$linkfun, linkinv = stats$linkinv, log_dens = log_dens,
                   score_eta_fun = score_eta_fun, score_phis_fun = score_phis_fun),
              class = "family")
}

# unit.lindley
# Creates a family object for the unit Lindley distribution with logit link.
# The unit Lindley distribution is defined on (0,1) and is parameterized by theta > 0:
#   f(y) = theta^2 / (1+theta) * (1+y) / (1-y)^3 * exp(-theta * y / (1-y))
# where the mean mu = 1/(1+theta), linked via logit(mu) = eta.
# NOTE: This family is currently UNAVAILABLE (the function immediately throws an error).
# The simulate() function also returns NA. The score_eta_fun is implemented but inactive.
#
# Returns:
#   (never returns; stops with an error message)
unit.lindley <- function () {
    stop("currently the 'unit.lindley()' family is unavailable.")
    stats <- make.link("logit")
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        # the log density function
        # you link logit(mu) to covariates
        # where mu = (1 / (1 + theta))
        mu <- as.matrix(mu_fun(eta))
        theta <- 1 / mu - 1
        comp1 <- 2 * log(theta) - log(1 + theta)
        comp2 <- - 3 * log(1 - y) 
        comp3 <- - (theta * y) / (1 - y)
        out <- comp1 + comp2 + comp3
        attr(out, "mu_y") <- mu
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        mu <- as.matrix(mu)
        theta <- 1 / mu - 1
        # the derivative of the log density w.r.t. theta
        comp1 <- 2 / theta - 1 / (1 + theta)
        comp3 <- - y / (1 - y)
        # the derivative of theta w.r.t mu
        tht_mu <- - 1 / mu^2
        # the derivative of mu w.r.t. eta
        mu_eta <- mu - mu * mu
        (comp1 + comp3) * tht_mu * mu_eta
    }
    simulate <- function (n, mu, phis, eta_zi) {
        NA
    }
    structure(list(family = "unit Lindley", link = stats$name, 
                   linkfun = stats$linkfun, linkinv = stats$linkinv, 
                   log_dens = log_dens, score_eta_fun = score_eta_fun,
                   simulate = simulate,
                   variance = function (mu) mu * (1 - mu)),
              class = "family")
}

# compoisson2
# NOTE: This function appears to be a duplicate of compoisson(). Both implement the same
# Conway-Maxwell-Poisson distribution with identical log_dens, score_eta_fun, and
# score_phis_fun. The intended distinction between compoisson() and compoisson2() is
# unclear - it may have been an alternative parameterization (e.g., via lambda vs. mu)
# that was never completed. Currently both return a family with family = "Conway Maxwell
# Poisson" and identical behavior.
#
# Arguments:
#   max: truncation point for the normalizing constant Z (default: 100)
#
# Returns:
#   A list of class "family" - identical to the output of compoisson().
compoisson2 <- function (max = 100) {
    stats <- make.link("log")
    .max <- max
    env <- new.env(parent = .GlobalEnv)
    assign(".max", max, envir = env)
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        # the log density function
        phis <- exp(phis)
        mu <- mu_fun(eta)
        Z <- function (lambda, nu, sumTo) {
            out <- lambda
            j <- seq(1, sumTo)
            log_lambda <- log(lambda)
            nu_log_factorial <- nu * cumsum(log(j))
            for (i in seq_along(out)) {
                out[i] <- 1 + sum(exp(j * log_lambda[i] - nu_log_factorial))
            }
            out
        }
        out <- y * log(mu) - phis * lgamma(y + 1) - log(Z(mu, phis, .max))
        attr(out, "mu_y") <- mu
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        phi <- exp(phis)
        Y <- function (lambda, nu, sumTo) {
            out <- lambda
            j <- seq(1, sumTo)
            log_lambda <- log(lambda)
            log_j <- log(j)
            nu_log_factorial <- nu * cumsum(log_j)
            for (i in seq_along(out)) {
                F1 <- j * log_lambda[i] - nu_log_factorial
                num <- sum(exp(log_j + F1))
                den <- 1 + sum(exp(F1))
                out[i] <- num / den
            }
            out
        }
        y - Y(mu, phi, .max)
    }
    score_phis_fun <- function (y, mu, phis, eta_zi) {
        phi <- exp(phis)
        W <- function (lambda, nu, sumTo) {
            out <- lambda
            j <- seq(1, sumTo)
            log_lambda <- log(lambda)
            log_factorial <- cumsum(log(j))
            log_log_factorial <- log(log_factorial)
            nu_log_factorial <- nu * log_factorial
            for (i in seq_along(out)) {
                num <- sum(exp(j * log_lambda[i] + log_log_factorial - nu_log_factorial))
                den <- 1 + sum(exp(j * log_lambda[i] - nu_log_factorial))
                out[i] <- num / den
            }
            out
        }
        (- lgamma(y + 1) + W(mu, phi, .max)) * phi
    }
    simulate <- function (n, mu, phis, eta_zi) {
        phi <- exp(phis)
    }
    structure(list(family = "Conway Maxwell Poisson", link = stats$name, 
                   linkfun = stats$linkfun, linkinv = stats$linkinv, log_dens = log_dens,
                   score_eta_fun = score_eta_fun, score_phis_fun = score_phis_fun),
              class = "family")
}

# find_lambda
# For the Conway-Maxwell-Poisson distribution, finds the rate parameter lambda such that
# the mean E[Y] = mu given the dispersion parameter nu. This is needed to simulate from
# the CMP distribution, since the CMP is typically parameterized by (lambda, nu) but
# the model is fit via the mean parameterization (mu, nu).
#
# The equation E[Y] = sum_{j>=1} j * lambda^j / (j!)^nu / Z(lambda,nu) = mu is solved
# numerically using uniroot(). Initial values are based on the approximation
# lambda ~ (mu + (nu-1)/(2*nu))^nu (from Shmueli et al., 2005).
#
# Arguments:
#   mu:    target mean vector (positive values)
#   nu:    dispersion parameter (positive; nu=1 -> Poisson, nu>1 -> under-dispersion)
#   sumTo: truncation point for the infinite series (default: 100)
#
# Returns:
#   A numeric vector of lambda values (same length as mu) such that E[Y | lambda, nu] = mu.
find_lambda <- function (mu, nu, sumTo = 100) {
    j <- seq(1, sumTo)
    nu_log_factorial <- nu * cumsum(log(j))
    f <- function (lambda, mu) {
        fact <- exp(j * log(lambda) - nu_log_factorial)
        sum(c(-mu, (j - mu) * fact))
    }
    out <- mu
    init_lambda <- (mu + (nu - 1) / (2 * nu))^nu
    for (i in seq_along(mu)) {
        int <- c(max(1e-06, init_lambda[i] - 10), min(sumTo, init_lambda[i] + 10))
        test <- try(uniroot(f, interval = int, mu = mu[i])$root, silent = TRUE)
        if (inherits(test, "try-error")) {
            test <- try(uniroot(f, interval = c(1e-06, sumTo), mu = mu[i])$root, 
                        silent = TRUE)
        }
        if (inherits(test, "try-error")) {
            stop("it was not possible to find lambda parameter of the ", 
                 "Conway Maxwell Poisson distribution;\nre-fit the model using ",
                 "\n\n\tmixed_model(..., family = compoisson(max = XXX))\n\n",
                 "where 'XXX' is a big enough count.")
        }
        out[i] <- test
    }
    out
}

# beta.binomial
# Creates a family object for the beta-binomial distribution, which models over-dispersed
# binomial data by mixing a beta prior on the success probability with the binomial likelihood.
# The marginal distribution for Y | size is:
#   P(Y=y) = C(size,y) * B(y + phi*mu, size - y + phi*(1-mu)) / B(phi*mu, phi*(1-mu))
# where phi = exp(phis) is the precision parameter and B is the beta function.
# Supports both logit and cloglog link functions. Provides analytic score functions
# and simulate() via rbeta/rbinom.
#
# Arguments:
#   link: link function name; either "logit" (default) or "cloglog"
#
# Returns:
#   A list of class "family" with log_dens, score_eta_fun, score_phis_fun, simulate.
beta.binomial <- function (link = "logit") {
    .link <- link
    env <- new.env(parent = .GlobalEnv)
    assign(".link", link, envir = env)
    stats <- make.link(link)
    dbbinom <- function (x, size, prob, phi, log = FALSE) {
        A <- phi * prob
        B <- phi * (1 - prob)
        log_numerator <- lbeta(x + A, size - x + B)
        log_denominator <- lbeta(A, B)
        fact <- lchoose(size, x)
        if (log) {
            fact + log_numerator - log_denominator
        } else {
            exp(fact + log_numerator - log_denominator)
        }
    }
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        phi <- exp(phis)
        eta <- as.matrix(eta)
        mu_y <- mu_fun(eta)
        out <- if (NCOL(y) == 2L) {
            dbbinom(y[, 1L], y[, 1L] + y[, 2L], mu_y, phi, TRUE)
        } else {
            dbbinom(y, rep(1L, length(y)), mu_y, phi, TRUE)
        }
        attr(out, "mu_y") <- mu_y
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        phi <- exp(phis)
        mu <- as.matrix(mu)
        if (NCOL(y) == 2L) {
            size <- y[, 1L] + y[, 2L]
            y <- y[, 1L]
        } else {
            size <- rep(1L, length(y))
        }
        phi_mu <- phi * mu
        phi_1mu <- phi * (1 - mu)
        comp1 <- (digamma(y + phi_mu) - digamma(size - y + phi_1mu)) * phi
        comp2 <- (digamma(phi_mu) - digamma(phi_1mu)) * phi
        mu.eta <- switch(.link,
                         "logit" = mu - mu * mu,
                         "cloglog" = - (1 - mu) * log(1 - mu))
        out <- (comp1 - comp2) * mu.eta
        out
    }
    score_phis_fun <- function (y, mu, phis, eta_zi) {
        phi <- exp(phis)
        mu <- as.matrix(mu)
        if (NCOL(y) == 2L) {
            size <- y[, 1L] + y[, 2L]
            y <- y[, 1L]
        } else {
            size <- rep(1L, length(y))
        }
        mu1 <- 1 - mu
        phi_mu <- phi * mu
        phi_1mu <- phi * mu1
        comp1 <- digamma(y + phi_mu) * mu + digamma(size - y + phi_1mu) * mu1 - 
            digamma(size + phi)
        comp2 <- digamma(phi_mu) * mu + digamma(phi_1mu) * mu1 - digamma(phi)
        out <- (comp1 - comp2) * phi
        out
    }
    simulate <- function (n, mu, phis, eta_zi) {
        phi <- exp(phis)
        probs <- rbeta(n, shape1 = mu * phi, shape2 = phi * (1 - mu))
        rbinom(n, size = 1, prob = probs)
    }
    structure(list(family = "beta binomial", link = stats$name, 
                   linkfun = stats$linkfun, linkinv = stats$linkinv, log_dens = log_dens,
                   score_eta_fun = score_eta_fun,
                   score_phis_fun = score_phis_fun, simulate = simulate),
              class = "family")
}

# Gamma.fam
# Creates a family object for the Gamma distribution with log link, using the
# shape-mean parameterization: Y ~ Gamma(shape=phi, scale=mu/phi) where phi = exp(phis).
# The mean is E[Y] = mu = exp(eta) and variance = mu^2/phi.
# This is the preferred Gamma family for GLMMadaptive (not the base R Gamma() family,
# which lacks log_dens). Provides analytic score_eta_fun, score_phis_fun, and simulate().
# Also used as a fallback when mixed_model() receives family = Gamma() with log link.
#
# Returns:
#   A list of class "family" with log_dens, score_eta_fun, score_phis_fun, simulate.
Gamma.fam <- function () {
    stats <- make.link("log")
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        phi <- exp(phis)
        eta <- as.matrix(eta)
        mu_y <- mu_fun(eta)
        out <- dgamma(y, shape = phi, scale = mu_y / phi, log = TRUE)
        attr(out, "mu_y") <- mu_y
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        phi <- exp(phis)
        mu <- as.matrix(mu)
        comp <- phi / mu
        mu.eta <- mu
        out <- comp * (y / mu - 1) * mu.eta
        out
    }
    score_phis_fun <- function (y, mu, phis, eta_zi) {
        phi <- exp(phis)
        mu <- as.matrix(mu)
        comp1 <- log(y) - log(mu) - y / mu
        comp2 <- log(phi) + 1 - digamma(phi)
        out <- (comp1 + comp2) * phi
        out
    }
    simulate <- function (n, mu, phis, eta_zi) {
        phi <- exp(phis)
        rgamma(n, shape = phi, scale = mu / phi)
    }
    structure(list(family = "Gamma", link = stats$name, 
                   linkfun = stats$linkfun, linkinv = stats$linkinv, 
                   log_dens = log_dens, score_eta_fun = score_eta_fun,
                   score_phis_fun = score_phis_fun, simulate = simulate),
              class = "family")
}

# censored.normal
# Creates a family object for censored normally-distributed data with identity link.
# Handles three types of observations, distinguished by a 2-column response matrix
# y = cbind(value, indicator) where indicator is:
#   0: observed (non-censored): contribution = dnorm(y, eta, sigma)
#   1: left-censored: contribution = pnorm(y, eta, sigma) [P(Y <= threshold)]
#   2: right-censored: contribution = 1 - pnorm(y, eta, sigma) [P(Y > threshold)]
# sigma = exp(phis) is the residual standard deviation.
# Provides analytic score_eta_fun and score_phis_fun with careful handling of
# boundary cases (probabilities clamped to avoid log(0)). Also provides simulate()
# which returns uncensored normal draws (censoring is the data structure, not simulated).
#
# Returns:
#   A list of class "family" with log_dens, score_eta_fun, score_phis_fun, simulate.
censored.normal <- function () {
    stats <- make.link("identity")
    log_dens <- function (y, eta, mu_fun, phis, eta_zi) {
        sigma <- exp(phis)
        # indicators for non-censored, left and right censored observations
        ind0 <- y[, 2L] == 0 # non-censored
        ind1 <- y[, 2L] == 1 # left-censored
        ind2 <- y[, 2L] == 2 # right-censored
        eta <- as.matrix(eta)
        out <- eta
        out[ind0, ] <- dnorm(y[ind0, 1L], eta[ind0, ], sigma, log = TRUE)
        out[ind1, ] <- pnorm(y[ind1, 1L], eta[ind1, ], sigma, log.p = TRUE)
        out[ind2, ] <- pnorm(y[ind2, 1L], eta[ind2, ], sigma, log.p = TRUE,
                             lower.tail = FALSE)
        attr(out, "mu_y") <- eta
        out
    }
    score_eta_fun <- function (y, mu, phis, eta_zi) {
        sigma <- exp(phis)
        # indicators for non-censored, left and right censored observations
        ind0 <- y[, 2L] == 0 # non-censored
        ind1 <- y[, 2L] == 1 # left-censored
        ind2 <- y[, 2L] == 2 # right-censored
        eta <- as.matrix(mu)
        out <- eta
        if (any(ind0)) out[ind0, ] <- (y[ind0, 1L] - eta[ind0, ]) / sigma^2
        if (any(ind1)) {
            A <- pnorm(y[ind1, 1L], eta[ind1, ], sigma)
            tt <- (y[ind1, 1L] - eta[ind1, ]) / sigma
            out[ind1, ] <- - exp(- 0.5 * tt^2) / (sqrt(2 * pi) * sigma * A)
        }
        if (any(ind2)) {
            P <- pnorm(y[ind2, 1L], eta[ind2, ], sigma)
            tt <- (y[ind2, 1L] - eta[ind2, ]) / sigma
            A <-  eta[ind2, ] * P - sigma * exp(- 0.5 * tt^2) / sqrt(2 * pi) 
            B <- pnorm(y[ind2, 1L], eta[ind2, ], sigma, lower.tail = FALSE)
            B <- pmax(B, sqrt(.Machine$double.eps))
            out[ind2, ] <- (-A / B + eta[ind2, ] * (1 - B) / B) / sigma^2
        }
        out
    }
    score_phis_fun <- function (y, mu, phis, eta_zi) {
        sigma <- exp(phis)
        # indicators for non-censored, left and right censored observations
        ind0 <- y[, 2L] == 0 # non-censored
        ind1 <- y[, 2L] == 1 # left-censored
        ind2 <- y[, 2L] == 2 # right-censored
        eta <- as.matrix(mu)
        out <- eta
        if (any(ind0)) out[ind0, ] <- - 1 + (y[ind0, 1L] - eta[ind0, ])^2 / sigma^2
        if (any(ind1)) {
            tt <- (y[ind1, 1L] - eta[ind1, ]) / sigma
            A <- (-tt * exp(- 0.5 * tt^2)) / sqrt(2 * pi) 
            B <- pnorm(y[ind1, 1L], eta[ind1, ], sigma)
            B <- pmax(B, sqrt(.Machine$double.eps))
            out[ind1, ] <- A / B
        }
        if (any(ind2)) {
            P <- pnorm(y[ind2, 1L], eta[ind2, ], sigma)
            tt <- (y[ind2, 1L] - eta[ind2, ]) / sigma
            A <- sigma^2 * P + sigma^2 * (-tt * exp(- 0.5 * tt^2)) / sqrt(2 * pi) 
            B <- pnorm(y[ind2, 1L], eta[ind2, ], sigma, lower.tail = FALSE)
            B <- pmax(B, sqrt(.Machine$double.eps))
            out[ind2, ] <- (1 - B) / B - A / (B * sigma^2)
        }
        out
    }
    simulate <- function (n, mu, phis, eta_zi) {
        rnorm(n = n, mean = mu, sd = exp(phis))
    }
    structure(list(family = "censored normal", link = stats$name, 
                   linkfun = stats$linkfun, linkinv = stats$linkinv, log_dens = log_dens,
                   score_eta_fun = score_eta_fun, score_phis_fun = score_phis_fun, 
                   simulate = simulate),
              class = "family")
}
