# print.MixMod
# Prints a brief summary of a fitted MixMod object. Displays the model call, family,
# link function, random-effects covariance matrix (as standard deviations and correlations
# if non-diagonal), fixed-effects coefficient estimates, zero-part coefficients (if present),
# dispersion parameters (if present), and the log-likelihood.
#
# Arguments:
#   x:      a fitted object of class "MixMod"
#   digits: number of significant digits for display (default: max(4, getOption("digits")-4))
#   ...:    currently ignored
#
# Returns: x invisibly (for method chaining).
print.MixMod <- function (x, digits = max(4, getOption("digits") - 4), ...) {
    cat("\nCall:\n", printCall(x$call), "\n\n", sep = "")
    cat("\nModel:")
    user_defined <- is.null(x$family)
    cat("\n family:", if (user_defined) "user-defined" else x$family$family)
    cat("\n link:", if (user_defined) "user-defined" else x$family$link, "\n")
    cat("\nRandom effects covariance matrix:\n")
    D <- x$D
    ncz <- nrow(D)
    diag.D <- all(abs(D[lower.tri(D)]) < sqrt(.Machine$double.eps))
    sds <- sqrt(diag(D))
    if (ncz > 1) {
        if (diag.D) {
            dat <- data.frame("StdDev" = round(sds, digits), row.names = rownames(D))
        } else {
            corrs <- cov2cor(D)
            corrs[upper.tri(corrs, TRUE)] <- 0
            mat <- round(cbind(sds, corrs[, -ncz]), digits)
            mat <- apply(mat, 2, sprintf, fmt = "% .4f")
            mat[mat == mat[1, 2]] <- ""
            mat[1, -1] <- abbreviate(colnames(mat)[-1], 6)
            colnames(mat) <- c(colnames(mat)[1], rep("", ncz - 1))
            dat <- data.frame(mat, check.rows = FALSE, check.names = FALSE)
            names(dat) <- c("StdDev", "Corr", if (ncz > 2) rep(" ", ncz - 2) else NULL)
            row.names(dat) <- dimnames(D)[[1]]
        }
    } else {
        dat <- data.frame("StdDev" = sds, row.names = rownames(D),
                          check.rows = FALSE, check.names = FALSE)
    }
    print(dat)
    cat("\nFixed effects:\n")
    print(x$coefficients)
    if (!is.null(x$gammas)) {
        cat("\nZero-part coefficients:\n")
        print(x$gammas)
    }
    if (!is.null(x$phis)) {
        if (x$family$family %in% c("negative binomial", "zero-inflated negative binomial")) {
            cat("\ndispersion parameter:\n", exp(x$phis), "\n")
        } else if (x$family$family %in% c("hurdle log-normal", "censored normal")) {
            cat("\nResidual std. dev.:\n", exp(x$phis), "\n")
        } else {
            cat("\nphi parameters:\n", x$phis, "\n")
        }
    }
    cat("\nlog-Lik:", x$logLik)
    cat("\n\n")
    invisible(x)
}

# vcov.MixMod
# Computes the variance-covariance matrix of the parameter estimates. By default, returns
# the inverse of the Hessian (observed information matrix). If sandwich = TRUE, returns
# the sandwich (robust) estimator V = H^{-1} * (sum_i s_i s_i') * H^{-1}, which is
# consistent under model misspecification.
#
# The parm argument controls which subset of parameters to return:
#   "all":          full p x p variance-covariance matrix
#   "fixed-effects": variance-covariance for betas only
#   "var-cov":       variance-covariance for the covariance matrix parameters (D elements)
#   "extra":         variance-covariance for phis (dispersion parameters)
#   "zero_part":     variance-covariance for gammas (zero-part fixed effects)
#
# Arguments:
#   object:   a fitted MixMod object
#   parm:     which parameters to extract (see above)
#   sandwich: logical; if TRUE returns the sandwich variance estimator (default: FALSE)
#   ...:      currently ignored
#
# Returns:
#   A numeric matrix of the requested variance-covariance submatrix.
vcov.MixMod <- function (object, parm = c("all", "fixed-effects", "var-cov","extra",
                                          "zero_part"), sandwich = FALSE, ...) {
    parm <- match.arg(parm)
    V <- solve(object$Hessian)
    if (sandwich) {
        meat <- object$score_vect_contributions
        ind <- !names(meat) %in% "score.D" & !sapply(meat, is.null)
        meat[ind] <- lapply(meat[ind], rowsum, group = object$id[[1]], reorder = FALSE)
        meat <- do.call('cbind', meat)
        meat <- Reduce("+", lapply(split(meat, row(meat)), function (x) x %o% x))
        V <- V %*% meat %*% V
    }
    if (parm == "all") {
        return(V)
    }
    if (parm == "fixed-effects") {
        n_betas <- length(object$coefficients)
        return(V[seq_len(n_betas), seq_len(n_betas), drop = FALSE])
    }
    if (parm == "var-cov") {
        D <- object$D
        diag_D <- ncol(D) > 1 && all(abs(D[lower.tri(D)]) < sqrt(.Machine$double.eps))
        include <- if (diag_D) {
            unconstr_D <- log(diag(D))
            n_betas <- length(object$coefficients)
            seq(n_betas + 1, n_betas + length(unconstr_D))
        } else {
            unconstr_D <- chol_transf(D)
            n_betas <- length(object$coefficients)
            seq(n_betas + 1, n_betas + length(unconstr_D))
        }
        return(V[include, include, drop = FALSE])
    }
    if (parm == "extra") {
        if (is.null(object$phis)) {
            stop("the model behind 'object' contains no extra (phis) parameters.\n")
        } else {
            ind_phis <- grep("phi_", colnames(V), fixed = TRUE)
            return(V[ind_phis, ind_phis, drop = FALSE])
        }
    }
    if (parm == "zero_part") {
        if (is.null(object$gammas)) {
            stop("the fitted model does not have an extra zero part.")
        } else {
            gammas <- object$gammas
            ind_gammas <- grep("zi_", colnames(V), fixed = TRUE)
            return(V[ind_gammas, ind_gammas, drop = FALSE])
        }
    }
}

# logLik.MixMod
# Extracts the marginal log-likelihood from a fitted MixMod object. The returned
# object is of class "logLik" with attributes:
#   df:   the number of parameters (= nrow(Hessian))
#   nobs: the number of groups/clusters (not observations)
# This allows AIC() and BIC() to work directly on MixMod objects.
#
# Arguments:
#   object: a fitted MixMod object
#   ...:    currently ignored
#
# Returns:
#   A numeric scalar of class "logLik" with df and nobs attributes.
logLik.MixMod <- function (object, ...) {
    out <- object$logLik
    attr(out, "df") <- nrow(object$Hessian)
    attr(out, "nobs") <- length(unique(object$id[[1]]))
    class(out) <- "logLik"
    out
}

# coef.MixMod
# Returns the cluster-specific (subject-specific) coefficients, computed as the sum of
# the fixed effects and the empirical Bayes estimates of the random effects:
#   beta_i = beta + b_i
# Returns an n_groups x p matrix where n_groups is the number of clusters and p is the
# number of fixed effects. For sub_model = "zero_part", returns the zero-part coefficients
# (or gammas alone if no ZI random effects).
#
# Arguments:
#   object:    a fitted MixMod object
#   sub_model: "main" (default) for the main model coefficients; "zero_part" for ZI part
#   ...:       currently ignored
#
# Returns:
#   A matrix (n_groups x p) of subject-specific coefficients, or gammas vector if
#   sub_model = "zero_part" and there are no zero-part random effects.
coef.MixMod <- function (object, sub_model = c("main", "zero_part"), ...) {
    sub_model <- match.arg(sub_model)
    b <- ranef(object)
    RE_zi <- grep("zi_", colnames(b), fixed = TRUE)
    if (sub_model == "main") {
        betas <- fixef(object, sub_model = "main")
        if (length(RE_zi)) 
            b <- b[, -RE_zi, drop = FALSE]
        out <- matrix(betas, nrow = nrow(b), ncol = length(betas), byrow = TRUE)
        colnames(out) <- names(betas)
        rownames(out) <- rownames(b)
        out[, colnames(b)] <- out[, colnames(b)] + b
        out
    } else {
        gammas <- fixef(object, sub_model = "zero_part")
        if (length(RE_zi)) {
            b <- b[, RE_zi, drop = FALSE]
            colnames(b) <- gsub("zi_", "", colnames(b), fixed = TRUE)
            out <- matrix(gammas, nrow = nrow(b), ncol = length(gammas), byrow = TRUE)
            colnames(out) <- names(gammas)
            rownames(out) <- rownames(b)
            out[, colnames(b)] <- out[, colnames(b)] + b
            out
        } else {
            gammas
        }
        
    }
}

# fixef.MixMod
# Extracts the fixed-effects (population-level) coefficients from a fitted MixMod object.
#
# Arguments:
#   object:    a fitted MixMod object
#   sub_model: "main" (default) returns the main fixed effects (betas);
#              "zero_part" returns the zero-part fixed effects (gammas)
#   ...:       currently ignored
#
# Returns:
#   A named numeric vector of fixed-effects coefficients.
fixef.MixMod <- function(object, sub_model = c("main", "zero_part"), ...) {
    sub_model <- match.arg(sub_model)
    if (sub_model == "main") {
        object$coefficients
    } else {
        if (!is.null(object$gammas)) 
            object$gammas
        else
            stop("the fitted model does not have an extra zero-part.")
    }
}

# ranef.MixMod
# Extracts the empirical Bayes (posterior mode) estimates of the random effects from a
# fitted MixMod object. These are the modes of p(b_i | y_i, theta_hat).
#
# Arguments:
#   object:    a fitted MixMod object
#   post_vars: logical; if TRUE, attaches the posterior variance matrices as an attribute
#              "post_vars" (list of nRE x nRE matrices, one per group) (default: FALSE)
#   ...:       currently ignored
#
# Returns:
#   An n_groups x nRE matrix of random-effect posterior mode estimates. If post_vars = TRUE,
#   the matrix has a "post_vars" attribute containing the list of posterior variance matrices.
ranef.MixMod <- function(object, post_vars = FALSE, ...) {
    out <- object$post_modes
    if (post_vars)
        attr(out, "post_vars") <- object$post_vars
    out
}

# summary.MixMod
# Computes a comprehensive summary for a fitted MixMod object, including:
#   - Fixed-effects coefficient table (Estimate, Std.Err, z-value, p-value)
#   - Zero-part coefficient table (if ZI model)
#   - Dispersion/shape parameter table (if model has phis)
#   - Random-effects covariance matrix D
#   - Log-likelihood, AIC, BIC
#   - Number of observations and convergence status
#
# The standard errors are derived from the variance-covariance matrix (optionally using
# the sandwich estimator). Two-sided Wald z-tests are used for fixed effects.
#
# Arguments:
#   object:   a fitted MixMod object
#   sandwich: logical; if TRUE uses the sandwich (robust) variance estimator (default: FALSE)
#   ...:      currently ignored
#
# Returns:
#   An object of class "summary.MixMod" (a list). Use print.summary.MixMod() to display.
summary.MixMod <- function (object, sandwich = FALSE, ...) {
    betas <- fixef(object)
    n_betas <- length(betas)
    V <- vcov(object, sandwich = sandwich)
    var_betas <- V[seq_len(n_betas), seq_len(n_betas), drop = FALSE]
    ses <- sqrt(diag(var_betas))
    D <- object$D
    n_D <- length(D[lower.tri(D, TRUE)])
    coef_table <- cbind("Estimate" = betas, "Std.Err" = ses, "z-value" = betas / ses,
                        "p-value" = 2 * pnorm(abs(betas / ses), lower.tail = FALSE))
    if (!is.null(object$gammas)) {
        gammas <- object$gammas
        ind_gammas <- grep("zi_", colnames(V), fixed = TRUE)
        ses <- sqrt(diag(V[ind_gammas, ind_gammas, drop = FALSE]))
        coef_table_zi <- cbind("Estimate" = gammas, "Std.Err" = ses, "z-value" = gammas / ses,
                               "p-value" = 2 * pnorm(abs(gammas / ses), lower.tail = FALSE))
    }
    out <- list(coef_table = coef_table, 
                coef_table_zi = if (!is.null(object$gammas)) coef_table_zi,D = D, 
                logLik = logLik(object),
                AIC = AIC(object), BIC = BIC(object), call = object$call,
                N = length(object$id[[1]]))
    if (!is.null(object$phis)) {
        phis <- object$phis
        ind_phis <- grep("phi_", colnames(V), fixed = TRUE)
        var_phis <- V[ind_phis, ind_phis, drop = FALSE]
        out$phis_table <- cbind("Estimate" = phis, "Std.Err" = sqrt(diag(var_phis)))
    }
    out$control <- object$control
    out$family <- object$family
    out$converged <- object$converged
    class(out) <- 'summary.MixMod'
    out
}

# print.summary.MixMod
# Prints the full summary of a MixMod model (from summary.MixMod). Displays:
#   - Model call, family, link function
#   - Number of observations and groups
#   - AIC, BIC, log-likelihood
#   - Random-effects covariance matrix (standard deviations and correlations)
#   - Fixed-effects coefficient table with z-statistics and p-values
#   - Zero-part coefficients (if ZI model)
#   - Dispersion/phi parameters (if applicable)
#   - Integration method and number of quadrature points
#   - Optimization method and convergence status
#
# Arguments:
#   x:      an object of class "summary.MixMod" (from summary.MixMod())
#   digits: number of significant digits (default: max(4, getOption("digits")-4))
#   ...:    currently ignored
#
# Returns: x invisibly.
print.summary.MixMod <- function (x, digits = max(4, getOption("digits") - 4), ...) {
    cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"),
        "\n\n", sep = "")
    cat("Data Descriptives:")
    cat("\nNumber of Observations:", x$N)
    cat("\nNumber of Groups:", attr(x$logLik, 'n'), "\n")
    cat("\nModel:")
    user_defined <- is.null(x$family)
    cat("\n family:", if (user_defined) "user-defined" else x$family$family)
    cat("\n link:", if (user_defined) "user-defined" else x$family$link, "\n")
    cat("\nFit statistics:\n")
    model.sum <- data.frame(log.Lik = x$logLik, AIC = x$AIC, BIC = x$BIC, row.names = "")
    print(model.sum)
    cat("\nRandom effects covariance matrix:\n")
    D <- x$D
    ncz <- nrow(D)
    diag.D <- all(abs(D[lower.tri(D)]) < sqrt(.Machine$double.eps))
    sds <- sqrt(diag(D))
    if (ncz > 1) {
        if (diag.D) {
            dat <- data.frame("StdDev" = round(sds, digits), row.names = rownames(D))
        } else {
            corrs <- cov2cor(D)
            corrs[upper.tri(corrs, TRUE)] <- 0
            mat <- round(cbind(sds, corrs[, -ncz]), digits)
            mat <- apply(mat, 2, sprintf, fmt = "% .4f")
            mat[mat == mat[1, 2]] <- ""
            mat[1, -1] <- abbreviate(colnames(mat)[-1], 6)
            colnames(mat) <- c(colnames(mat)[1], rep("", ncz - 1))
            dat <- data.frame(mat, check.rows = FALSE, check.names = FALSE)
            names(dat) <- c("StdDev", "Corr", if (ncz > 2) rep(" ", ncz - 2) else NULL)
            row.names(dat) <- dimnames(D)[[1]]
        }
    } else {
        dat <- data.frame("StdDev" = sds, row.names = rownames(D),
                          check.rows = FALSE, check.names = FALSE)
    }
    print(dat)
    cat("\nFixed effects:\n")
    coef_table <- as.data.frame(x[["coef_table"]])
    coef_table[1:3] <- lapply(coef_table[1:3], round, digits = digits)
    coef_table[["p-value"]] <- format.pval(coef_table[["p-value"]], eps = 1e-04)
    print(coef_table)
    if (!is.null(x[["coef_table_zi"]])) {
        cat("\nZero-part coefficients:\n")
        coef_table <- as.data.frame(x[["coef_table_zi"]])
        coef_table[1:3] <- lapply(coef_table[1:3], round, digits = digits)
        coef_table[["p-value"]] <- format.pval(coef_table[["p-value"]], eps = 1e-04)
        print(coef_table)
    }
    if (!is.null(x$phis_table)) {
        if (NB <- x$family$family %in% c("negative binomial", "zero-inflated negative binomial",
                                         "hurdle negative binomial")) {
            cat("\nlog(dispersion) parameter:\n")
        } else if (NB <- x$family$family %in% c("hurdle log-normal", "censored normal")) {
            cat("\nlog(residual std. dev.):\n")
        } else {
            cat("\nphi parameters:\n")
        }
        phis_table <- as.data.frame(x$phis_table)
        if (NB) 
            row.names(phis_table) <- " "
        phis_table[] <- lapply(phis_table, round, digits = digits)
        print(phis_table)
    }
    cat("\nIntegration:")
    cat("\nmethod: adaptive Gauss-Hermite quadrature rule")
    cat("\nquadrature points:", x$control$nAGQ)
    cat("\n\nOptimization:")
    methd <- if (x$control$iter_EM == 0) "quasi-Newton" 
    else if (isTRUE(attr(x$converged, "during_EM"))) "EM" 
    else "hybrid EM and quasi-Newton"
    cat("\nmethod:", methd)
    cat("\nconverged:", as.logical(x$converged), "\n")
    invisible(x)
}

# coef.summary.MixMod
# Extracts the fixed-effects coefficient table from a summary.MixMod object.
# Returns the data.frame with columns: Estimate, Std.Err, z-value, p-value.
#
# Arguments:
#   object: an object of class "summary.MixMod" (from summary.MixMod())
#   ...:    currently ignored
coef.summary.MixMod <- function (object, ...) {
    object$coef_table
}

# confint.MixMod
# Computes Wald confidence intervals for parameters of a fitted MixMod object.
# The intervals are based on the normal approximation: estimate ± z_{alpha/2} * SE.
# For variance-covariance parameters (parm = "var-cov"), confidence intervals are
# computed on the unconstrained (log-Cholesky) scale and then back-transformed to
# ensure positivity of variance components.
# For the negative binomial dispersion parameter, confidence intervals for phis are
# exponentiated to give intervals for the size parameter.
#
# Arguments:
#   object:   a fitted MixMod object
#   parm:     which parameters to compute CIs for (see vcov.MixMod for options)
#   level:    confidence level (default: 0.95)
#   sandwich: logical; if TRUE uses sandwich SE (default: FALSE)
#   ...:      currently ignored
#
# Returns:
#   A matrix with three columns: lower CI, upper CI, and point estimate.
confint.MixMod <- function (object, parm = c("fixed-effects", "var-cov","extra",
                                             "zero_part"),
                            level = 0.95, sandwich = FALSE, ...) {
    parm <- match.arg(parm)
    V <- vcov(object, sandwich = sandwich)
    if (parm == "fixed-effects") {
        betas <- fixef(object)
        n_betas <- length(betas)
        ses_betas <- sqrt(diag(V[seq_len(n_betas), seq_len(n_betas), drop = FALSE]))
        out <- cbind(betas + qnorm((1 - level) / 2) * ses_betas, betas,
                     betas + qnorm((1 + level) / 2) * ses_betas)
    } else if (parm == "var-cov") {
        D <- object$D
        diag_D <- ncol(D) > 1 && all(abs(D[lower.tri(D)]) < sqrt(.Machine$double.eps))
        if (diag_D) {
            unconstr_D <- log(diag(D))
            n_betas <- length(object$coefficients)
            include <- seq(n_betas + 1, n_betas + length(unconstr_D))
            ses_unconstr_D <- sqrt(diag(V[include, include, drop = FALSE]))
            out <- cbind(unconstr_D + qnorm((1 - level) / 2) * ses_unconstr_D,
                         unconstr_D,
                         unconstr_D + qnorm((1 + level) / 2) * ses_unconstr_D)
            out <- exp(out)
            rownames(out) <- paste0("var.", rownames(D))
        } else {
            unconstr_D <- chol_transf(D)
            n_betas <- length(object$coefficients)
            include <- seq(n_betas + 1, n_betas + length(unconstr_D))
            ses_unconstr_D <- sqrt(diag(V[include, include, drop = FALSE]))
            out <- cbind(unconstr_D + qnorm((1 - level) / 2) * ses_unconstr_D,
                         unconstr_D,
                         unconstr_D + qnorm((1 + level) / 2) * ses_unconstr_D)
            ind <- lower.tri(D, TRUE)
            out[, 1] <- chol_transf(out[, 1])[ind]
            out[, 2] <- chol_transf(out[, 2])[ind]
            out[, 3] <- chol_transf(out[, 3])[ind]
            nams <- rownames(D)
            rownames(out) <- apply(which(ind, arr.ind = TRUE), 1, function (k) {
                if (k[1L] == k[2L]) paste0("var.", nams[k[1]]) else {
                    paste0("cov.", abbreviate(nams[k[2]], 5), "_", abbreviate(nams[k[1]], 5)) 
                }
            })
        }
    } else if (parm == "extra") {
        if (is.null(object$phis)) {
            stop("the model behind 'object' contains no extra (phis) parameters.\n")
        } else {
            phis <- object$phis
            ind_phis <- grep("phi_", colnames(V), fixed = TRUE)
            ses_phis <- sqrt(diag(V[ind_phis, ind_phis, drop = FALSE]))
            out <- cbind(phis + qnorm((1 - level) / 2) * ses_phis, phis,
                         phis + qnorm((1 + level) / 2) * ses_phis)
            if (object$family$family %in% c("negative binomial", 
                                            "zero-inflated negative binomial"))
                out <- exp(out)
        }
    } else {
        if (is.null(object$gammas)) {
            stop("the fitted model does not have an extra zero part.")
        } else {
            gammas <- object$gammas
            ind_gammas <- grep("zi_", colnames(V), fixed = TRUE)
            ses_gammas <- sqrt(diag(V[ind_gammas, ind_gammas, drop = FALSE]))
            out <- cbind(gammas + qnorm((1 - level) / 2) * ses_gammas, gammas,
                         gammas + qnorm((1 + level) / 2) * ses_gammas)
        }
    }
    colnames(out) <- c(paste(round(100 * c((1 - level) / 2, 
                                           (1 + level) / 2), 1), "%"), "Estimate")[c(1,3,2)]
    out
}

# anova.MixMod
# Performs hypothesis tests for a fitted MixMod object. Supports two modes:
#   1. Likelihood ratio test (LRT): when object2 is provided (a nested model), computes
#      LRT = -2*(logLik(object) - logLik(object2)) and tests against chi-squared with
#      df = |p2 - p1| degrees of freedom.
#   2. Wald test: when L (a contrast matrix) is provided instead, computes the Wald test
#      statistic W = (L*beta)' * (L*V_beta*L')^{-1} * (L*beta) ~ chi-squared(nrow(L)).
#      With sandwich = TRUE, uses the robust variance estimator.
#
# Arguments:
#   object:   a fitted MixMod object (smaller/null model for LRT, or the model for Wald)
#   object2:  a second fitted MixMod object (larger/alternative model for LRT)
#   test:     logical; if TRUE performs the test (default: TRUE)
#   L:        numeric matrix for a Wald test (ncol must equal number of fixed effects)
#   sandwich: logical; use sandwich SE for Wald test (default: FALSE)
#   ...:      currently ignored
#
# Returns:
#   An object of class "aov.MixMod" (list) with test statistics, p-values, AIC, BIC.
anova.MixMod <- function (object, object2, test = TRUE, L = NULL,
                          sandwich = FALSE, ...) {
    if (missing(object2) && is.null(L))
        stop("either argument 'object2' or argument 'L' needs to be specified.\n")
    if (!missing(object2)) {
        if (!object$converged)
            warning("it seems that '", deparse(substitute(object)),
                    "' has not converged.\n")
        if (!object2$converged)
            warning("it seems that '", deparse(substitute(object2)),
                    "' has not converged.")
        L0 <- logLik(object)
        L1 <- logLik(object2)
        nb0 <- attr(L0, "df")
        nb1 <- attr(L1, "df")
        df <- abs(nb1 - nb0)
        if (test && df == 0) {
            test <- FALSE
            warning("the two objects represent models with the same number of parameters;",
                    " argument 'test' is set to FALSE.")
        }
        fam <- object$family$family
        fam2 <- object2$family$family
        if (test &&  (fam != fam2 && ((fam != "poisson" & fam2 != "negative binomial") &&
                                      (fam != "zero-inflated poisson" & fam2 != "zero-inflated negative binomial")))) {
            warning("it seems that the two objects represent model with different families;",
                    " are the models nested? If not, you should set 'test' to FALSE.")
        }
        out <- list(nam0 = deparse(substitute(object)), L0 = L0,
                    aic0 = AIC(object), bic0 = BIC(object),
                    nam1 = deparse(substitute(object2)), L1 = L1, aic1 = AIC(object2),
                    bic1 = BIC(object2), df = df, test = test)
        if (test) {
            LRT <- abs(- 2 * (L0 - L1))
            attributes(LRT) <- NULL
            out$LRT <- LRT
            out$p.value <- pchisq(LRT, df, lower.tail = FALSE)
        }
    } else {
        betas <- fixef(object)
        n_betas <- length(betas)
        if (!is.numeric(L) || ncol(L) != n_betas)
            stop("L matrix not of appropriate dimensions. ",
                 "It should have ", n_betas, " columns.\n")
        colnames(L) <- abbreviate(names(betas), 6)
        rownames(L) <- rep("", nrow(L))
        V <- vcov(object, sandwich = sandwich)
        var_betas <- V[seq_len(n_betas), seq_len(n_betas)]
        Lbetas <- c(L %*% betas)
        LVtL <- L %*% tcrossprod(var_betas, L)
        stat <- c(crossprod(Lbetas, solve(LVtL, Lbetas)))
        pval <- pchisq(stat, nrow(L), lower.tail = FALSE)
        res <- data.frame(Chisq = stat, df = nrow(L),
                          "Pr(>|Chi|)" = pval, check.names = FALSE, row.names = " ")
        out <- list(aovTab.L = res, L = L)
    }
    class(out) <- "aov.MixMod"
    out
}

# print.aov.MixMod
# Prints the result of anova.MixMod(). For LRT (two-model comparison), displays a
# table with AIC, BIC, log-likelihood, LRT statistic, df, and p-value. For Wald tests
# (L matrix provided), displays the contrast matrix and the chi-squared test results.
print.aov.MixMod <- function (x, ...) {
    if (is.null(x$L0)) {
        f <- function (dat) {
            dat[] <- lapply(dat, function (x)
                round(unlist(x), 4))
            dat$'Pr(>|Chi|)' <- format.pval(dat$'Pr(>|Chi|)', eps = 1e-04)
            dat
        }
        cat("\nMarginal Wald Tests Table\n")
        if (!is.null(x$aovTab.L)) {
            cat("\nUser-defined contrasts matrix:\n")
            print(x$L)
            cat("\n")
            print(f(x$aovTab.L))
        }
        cat("\n")
    } else {
        dat <- if (x$test) {
            p.val <- round(x$p.value, 4)
            p.val <- if (p.val < 0.0001) "<0.0001" else p.val
            data.frame(AIC = round(c(x$aic0, x$aic1), 2),
                       BIC = round(c(x$bic0, x$bic1), 2),
                       log.Lik = round(c(x$L0, x$L1), 2),
                       LRT = c(" ", round(x$LRT, 2)), df = c("", x$df),
                       p.value = c("", p.val), row.names = c(x$nam0, x$nam1))
        } else {
            data.frame(AIC = round(c(x$aic0, x$aic1), 2),
                       BIC = round(c(x$bic0, x$bic1), 2),
                       log.Lik = round(c(x$L0, x$L1), 2), df = c("", x$df),
                       row.names = c(x$nam0, x$nam1))
        }
        cat("\n")
        print(dat)
        cat("\n")
    }
    invisible(x)
}

# fitted.MixMod
# Computes fitted values (predicted means) for a fitted MixMod object. Supports three types:
#   "mean_subject":     uses fixed effects only (random effects = 0), giving the "average
#                       subject" prediction; appropriate for population-level inference.
#   "subject_specific": adds the empirical Bayes random-effect estimates to the fixed effects,
#                       giving predictions conditional on estimated random effects.
#   "marginal":         uses marginalized coefficients (via marginal_coefs()) that account for
#                       the random-effects distribution; appropriate for marginal inference.
# For zero-inflated models, the fitted values are scaled by (1 - P(structural zero)).
#
# Arguments:
#   object:   a fitted MixMod object
#   type:     prediction type (default: "mean_subject")
#   link_fun: a function to apply to the linear predictor before computing the mean
#             (used for marginal_coefs in non-standard parameterizations)
#   ...:      currently ignored
#
# Returns:
#   A named numeric vector of fitted means (on the response scale, after the inverse link).
fitted.MixMod <- function (object, type = c("mean_subject", "subject_specific", "marginal"),
                           link_fun = NULL, ...) {
    type <- match.arg(type)
    X <- model.matrix(object$Terms$termsX, object$model_frames$mfX)
    if (type == "mean_subject") {
        betas <- fixef(object)
        eta <- c(X %*% betas)
    } else if (type == "subject_specific") {
        betas <- fixef(object)
        b <- ranef(object)
        RE_zi <- grep("zi_", colnames(b), fixed = TRUE)
        if (length(RE_zi))
            b <- b[, -RE_zi, drop = FALSE]
        id <- match(object$id[[1L]], unique(object$id[[1L]]))
        Z <- mapply(constructor_Z, object$Terms$termsZ, object$model_frames$mfZ, 
                    MoreArgs = list(id = id), SIMPLIFY = FALSE)
        Z <- do.call("cbind", Z)
        eta <- c(X %*% betas) + rowSums(Z * b[id, , drop = FALSE])
    } else {
        betas <- marginal_coefs(object, link_fun = link_fun)$betas
        eta <- c(X %*% betas)
    }
    if (!is.null(object$offset))
        eta <- eta + object$offset
    mu <- object$Funs$mu_fun(eta)
    if (!is.null(object$gammas)) {
        X_zi <- model.matrix(object$Terms$termsX_zi, object$model_frames$mfX_zi)
        offset_zi <- model.offset(object$model_frames$mfX_zi)
        gammas <- fixef(object, "zero_part")
        eta_zi <- c(X_zi %*% gammas)
        if (type == "subject_specific" && !is.null(object$Terms$termsZ_zi)) {
            b <- ranef(object)
            RE_zi <- grep("zi_", colnames(b), fixed = TRUE)
            if (length(RE_zi))
                b <- b[, RE_zi, drop = FALSE]
            id <- match(object$id[[1]], unique(object$id[[1]]))
            Z_zi <- mapply(constructor_Z, object$Terms$termsZ_zi, object$model_frames$mfZ_zi, 
                           MoreArgs = list (id = id), SIMPLIFY = FALSE)
            Z_zi <- do.call("cbind", Z_zi)
            eta_zi <- eta_zi + rowSums(Z_zi * b[id, , drop = FALSE])
        }
        if (!is.null(offset_zi))
            eta_zi <- eta_zi + offset_zi
        mu <- plogis(eta_zi, lower.tail = FALSE) * mu
    }
    names(mu) <- rownames(X)
    mu
}

# residuals.MixMod
# Computes residuals as observed minus fitted values. Uses fitted.MixMod() internally
# so supports the same three prediction types. Factor responses (binomial) are converted
# to 0/1 before computing residuals. An optional transformation tasnf_y can be applied
# to the response before subtracting the fitted values.
#
# Arguments:
#   object:  a fitted MixMod object
#   type:    "mean_subject" (default), "subject_specific", or "marginal"
#   link_fun: optional link function transformation (passed to fitted())
#   tasnf_y: an optional transformation applied to y (default: identity function)
#   ...:     currently ignored
#
# Returns:
#   A numeric vector of residuals (transformed y minus fitted values).
residuals.MixMod <- function (object, type = c("mean_subject", "subject_specific",
                                               "marginal"), link_fun = NULL,
                              tasnf_y = function (x) x, ...) {
    type <- match.arg(type)
    fits <- fitted(object, type = type, link_fun = link_fun)
    y <- model.response(object$model_frames$mfX)
    if (is.factor(y)) {
        y <- as.numeric(y != levels(y)[1L])
    }
    tasnf_y(y) - fits
}

# marginal_coefs (generic)
# Generic function for computing marginalized (population-averaged) coefficients.
# Dispatches to marginal_coefs.MixMod() for MixMod objects.
marginal_coefs <- function (object, ...) UseMethod("marginal_coefs")

# marginal_coefs.MixMod
# Computes marginalized (population-averaged) coefficients for a fitted MixMod object,
# following the approach of Hedeker et al. (2017). These coefficients represent the
# marginal (not conditional) relationship between covariates and the response, averaging
# over the distribution of the random effects.
#
# The method works by:
#   1. Simulating M samples of random effects b from N(0, D).
#   2. Computing E_{b}[g^{-1}(X*beta + Z*b)] for each set of covariates.
#   3. Running a GLM with the marginalized probabilities as responses to extract
#      the marginal regression coefficients.
# This is repeated K times (with different random seeds), and the results are averaged.
# Parallel computation is supported via the parallel package.
#
# If std_errors = TRUE, the delta method is used to propagate uncertainty from the
# model parameters to the marginalized coefficients, using the score contributions
# stored in the fitted model.
#
# Arguments:
#   object:    a fitted MixMod object
#   std_errors: logical; compute standard errors via delta method (default: FALSE)
#   link_fun:  optional custom link function (default: uses the model's link)
#   M:         number of random-effect samples per iteration (default: 3000)
#   K:         number of iterations to average over (default: 100)
#   seed:      random seed for reproducibility (default: 1)
#   cores:     number of parallel cores (default: detected cores - 1)
#   sandwich:  logical; use sandwich SE for the delta method (default: FALSE)
#   ...:       currently ignored
#
# Returns:
#   An object of class "m_coefs" with components:
#     betas:     marginalized fixed-effects vector
#     coef_table: coefficient table (if std_errors = TRUE)
#     vcov:       variance-covariance matrix (if std_errors = TRUE)
marginal_coefs.MixMod <- function (object, std_errors = FALSE, link_fun = NULL,
                                   M = 3000L, K = 100L,
                                   seed = 1L,
                                   cores = max(parallel::detectCores() - 1, 1),
                                   sandwich = FALSE, ...) {
    offset <- object$offset
    X <- model.matrix(object$Terms$termsX, object$model_frames$mfX)
    id <- match(object$id[[1L]], unique(object$id[[1L]]))
    Z <- mapply(constructor_Z, object$Terms$termsZ, object$model_frames$mfZ, 
                MoreArgs = list(id = id), SIMPLIFY = FALSE)
    Z <- do.call("cbind", Z)
    betas <- fixef(object)
    D <- object$D
    if (!is.null(object$gammas)) {
        offset_zi <- model.offset(object$model_frames$mfX_zi)
        X_zi <- model.matrix(object$Terms$termsX_zi, object$model_frames$mfX_zi)
        if (!is.null(object$Terms$termsZ_zi)) {
            Z_zi <- mapply(constructor_Z, object$Terms$termsZ_zi, object$model_frames$mfZ_zi, 
                           MoreArgs = list (id = id), SIMPLIFY = FALSE)
            Z_zi <- do.call("cbind", Z_zi)
        } else {
            Z_zi <- NULL
        }
        gammas <- fixef(object, "zero_part")
    } else {
        X_zi <- Z_zi <- gammas <- NULL
    }
    compute_marg_coefs <- function (object, X, betas, Z, X_zi, gammas, Z_zi, D, M, 
                                    link_fun, seed) {
        if (!exists(".Random.seed", envir = .GlobalEnv)) 
            runif(1L)
        RNGstate <- get(".Random.seed", envir = .GlobalEnv)
        on.exit(assign(".Random.seed", RNGstate, envir = .GlobalEnv))
        mu_fun <- object$Funs$mu_fun
        if (is.null(link_fun)) {
            link_fun <- object$family$linkfun
        }
        if (is.null(link_fun)) {
            stop("you must specify the 'link_fun' argument.\n")
        }
        Xbetas <- c(X %*% betas)
        if (!is.null(offset)) {
            Xbetas <- Xbetas + offset
        }
        if (!is.null(gammas)) {
            eta_zi <- c(X_zi %*% gammas)
            if (!is.null(offset_zi)) {
                eta_zi <- eta_zi + offset_zi
            }
        }
        id <- match(object$id[[1]], unique(object$id[[1]]))
        nRE <- ncol(D)
        N <- nrow(X)
        n <- length(unique(id))
        eS <- eigen(D, symmetric = TRUE)
        ev <- eS$values
        V <- eS$vectors %*% diag(sqrt(pmax(ev, 0)), nRE)
        marg_inv_mu <- numeric(N)
        for (i in seq_len(n)) {
            set.seed(seed + i)
            id_i <- id == i
            b <- V %*% matrix(rnorm(M * nRE), nRE, M)
            Zb <- Z[id_i, , drop = FALSE] %*% b[seq_len(ncol(Z)), , drop = FALSE]
            mu <- mu_fun(Xbetas[id_i] + Zb)
            if (!is.null(gammas)) {
                eta_zi_id_i <- eta_zi[id_i]
                if (!is.null(object$Terms$termsZ_zi)) {
                    eta_zi_id_i <- eta_zi_id_i + Z_zi[id_i, , drop = FALSE] %*% 
                        b[-seq_len(ncol(Z)), , drop = FALSE]
                }
                mu <- plogis(eta_zi_id_i, lower.tail = FALSE) * mu
            }
            marg_inv_mu[id_i] <- link_fun(rowMeans(mu))
        }
        res <- c(solve(crossprod(X), crossprod(X, marg_inv_mu)))
        names(res) <- names(betas)
        res
    }
    out <- list(betas = compute_marg_coefs(object, X, betas, Z, X_zi, gammas, Z_zi, D, M, 
                                           link_fun, seed))
    if (std_errors) {
        blocks <- split(seq_len(K), rep(seq_len(cores), each = ceiling(K / cores),
                                        length.out = K))
        D <- object$D
        diag_D <- ncol(D) > 1 && all(abs(D[lower.tri(D)]) < sqrt(.Machine$double.eps))
        list_thetas <- list(betas = betas, D = if (diag_D) log(diag(D)) else chol_transf(D))
        if (!is.null(object$phis)) {
            list_thetas <- c(list_thetas, list(phis = object$phis))
        }
        if (!is.null(gammas)) {
            list_thetas <- c(list_thetas, list(gammas = gammas))
        }
        tht <- unlist(as.relistable(list_thetas))
        V <- vcov(object, sandwich = sandwich)
        cluster_compute_marg_coefs <- function (block, tht, list_thetas, V, XX, Z, X_zi, 
                                                Z_zi, M, compute_marg_coefs, chol_transf,
                                                object, link_fun, seed) {
            if (!exists(".Random.seed", envir = .GlobalEnv)) 
                runif(1L)
            RNGstate <- get(".Random.seed", envir = .GlobalEnv)
            on.exit(assign(".Random.seed", RNGstate, envir = .GlobalEnv))
            n_block <- length(block)
            m_betas <- matrix(0.0, n_block, length(list_thetas[["betas"]]))
            for (b in seq_along(block)) {
                seed. <- seed + block[b]
                set.seed(seed.)
                new_tht <- relist(MASS::mvrnorm(1, tht, V), skeleton = list_thetas)
                new_betas <- new_tht$betas
                new_D <- if (diag_D) diag(exp(new_tht$D), length(new_tht$D)) else chol_transf(new_tht$D)
                new_gammas <- new_tht$gammas
                m_betas[b, ] <- compute_marg_coefs(object, XX, new_betas, Z, X_zi, 
                                                   new_gammas, Z_zi, new_D, M, link_fun, 
                                                   seed = seed.)
            }
            m_betas
        }
        if (cores > 1L) {
            cl <- parallel::makeCluster(cores)
            parallel::clusterSetRNGStream(cl = cl, iseed = seed)
            res <- parallel::parLapply(cl, blocks, cluster_compute_marg_coefs, tht = tht,
                                       list_thetas = list_thetas, V = V, XX = X, Z = Z, 
                                       X_zi = X_zi, Z_zi = Z_zi, M = M,
                                       object = object, compute_marg_coefs = compute_marg_coefs,
                                       chol_transf = chol_transf, link_fun = link_fun, seed = seed)
            parallel::stopCluster(cl)
        } else {
            res <- lapply(blocks, cluster_compute_marg_coefs, tht = tht,
                          list_thetas = list_thetas, V = V, XX = X, Z = Z, 
                          X_zi = X_zi, Z_zi = Z_zi, M = M,
                          object = object, compute_marg_coefs = compute_marg_coefs,
                          chol_transf = chol_transf, link_fun = link_fun, seed = seed)
        }
        out$MC_Ksamples <- do.call("rbind", res)
        out$var_betas <- var(do.call("rbind", res))
        dimnames(out$var_betas) <- list(names(out$betas), names(out$betas))
        ses <- sqrt(diag(out$var_betas))
        coef_table <- cbind("Estimate" = out$betas, "Std.Err" = ses,
                            "z-value" = out$betas / ses,
                            "p-value" = 2 * pnorm(abs(out$betas / ses), lower.tail = FALSE))
        out$coef_table <- coef_table
    }
    class(out) <- "m_coefs"
    out
}

# print.m_coefs / coef.m_coefs / vcov.m_coefs
# Methods for the "m_coefs" class returned by marginal_coefs.MixMod().
# print.m_coefs: displays the marginalized coefficients (with a table if std_errors=TRUE)
# coef.m_coefs:  extracts the betas vector
# vcov.m_coefs:  extracts the variance-covariance matrix (NULL if std_errors was FALSE)
print.m_coefs <- function (x, digits = max(4, getOption("digits") - 4), ...) {
    if (is.null(x$coef_table)) {
        print(round(x$betas, digits = digits))
    } else {
        coef_table <- as.data.frame(x$coef_table)
        coef_table[1:3] <- lapply(coef_table[1:3], round, digits = digits)
        coef_table[["p-value"]] <- format.pval(coef_table[["p-value"]], eps = 1e-04)
        print(coef_table)
        cat("\n")
    }
    invisible(x)
}

coef.m_coefs <- function (object, ...) {
    object$betas
}

vcov.m_coefs <- function (object, ...) {
    if (is.null(object$coef_table)) {
        NULL
    } else {
        object$var_betas
    }
}

# effectPlotData (generic)
# Generic function for computing effect plot data. Dispatches to effectPlotData.MixMod().
effectPlotData <- function (object, newdata, level, ...) UseMethod("effectPlotData")

# effectPlotData.MixMod
# Computes predicted values and confidence intervals for a grid of covariate values
# (newdata), suitable for creating effect plots. Supports both mean-subject predictions
# (random effects = 0) and marginal predictions (averaged over the random-effects dist.).
# Also handles continuation ratio (CR) models by computing marginal category probabilities
# via cr_marg_probs().
#
# The confidence intervals are computed via the delta method:
#   CI = eta_hat ± z_{alpha/2} * sqrt(X_new * V_beta * X_new')
#
# Arguments:
#   object:            a fitted MixMod object
#   newdata:           data.frame with covariate values for predictions
#   level:             confidence level (default: 0.95)
#   marginal:          logical; if TRUE computes marginal (population-averaged) predictions
#   CR_cohort_varname: character; variable name for the CR cohort in newdata (for CR models)
#   direction:         "forward" or "backward" for CR models (passed to cr_marg_probs())
#   K, seed:           passed to marginal_coefs() when marginal = TRUE
#   sandwich:          logical; use sandwich SE (default: FALSE)
#   ...:               passed to marginal_coefs()
#
# Returns:
#   The newdata data.frame with additional columns: pred (point prediction on linear
#   predictor scale), low (lower CI), upp (upper CI), and exp_pred/exp_low/exp_upp
#   (on the response scale) for non-CR models.
effectPlotData.MixMod <- function (object, newdata, level = 0.95, marginal = FALSE,
                                   CR_cohort_varname = NULL, direction = NULL,
                                   K = 200, seed = 1, sandwich = FALSE, ...) {
    termsX <- delete.response(object$Terms$termsX)
    mfX <- model.frame(termsX, newdata, 
                       xlev = .getXlevels(termsX, object$model_frames$mfX))
    X <- model.matrix(termsX, mfX)
    if (is.null(object$gammas)) {
        if (marginal) {
            mcoefs <- marginal_coefs(object, std_errors = TRUE, seed = seed, ...)
            betas <- mcoefs$betas
            var_betas <- mcoefs$var_betas
        } else {
            betas <- fixef(object)
            var_betas <- vcov(object, parm = "fixed-effects", sandwich = sandwich)
        }
        if (is.null(CR_cohort_varname)) {
            pred <- c(X %*% betas)
            ses <- sqrt(diag(X %*% var_betas %*% t(X)))
            newdata$pred <- pred
            newdata$low <- pred + qnorm((1 - level) / 2) * ses
            newdata$upp <- pred + qnorm((1 + level) / 2) * ses
        } else {
            if (!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) 
                runif(1L)
            R.seed <- get(".Random.seed", envir = .GlobalEnv)
            set.seed(seed)
            RNGstate <- structure(seed, kind = as.list(RNGkind()))
            on.exit(assign(".Random.seed", R.seed, envir = .GlobalEnv))
            ##
            cohort_var <- newdata[[CR_cohort_varname]]
            nlvs <- nlevels(cohort_var)
            eta <- do.call("cbind", split(c(X %*% betas), cohort_var))
            logit_marg_probs <- qlogis(cr_marg_probs(eta, direction = direction))
            new_betas <- MASS::mvrnorm(K, betas, var_betas)
            sim_marg_probs <- array(0.0, dim = c(dim(logit_marg_probs), K))
            for (k in seq_len(K)) {
                eta_k <- do.call("cbind", split(c(X %*% new_betas[k, ]), cohort_var))
                sim_marg_probs[, , k] <- qlogis(cr_marg_probs(eta_k, direction = direction))
            }
            logit_marg_probs_low <- apply(sim_marg_probs, c(1, 2), quantile, 
                                          probs = (1 - level) / 2)
            logit_marg_probs_upp <- apply(sim_marg_probs, c(1, 2), quantile, 
                                          probs = (1 + level) / 2)
            newdata <- newdata[cohort_var == levels(cohort_var)[1L], 
                               names(newdata) != CR_cohort_varname, drop = FALSE]
            newdata <- do.call("rbind", rep(list(newdata), nlvs + 1)); row.names(newdata) <- NULL
            newdata[["ordinal_response"]] <- rep(seq_len(nlvs + 1), each = nrow(newdata) / (nlvs + 1))
            newdata$pred <- c(logit_marg_probs)
            newdata$low <- c(logit_marg_probs_low)
            newdata$upp <- c(logit_marg_probs_upp)
        }
    } else {
        if (!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) 
            runif(1)
        R.seed <- get(".Random.seed", envir = .GlobalEnv)
        set.seed(seed)
        RNGstate <- structure(seed, kind = as.list(RNGkind()))
        on.exit(assign(".Random.seed", R.seed, envir = .GlobalEnv))
        mu_fun <- object$Funs$mu_fun
        link_fun <- object$family$linkfun
        betas <- fixef(object)
        gammas <- fixef(object, sub_model = "zero_part")
        termsX_zi <- object$Terms$termsX_zi
        mfX_zi <- model.frame(termsX_zi, newdata, 
                              xlev = .getXlevels(termsX_zi, object$model_frames$mfX_zi))
        X_zi <- model.matrix(termsX_zi, mfX_zi)
        list_thetas <- list(betas = betas, gammas = gammas)
        tht <- unlist(as.relistable(list_thetas))
        V <- vcov(object, sandwich = sandwich)
        ind <- c(seq_len(length(betas)), grep("zi_", colnames(V), fixed = TRUE))
        V <- V[ind, ind, drop = FALSE]
        new_tht <- MASS::mvrnorm(K, tht, V)
        if (marginal) {
            mcoefs <- marginal_coefs(object, std_errors = TRUE, seed = seed, 
                                     ...)
            betas <- mcoefs$betas
            var_betas <- mcoefs$var_betas
            pred <- c(X %*% betas)
            ses <- sqrt(diag(X %*% var_betas %*% t(X)))
            newdata$pred <- pred
            newdata$low <- pred + qnorm((1 - level) / 2) * ses
            newdata$upp <- pred + qnorm((1 + level) / 2) * ses
        } else {
            eta_y <- c(X %*% betas)
            eta_zi <- c(X_zi %*% gammas)
            pred <- link_fun(mu_fun(eta_y) / (1 + exp(eta_zi)))
            Preds <- matrix(0.0, length(pred), K)
            for (k in seq_len(K)) {
                thetas_k <- relist(new_tht[k, ], skeleton = list_thetas)
                betas_k <- thetas_k$betas
                gammas_k <- thetas_k$gammas
                eta_y <- c(X %*% betas_k)
                eta_zi <- c(X_zi %*% gammas_k)
                Preds[, k] <- link_fun(mu_fun(eta_y) / (1 + exp(eta_zi)))
            }
            newdata$pred <- pred
            Qs <- apply(Preds, 1, quantile, probs = c((1 - level) / 2, (1 + level) / 2))
            newdata$low <- Qs[1, ]
            newdata$upp <- Qs[2, ]
        }
    }
    newdata
}

# create_lists
# Helper function that organizes a new data.frame (newdata) into the per-group list
# format needed by predict.MixMod(). Extracts design matrices (X, Z, X_zi, Z_zi),
# handles missing data (complete-cases filtering), and arranges observations into
# per-group sublists matching the structure used during model fitting.
#
# Arguments:
#   object:  a fitted MixMod object (provides model terms, factor levels, ID name)
#   newdata: a data.frame with the same variables as the original data, including the
#            grouping variable
#
# Returns:
#   A list with components: y_lis, X_lis, Z_lis, offset_lis, X_zi_lis, Z_zi_lis,
#   offset_zi_lis, id, id_unq, keep (indicator of complete cases), and newdata
#   (filtered to complete cases).
create_lists <- function (object, newdata) {
    if (!inherits(object, "MixMod")) {
        stop("only works for 'MixMod' objects.")
    }
    id_nam <- object$id_name
    id <- newdata[[id_nam]]
    id <- match(id, unique(id))
    id_unq <- unique(id)
    # terms & model frames
    termsX <- object$Terms$termsX
    mfX <- model.frame(termsX, newdata, 
                       xlev = .getXlevels(termsX, object$model_frames$mfX))
    termsZ <- object$Terms$termsZ
    mfZ <- mapply(model.frame.default, formula = termsZ, 
                  xlev = mapply(.getXlevels, termsZ, object$model_frames$mfZ, SIMPLIFY = FALSE), 
                  MoreArgs = list(data = newdata), SIMPLIFY = FALSE)
    if (!is.null(object$gammas)) {
        termsX_zi <- delete.response(object$Terms$termsX_zi)
        mfX_zi <- model.frame(termsX_zi, newdata, 
                              xlev = .getXlevels(termsX_zi, object$model_frames$mfX_zi))
        if (!is.null(object$Terms$termsZ_zi)) {
            termsZ_zi <- object$Terms$termsZ_zi
            mfZ_zi <- mapply(model.frame.default, formula = termsZ_zi, 
                             xlev = mapply(.getXlevels, termsZ_zi, object$model_frames$mfZ_zi, 
                                           SIMPLIFY = FALSE), 
                             MoreArgs = list(data = newdata), SIMPLIFY = FALSE)
        }
    }
    # delete missing data
    complete_cases <- cbind(complete.cases(mfX), sapply(mfZ, complete.cases))
    if (!is.null(object$gammas))
        complete_cases <- cbind(complete_cases, complete.cases(mfX_zi))
    if (!is.null(object$Terms$termsZ_zi))
        complete_cases <- cbind(complete_cases, complete.cases(mfZ_zi))
    keep <- apply(complete_cases, 1, all)
    mfX <- mfX[keep, , drop = FALSE]
    mfZ[] <- lapply(mfZ, function (mf) mf[keep, , drop = FALSE])
    if (!is.null(object$gammas))
        mfX_zi <- mfX_zi[keep, , drop = FALSE]
    if (!is.null(object$Terms$termsZ_zi))
        mfZ_zi[] <- lapply(mfZ_zi, function (mf) mf[keep, , drop = FALSE])
    # response vector and design matrices
    y <- model.response(mfX)
    if (is.null(y)) {
        stop("the outcome variable does seem to exist in 'newdata'; do you perhaps want ",
             "'mean_subject' predictions?\n")
    }
    if (is.factor(y)) {
        y <- as.numeric(y != levels(y)[1L])
    }
    offset <- model.offset(mfX)
    X <- model.matrix(termsX, mfX)
    Z <- mapply(constructor_Z, termsZ, mfZ, MoreArgs = list(id = id), SIMPLIFY = FALSE)
    Z <- do.call("cbind", Z)
    if (!is.null(object$gammas)) {
        offset_zi <- model.offset(mfX_zi)
        X_zi <- model.matrix(termsX_zi, mfX_zi)
        if (!is.null(object$Terms$termsZ_zi)) {
            Z_zi <- mapply(constructor_Z, termsZ_zi, mfZ_zi, MoreArgs = list(id = id), 
                        SIMPLIFY = FALSE)
            Z_zi <- do.call("cbind", Z_zi)
        } else {
            Z_zi <- NULL
        }
    } else {
        X_zi <- offset_zi <- Z_zi <- NULL
    }
    y_lis <- if (is.matrix(y)) lapply(id_unq, function (i) y[id == i, , drop = FALSE]) else split(y, id)
    N <- if (NCOL(y) == 2) y[, 1] + y[, 2]
    N_lis <- if (NCOL(y) == 2) split(N, id)
    X_lis <- lapply(id_unq, function (i) X[id == i, , drop = FALSE])
    Z_lis <- lapply(id_unq, function (i) Z[id == i, , drop = FALSE])
    offset_lis <- if (!is.null(offset)) split(offset, id)
    Zty_fun <- function (z, y) {
        if (NCOL(y) == 2) crossprod(z, y[, 1]) else crossprod(z, y)
    }
    Zty_lis <- lapply(mapply(Zty_fun, Z_lis, y_lis, SIMPLIFY = FALSE), drop)
    Xty <- drop(if (NCOL(y) == 2) crossprod(X, y[, 1]) else crossprod(X, y))
    X_zi_lis <- if (!is.null(X_zi)) lapply(id_unq, function (i) X_zi[id == i, , drop = FALSE])
    Z_zi_lis <- if (!is.null(Z_zi)) lapply(id_unq, function (i) Z_zi[id == i, , drop = FALSE])
    offset_zi_lis <- if (!is.null(offset_zi)) split(offset_zi, id)
    log_dens <- object$Funs$log_dens
    mu_fun <- object$Funs$mu_fun
    var_fun <- object$Funs$var_fun
    mu.eta_fun <- object$Funs$mu.eta_fun
    score_eta_fun <- object$Funs$score_eta_fun
    score_phis_fun <- object$Funs$score_phis_fun
    score_eta_zi_fun <- object$Funs$score_eta_zi_fun
    family <- object$family
    canonical <- !is.null(family$family) &&
        ((family$family == "binomial" && family$link == "logit") ||
             (family$family == "poisson" && family$link == "log"))
    known_families <- c("binomial", "poisson")
    user_defined <- !family$family %in% known_families
    numer_deriv <- if (object$control$numeric_deriv == "fd") fd else cd
    numer_deriv_vec <- if (object$control$numeric_deriv == "fd") fd_vec else cd_vec
    nRE <- if (!is.null(Z_zi)) ncol(Z) + ncol(Z_zi) else ncol(Z)
    start <- matrix(0.0, length(y_lis), nRE)
    betas <- fixef(object)
    invD <- solve(object$D)
    phis <- object$phis
    gammas <- object$gammas
    list(y_lis = y_lis, N_lis = N_lis, X_lis = X_lis, Z_lis = Z_lis, Z = Z, Z_zi = Z_zi,
         X_zi_lis = X_zi_lis, Z_zi_lis = Z_zi_lis, offset_zi_lis = offset_zi_lis,
         id = id, id_nam = id_nam, offset_lis = offset_lis, betas = betas, invD = invD, 
         phis = phis, gammas = gammas, start = start, canonical = canonical, 
         user_defined = user_defined, Zty_lis = Zty_lis, log_dens = log_dens, 
         mu_fun = mu_fun, var_fun = var_fun, 
         mu.eta_fun = mu.eta_fun, score_eta_fun = score_eta_fun, termsZ = termsZ,
         score_phis_fun = score_phis_fun, score_eta_zi_fun = score_eta_zi_fun)
}

# predict.MixMod
# Computes predictions (with optional confidence intervals) from a fitted MixMod object
# for new covariate data. Supports four prediction types:
#   "mean_subject":     predictions at the population mean (random effects = 0)
#   "subject_specific": predictions conditional on estimated/new-group random effects;
#                       for subjects with observed data, uses their estimated random effects;
#                       for new subjects (in newdata2), samples random effects from the
#                       posterior using a Metropolis-Hastings algorithm
#   "marginal":         population-averaged predictions via marginal_coefs()
#   "zero_part":        predictions for the zero-inflation probability plogis(X_zi * gammas)
#
# When se.fit = TRUE, confidence intervals are computed via the delta method (for
# mean_subject and marginal) or by simulating from the posterior of the parameters
# (for subject_specific with new subjects, using MH sampling).
#
# Arguments:
#   object:         a fitted MixMod object
#   newdata:        data.frame with covariate values for prediction
#   newdata2:       data.frame for new subjects not in the training data (for subject_specific)
#   type_pred:      "response" (default) or "link" (on the linear predictor scale)
#   type:           prediction type (see above)
#   se.fit:         logical; if TRUE computes standard errors/CIs (default: FALSE)
#   M:              number of Metropolis-Hastings samples for new subjects (default: 300)
#   df:             degrees of freedom for the t proposal in MH (default: 10)
#   scale:          scale for the t proposal in MH (default: 0.3)
#   level:          confidence level (default: 0.95)
#   seed:           random seed for MH sampling (default: 1)
#   return_newdata: logical; if TRUE returns newdata with predictions appended (default: FALSE)
#   sandwich:       logical; use sandwich SE for marginal predictions (default: FALSE)
#   ...:            currently ignored
#
# Returns:
#   A numeric vector of predictions (or a list with $pred and $se.fit if se.fit = TRUE),
#   or the newdata data.frame with appended prediction columns if return_newdata = TRUE.
predict.MixMod <- function (object, newdata, newdata2 = NULL,
                            type_pred = c("response", "link"),
                            type = c("mean_subject", "subject_specific", "marginal", "zero_part"),
                            se.fit = FALSE, M = 300, df = 10, scale = 0.3, level = 0.95,
                            seed = 1, return_newdata = FALSE, sandwich = FALSE, ...) {
    type_pred <- match.arg(type_pred)
    type <- match.arg(type)
    if (!is.null(object$gammas) && type != "zero_part" && type_pred == "link") {
        warning("for model with an extra zero-part only predictions at the level of the ",
                "response variable are returned;\n'type_pred' is set to 'response'.")
        type_pred <- "response"
    }
    if (missing(newdata)) {
        newdata <- object$data
    }
    termsX <- delete.response(object$Terms$termsX)
    mfX <- model.frame(termsX, newdata, 
                       xlev = .getXlevels(termsX, object$model_frames$mfX))
    X <- model.matrix(termsX, mfX)
    offset <- model.offset(mfX)
    if (!is.null(object$gammas)) {
        termsX_zi <- delete.response(object$Terms$termsX_zi)
        mfX_zi <- model.frame(termsX_zi, newdata, 
                              xlev = .getXlevels(termsX_zi, object$model_frames$mfX_zi))
        X_zi <- model.matrix(termsX_zi, mfX_zi)
        offset_zi <- model.offset(mfX_zi)
        gammas <- fixef(object, sub_model = "zero_part")
        eta_zi <- c(X_zi %*% gammas)
        if (!is.null(offset_zi)) {
            eta_zi <- eta_zi + offset_zi
        }
    }
    if (type %in% c("mean_subject", "marginal")) {
        if (type == "mean_subject") {
            betas <- fixef(object)
            var_betas <- vcov(object, parm = "fixed-effects", sandwich = sandwich)
            eta_y <- c(X %*% betas)
            if (!is.null(offset)) {
                eta_y <- eta_y + offset
            }
            pred <- if (type_pred == "link") eta_y else object$family$linkinv(eta_y)
            if (!is.null(object$gammas)) {
                if (object$family$family == "hurdle log-normal") {
                    pred <- exp(pred + 0.5 * exp(object$phis)^2)
                }
                pred <- plogis(eta_zi, lower.tail = FALSE) * pred
            }
            names(pred) <- row.names(newdata)
            se_fit <- if (se.fit && is.null(object$gammas)) sqrt(diag(X %*% var_betas %*% t(X)))
        } else {
            if (!is.null(object$gammas)) {
                stop("the predict() method is not yet implemented for models with an extra zero-part.")
            }
            mcoefs <- marginal_coefs(object, std_errors = TRUE, ...)
            betas <- coef(mcoefs)
            var_betas <- mcoefs$var_betas
            pred <- if (type_pred == "link") c(X %*% betas) else object$family$linkinv(c(X %*% betas))
            names(pred) <- row.names(newdata)
            se_fit <- if (se.fit) sqrt(diag(X %*% var_betas %*% t(X)))
        }
    } else if (type == "zero_part") {
        if (is.null(object$gammas))
            stop("the fitted model does not have an extra zero part.")
        pred <- if (type_pred == "link") eta_zi else plogis(eta_zi)
        names(pred) <- row.names(newdata)
        var_gammas <- vcov(object, parm = "zero_part", sandwich = sandwich)
        se_fit <- if (se.fit) sqrt(diag(X_zi %*% var_gammas %*% t(X_zi)))
    } else {
        Lists <- create_lists(object, newdata)
        id <- Lists[["id"]]
        betas <- Lists[["betas"]]
        gammas <- Lists[["gammas"]]
        phis <- Lists[["phis"]]
        Z <- Lists[["Z"]]
        ncz <- ncol(Z)
        Z_zi <- Lists[["Z_zi"]]
        EBs <- find_modes(b = Lists$start, y_lis = Lists[["y_lis"]], 
                          N_lis = Lists[["N_lis"]], X_lis = Lists[["X_lis"]], 
                          Z_lis = Lists[["Z_lis"]], offset_lis = Lists[["offset_lis"]], 
                          X_zi_lis = Lists[["X_zi_lis"]], Z_zi_lis = Lists[["Z_zi_lis"]], 
                          offset_zi_lis = Lists[["offset_zi_lis"]], 
                          betas = betas, invD = Lists[["invD"]], phis = Lists[["phis"]], 
                          gammas = gammas, canonical = Lists[["canonical"]], 
                          user_defined = Lists[["user_defined"]], 
                          Zty_lis = Lists[["Zty_lis"]], log_dens = Lists[["log_dens"]], 
                          mu_fun = Lists[["mu_fun"]], var_fun = Lists[["var_fun"]], 
                          mu.eta_fun = Lists[["mu.eta_fun"]], 
                          score_eta_fun = Lists[["score_eta_fun"]], 
                          score_phis_fun = Lists[["score_phis_fun"]], 
                          score_eta_zi_fun = Lists[["score_eta_zi_fun"]])
        eta <- c(X %*% betas) + rowSums(Z * EBs$post_modes[id, seq_len(ncz), drop = FALSE])
        if (!is.null(offset))
            eta <- eta + offset
        if (!is.null(object$Terms$termsZ_zi)) {
            eta_zi <- eta_zi + rowSums(Z_zi * EBs$post_modes[id, -seq_len(ncz), drop = FALSE])
        }
        pred <- if (type_pred == "link") eta else object$family$linkinv(eta)
        if (!is.null(object$gammas)) {
            if (object$family$family == "hurdle log-normal") {
                pred <- exp(pred + 0.5 * exp(phis)^2)
            }
            pred <- plogis(eta_zi, lower.tail = FALSE) * pred
            attr(pred, "zi_probs") <- plogis(eta_zi)
        }
        names(pred) <- row.names(newdata)
        if (se.fit) {
            if (!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) 
                runif(1)
            R.seed <- get(".Random.seed", envir = .GlobalEnv)
            set.seed(seed)
            RNGstate <- structure(seed, kind = as.list(RNGkind()))
            on.exit(assign(".Random.seed", R.seed, envir = .GlobalEnv))
            log_post_b <- function (b_i, y_i, X_i, Z_i, offset_i, X_zi_i, Z_zi_i, offset_zi_i,
                                    betas, invD, phis, gammas, log_dens, mu_fun) {
                ncz <- ncol(Z_i)
                eta_y <- as.vector(X_i %*% betas + Z_i %*% b_i[seq_len(ncz)])
                if (!is.null(offset_i))
                    eta_y <- eta_y + offset_i
                if (!is.null(X_zi_i)) {
                    eta_zi <- as.vector(X_zi_i %*% gammas)
                    if (!is.null(Z_zi_i))
                        eta_zi <- eta_zi + as.vector(Z_zi_i %*% b_i[-seq_len(ncz)])
                    if (!is.null(offset_zi_i))
                        eta_zi <- eta_zi + offset_zi_i
                }
                sum(log_dens(y_i, eta_y, mu_fun, phis, eta_zi), na.rm = TRUE) -
                    c(0.5 * crossprod(b_i, invD) %*% b_i)
            }
            log_dens <- object$Funs$log_dens
            mu_fun <- object$Funs$mu_fun
            calc_alpha <- function (log_post_new, log_post_old, log_prop_new, 
                                    log_prop_old) {
                min(exp(log_post_new + log_prop_old - log_post_old - log_prop_new), 1)
            }
            phis <- object$phis
            D <- object$D
            diag_D <- ncol(D) > 1 && all(abs(D[lower.tri(D)]) < sqrt(.Machine$double.eps))
            list_thetas <- list(betas = betas, D = if (diag_D) log(diag(D)) else chol_transf(D))
            if (!is.null(phis)) {
                list_thetas <- c(list_thetas, list(phis = phis))
            }
            if (!is.null(gammas)) {
                list_thetas <- c(list_thetas, list(gammas = gammas))
            }
            tht <- unlist(as.relistable(list_thetas))
            V <- vcov(object, sandwich = sandwich)
            tht_new <- MASS::mvrnorm(M, tht, V)
            row_split_ind <- row(EBs$post_modes)
            mu <- split(EBs$post_modes, row_split_ind)
            Sigma <- lapply(EBs$post_hessians, solve)
            scale <- rep(scale, length.out = length(Sigma))
            Sigma <- mapply("*", scale, Sigma, SIMPLIFY = FALSE)
            EBs_proposed <- mapply(rmvt, mu = mu, Sigma = Sigma, SIMPLIFY = FALSE,
                                   MoreArgs = list(n = M, df = df))
            dmvt_proposed <- mapply(dmvt, x = EBs_proposed, mu = mu, Sigma = Sigma,
                                    MoreArgs = list(df = df, log = TRUE, prop = FALSE),
                                    SIMPLIFY = FALSE)
            b_current <- mu
            dmvt_current <- mapply(dmvt, x = mu, mu = mu, Sigma = Sigma, SIMPLIFY = FALSE,
                                   MoreArgs = list(df = df, log = TRUE, prop = FALSE))
            y_lis <- Lists[["y_lis"]]
            X_lis <- Lists[["X_lis"]]
            Z_lis <- Lists[["Z_lis"]]
            offset_lis <- Lists[["offset_lis"]]
            if (is.null(offset_lis))
                offset_lis <- rep(list(NULL), length(y_lis))
            X_zi_lis <- Lists[["X_zi_lis"]]
            if (is.null(X_zi_lis))
                X_zi_lis <- rep(list(NULL), length(y_lis))
            Z_zi_lis <- Lists[["Z_zi_lis"]]
            if (is.null(Z_zi_lis))
                Z_zi_lis <- rep(list(NULL), length(y_lis))
            offset_zi_lis <- Lists[["offset_zi_lis"]]
            if (is.null(offset_zi_lis))
                offset_zi_lis <- rep(list(NULL), length(y_lis))
            n <- length(pred)
            Preds <- matrix(0.0, n, M)
            b <- vector("list", M)
            if (!is.null(object$gammas)) {
                zi_probs <- matrix(0.0, n, M)
            }
            success_rate <- matrix(FALSE, M, length(y_lis))
            for (m in seq_len(M)) {
                # Extract simulared new parameter values
                new_pars <- relist(tht_new[m, ], skeleton = list_thetas)
                betas_m <- new_pars$betas
                phis_m <- new_pars$phis
                gammas_m <- new_pars$gammas
                D_m <- if (diag_D) diag(exp(new_pars$D), length(new_pars$D)) else chol_transf(new_pars$D)
                invD_m <- solve(D)
                # Simulate new EBs
                log_post_b_current <- mapply(log_post_b, b_i = b_current, y_i = y_lis, 
                                             X_i = X_lis, Z_i = Z_lis, offset_i = offset_lis,
                                             X_zi_i = X_zi_lis, Z_zi_i = Z_zi_lis, 
                                             offset_zi_i = offset_zi_lis,
                                             MoreArgs = list(betas = betas_m, invD = invD_m, 
                                                             phis = phis_m, gammas = gammas_m,
                                                             log_dens = log_dens, 
                                                             mu_fun = mu_fun),
                                             SIMPLIFY = FALSE)
                b_new <- lapply(EBs_proposed, function (x, m) x[m, ], m = m)
                log_post_b_new <- mapply(log_post_b, b_i = b_new, y_i = y_lis, X_i = X_lis, 
                                         Z_i = Z_lis, offset_i = offset_lis, 
                                         X_zi_i = X_zi_lis, Z_zi_i = Z_zi_lis, 
                                         offset_zi_i = offset_zi_lis,
                                         MoreArgs = list(betas = betas_m, invD = invD_m, 
                                                         phis = phis_m, gammas = gammas_m,
                                                         log_dens = log_dens, 
                                                         mu_fun = mu_fun),
                                         SIMPLIFY = FALSE)
                alphas <- mapply(calc_alpha, log_post_b_new, log_post_b_current, dmvt_current, 
                                 lapply(dmvt_proposed, "[", m))
                keep_ind <- runif(length(alphas)) <= alphas
                if (any(keep_ind)) {
                    b_current[keep_ind] <- b_new[keep_ind]
                    dmvt_current[keep_ind] <- lapply(dmvt_proposed, "[", m)[keep_ind]
                }
                success_rate[m, ] <- keep_ind
                # Calculate Predictions
                b[[m]] <- do.call("rbind", b_current)
                eta <- c(X %*% betas_m) + rowSums(Z * b[[m]][id, seq_len(ncz), drop = FALSE])
                if (!is.null(offset))
                    eta <- eta + offset
                Preds[, m] <- if (type_pred == "link") eta else object$family$linkinv(eta)
                if (!is.null(object$gammas)) {
                    eta_zi <- as.vector(X_zi %*% gammas_m)
                    if (!is.null(object$Terms$termsZ_zi)) {
                        eta_zi <- eta_zi + rowSums(Z_zi * b[[m]][id, -seq_len(ncz), drop = FALSE])
                    }
                    if (!is.null(offset_zi))
                        eta_zi <- eta_zi + offset_zi
                    Preds[, m] <- plogis(eta_zi, lower.tail = FALSE) * Preds[, m]
                    zi_probs[, m] <- plogis(eta_zi)
                }
            }
            se_fit <- apply(Preds, 1, sd, na.rm = TRUE)
            Qs <- apply(Preds, 1, quantile, 
                        probs = c((1 - level) / 2, (1 + level) / 2))
            low <- Qs[1, ]
            upp <- Qs[2, ]
            if (!is.null(gammas)) {
                Qs_zi <- apply(zi_probs, 1, quantile, 
                               probs = c((1 - level) / 2, (1 + level) / 2))
                
                attr(low, "zi_probs") <- Qs_zi[1, ]
                attr(upp, "zi_probs") <- Qs_zi[2, ]
            }
            names(se_fit) <- names(low) <- names(upp) <- names(pred)
        }
        if (!is.null(newdata2)) {
            id_nam <- Lists[["id_nam"]]
            id2 <- newdata2[[id_nam]]
            id2 <- match(id2, unique(newdata[[id_nam]]))
            # terms & model frames
            mfX2 <- model.frame(termsX, newdata2, 
                                xlev = .getXlevels(termsX, object$model_frames$mfX))
            termsZ <- Lists[["termsZ"]]
            mfZ2 <- mapply(model.frame.default, formula = termsZ, 
                           xlev = mapply(.getXlevels, termsZ, object$model_frames$mfZ, SIMPLIFY = FALSE), 
                           MoreArgs = list(data = newdata2), SIMPLIFY = FALSE)
            if (!is.null(object$gammas)) {
                termsX_zi <- object$Terms$termsX_zi
                mfX2_zi <- model.frame(termsX_zi, newdata2, 
                                       xlev = .getXlevels(termsX_zi, object$model_frames$mfX_zi))
                if (!is.null(object$Terms$termsZ_zi)) {
                    termsZ_zi <- object$Terms$termsZ_zi
                    mfZ2_zi <- mapply(model.frame.default, formula = termsZ_zi, 
                                      xlev = mapply(.getXlevels, termsZ_zi, object$model_frames$mfZ_zi, 
                                                    SIMPLIFY = FALSE), 
                                      MoreArgs = list(data = newdata2), SIMPLIFY = FALSE)
                }
            }
            # delete missing data
            complete_cases <- cbind(complete.cases(mfX2), sapply(mfZ2, complete.cases))
            if (!is.null(object$gammas))
                complete_cases <- cbind(complete_cases, complete.cases(mfX2_zi))
            if (!is.null(object$Terms$termsZ_zi))
                complete_cases <- cbind(complete_cases, complete.cases(mfZ2_zi))
            keep <- apply(complete_cases, 1, all)
            mfX2 <- mfX2[keep, , drop = FALSE]
            mfZ2[] <- lapply(mfZ2, function (mf) mf[keep, , drop = FALSE])
            if (!is.null(object$gammas))
                mfX2_zi <- mfX2_zi[keep, , drop = FALSE]
            if (!is.null(object$Terms$termsZ_zi))
                mfZ2_zi[] <- lapply(mfZ2_zi, function (mf) mf[keep, , drop = FALSE])
            # design matrices
            X2 <- model.matrix(termsX, mfX2)
            offset2 <- model.offset(mfX2)
            Z2 <- mapply(constructor_Z, termsZ, mfZ2, MoreArgs = list(id = id2), SIMPLIFY = FALSE)
            Z2 <- do.call("cbind", Z2)
            ncz <- ncol(Z2)
            eta2 <- c(X2 %*% betas) + rowSums(Z2 * EBs$post_modes[id2, seq_len(ncz), drop = FALSE])
            if (!is.null(offset2)) {
                eta2 <- eta2 + offset2
            }
            pred2 <- if (type_pred == "link") eta2 else object$family$linkinv(eta2)
            if (!is.null(object$gammas)) {
                X2_zi <- model.matrix(termsX_zi, mfX2_zi)
                offset2_zi <- model.offset(mfX2_zi)
                eta2_zi <- c(X2_zi %*% gammas)
                if (!is.null(object$Terms$termsZ_zi)) {
                    Z2_zi <- mapply(constructor_Z, termsZ_zi, mfZ2_zi, MoreArgs = list(id = id2), 
                                   SIMPLIFY = FALSE)
                    Z2_zi <- do.call("cbind", Z2_zi)
                    eta2_zi <- eta2_zi + rowSums(Z2_zi * EBs$post_modes[id2, -seq_len(ncz), drop = FALSE])
                }
                if (!is.null(offset2_zi))
                    eta2_zi <- eta2_zi + offset2_zi
                if (object$family$family == "hurdle log-normal") {
                    pred2 <- exp(pred2 + 0.5 * exp(phis)^2)
                }
                pred2 <- plogis(eta2_zi, lower.tail = FALSE) * pred2
                attr(pred2, "zi_probs") <- plogis(eta2_zi)
            }
            names(pred2) <- row.names(newdata2)
            if (se.fit) {
                Preds2 <- matrix(0.0, length(pred2), M)
                if (!is.null(gammas)) {
                    zi_probs2 <- matrix(0.0, length(pred2), M)
                }
                for (m in seq_len(M)) {
                    new_pars <- relist(tht_new[m, ], skeleton = list_thetas)
                    betas_m <- new_pars$betas
                    b_m <- b[[m]]
                    gammas_m <- new_pars$gammas
                    eta2 <- c(X2 %*% betas_m) + rowSums(Z2 * b_m[id2, seq_len(ncz), drop = FALSE])
                    if (!is.null(offset2))
                        eta2 <- eta2 + offset2
                    Preds2[, m] <- if (type_pred == "link") eta2 else object$family$linkinv(eta2)
                    if (!is.null(gammas)) {
                        eta2_zi <- c(X2_zi %*% gammas_m)
                        if (!is.null( object$Terms$termsZ_zi))
                            eta2_zi <- eta2_zi + rowSums(Z2_zi * b_m[id2, -seq_len(ncz), drop = FALSE])
                        if (!is.null(offset2_zi))
                            eta2_zi <- eta2_zi + offset2_zi
                        Preds2[, m] <- plogis(eta2_zi, lower.tail = FALSE) * Preds2[, m]
                        zi_probs2[, m] <- plogis(eta2_zi)
                    }
                }
                se_fit2 <- apply(Preds2, 1, sd, na.rm = TRUE)
                Qs2 <- apply(Preds2, 1, quantile, probs = c((1 - level) / 2, (1 + level) / 2))
                low2 <- Qs2[1, ]
                upp2 <- Qs2[2, ]
                if (!is.null(gammas)) {
                    Qs2_zi <- apply(zi_probs2, 1, quantile, 
                                    probs = c((1 - level) / 2, (1 + level) / 2))
                    
                    attr(low2, "zi_probs") <- Qs2_zi[1, ]
                    attr(upp2, "zi_probs") <- Qs2_zi[2, ]
                }
                names(se_fit2) <- names(low2) <- names(upp2) <- names(pred2)
            }
        }
    }
    if (return_newdata) {
        na_exclude <- attr(mfX, "na.action")
        if (!is.null(object$gammas)) { 
            na_exclude <- union(attr(mfX_zi, "na.action"), na_exclude)
        }
        if (!is.null(na_exclude))
            newdata <- newdata[-na_exclude, ]
        newdata$pred <- pred
        if (se.fit && type == "marginal") {
            newdata$se.fit <- se_fit
        }
        if (se.fit && type == "subject_specific") {
            newdata$se.fit <- se_fit
            newdata$low <- low
            newdata$upp <- upp
        }
        if (!is.null(newdata2)) {
            na_exclude2 <- attr(mfX2, "na.action")
            if (!is.null(object$gammas)) { 
                na_exclude2 <- union(attr(mfX2_zi, "na.action"), na_exclude)
            }
            if (!is.null(na_exclude))
                newdata2 <- newdata2[-na_exclude2, ]
            newdata2$pred <- pred2
            if (se.fit && type == "subject_specific") {
                newdata2$se.fit <- se_fit2
                newdata2$low <- low2
                newdata2$upp <- upp2
            }
            return(list(newdata = newdata, newdata2 = newdata2))
        } else {
            return(newdata)
        }
    } else {
        if (se.fit) {
            if (is.null(newdata2)) {
                if (type == "subject_specific") 
                    list(pred = pred, se.fit = se_fit, low = low, upp = upp,
                         success_rate = colMeans(success_rate))
                else 
                    list(pred = pred, se.fit = se_fit)
            } else {
                if (type == "subject_specific")
                    list(pred = pred, pred2 = pred2, se.fit = se_fit, se.fit2 = se_fit2,
                         low = low, upp = upp, low2 = low2, upp2 = upp2)
                else
                    list(pred = pred, pred2 = pred2, se.fit = se_fit, se.fit2 = se_fit2)
            }
        } else {
            if (is.null(newdata2)) pred else list(pred = pred, pred2 = pred2)
        }
    }
}

# simulate.MixMod
# Simulates response data from a fitted MixMod object. Supports two modes:
#   "subject_specific": simulates using the estimated random effects (empirical Bayes)
#                       conditional on the data; appropriate for posterior predictive checks
#   "mean_subject":     simulates without random effects (marginal/population-level)
#
# When new_RE = TRUE, new random effects are drawn from N(0, D) for each simulation
# instead of using the estimated ones.
# When acount_MLEs_var = TRUE, uncertainty in the fixed effects is also propagated by
# drawing betas from N(beta_hat, V_beta) (or t-distribution for penalized models).
#
# Arguments:
#   object:          a fitted MixMod object
#   nsim:            number of simulations (default: 1)
#   seed:            random seed (default: NULL, uses current RNG state)
#   type:            "subject_specific" (default) or "mean_subject"
#   new_RE:          logical; if TRUE draw new random effects for each simulation
#   acount_MLEs_var: logical; if TRUE also draw new fixed effects from their distribution
#   sim_fun:         optional custom simulation function(n, mu, phis, eta_zi); if NULL,
#                    uses the simulate() function from the family object, or falls back to
#                    a built-in default for binomial, Poisson, etc.
#   sandwich:        logical; use sandwich SE when acount_MLEs_var = TRUE
#   ...:             currently ignored
#
# Returns:
#   A data.frame with nsim columns (one per simulation), each containing simulated
#   response values for all observations.
simulate.MixMod <- function (object, nsim = 1, seed = NULL,
                             type = c("subject_specific", "mean_subject"),
                             new_RE = FALSE, acount_MLEs_var = FALSE, sim_fun = NULL,
                             sandwich = FALSE, ...) {
    if (!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) 
        runif(1)
    if (is.null(seed)) 
        RNGstate <- get(".Random.seed", envir = .GlobalEnv)
    else {
        R.seed <- get(".Random.seed", envir = .GlobalEnv)
        set.seed(seed)
        RNGstate <- structure(seed, kind = as.list(RNGkind()))
        on.exit(assign(".Random.seed", R.seed, envir = .GlobalEnv))
    }
    type <- match.arg(type)
    if (is.null(sim_fun)) {
        if (object$family$family == "binomial") {
            N <- if ((NCOL(y <- model.response(object$model_frames$mfX))) == 2) 
                y[, 1] + y[, 2] else 1
            .N <- N
            env <- new.env(parent = .GlobalEnv)
            assign(".N", N, envir = env)
            sim_fun <- function (n, mu, phis, eta_zi) {
                rbinom(n = n, size = .N, prob = mu)
            }
            environment(sim_fun) <- env
        } else if (object$family$family == "zero-inflated binomial") {
            N <- if ((NCOL(y <- model.response(object$model_frames$mfX))) == 2) 
                y[, 1] + y[, 2] else 1
            .N <- N
            env <- new.env(parent = .GlobalEnv)
            assign(".N", N, envir = env)
            sim_fun <- function (n, mu, phis, eta_zi) {
                out <- rbinom(n = n, size = .N, prob = mu)
                extra_zeros <- as.logical(rbinom(n, 1, plogis(eta_zi)))
                out[extra_zeros] <- 0
                out
            }
            environment(sim_fun) <- env
        } else if (object$family$family == "poisson") {
            sim_fun <- function (n, mu, phis, eta_zi) {
                rpois(n = n, lambda = mu)
            }
        } else if (object$family$family == "negative binomial") {
            sim_fun <- function (n, mu, phis, eta_zi) {
                rnbinom(n = n, size = exp(phis), mu = mu)
            }
        } else if (object$family$family == "zero-inflated poisson") {
            sim_fun <- function (n, mu, phis, eta_zi) {
                out <- rpois(n = n, lambda = mu)
                out[as.logical(rbinom(n, 1, plogis(eta_zi)))] <- 0
                out
            }
        } else if (object$family$family == "zero-inflated negative binomial") {
            sim_fun <- function (n, mu, phis, eta_zi) {
                out <- rnbinom(n = n, size = exp(phis), mu = mu)
                out[as.logical(rbinom(n, 1, plogis(eta_zi)))] <- 0
                out
            }
        } else if (!is.null(object$family$simulate) && is.function(object$family$simulate)) {
            sim_fun <- object$family$simulate
        } else {
            stop("'sim_fun()' needs to be specified; check the help page.")
        }
    }
    id <- object$id[[1]]
    id <- match(id, unique(id))
    n <- length(unique(id))
    X <- model.matrix(object$Terms$termsX, object$model_frames$mfX)
    Z <- mapply(constructor_Z, object$Terms$termsZ, object$model_frames$mfZ, 
                MoreArgs = list(id = id), SIMPLIFY = FALSE)
    Z <- do.call("cbind", Z)
    offset <- model.offset(object$model_frames$mfX)
    if (has_X_Zi <- !is.null(object$Terms$termsX_zi)) {
        X_zi <- model.matrix(object$Terms$termsX_zi, object$model_frames$mfX_zi)
        offset_zi <- model.offset(object$model_frames$mfX_zi)
    }
    if (has_Z_Zi <- !is.null(object$Terms$termsZ_zi)) {
        Z_zi <- mapply(constructor_Z, object$Terms$termsZ_zi, object$model_frames$mfZ_zi, 
                       MoreArgs = list (id = id), SIMPLIFY = FALSE)
        Z_zi <- do.call("cbind", Z_zi)
    }
    betas <- fixef(object)
    D <- object$D
    gammas <- object$gammas
    phis <- object$phis
    diag_D <- ncol(D) > 1 && all(abs(D[lower.tri(D)]) < sqrt(.Machine$double.eps))
    nRE <- ncol(D)
    ind <- vector("logical", nRE)
    ind[grep("zi_", colnames(D), fixed = TRUE, invert = TRUE)] <- TRUE
    if (acount_MLEs_var) {
        list_thetas <- list(betas = betas, 
                            D = if (diag_D) log(diag(D)) else chol_transf(D))
        if (!is.null(phis)) {
            list_thetas <- c(list_thetas, list(phis = phis))
        }
        if (!is.null(gammas)) {
            list_thetas <- c(list_thetas, list(gammas = gammas))
        }
        tht <- unlist(as.relistable(list_thetas))
        new_thetas <- MASS::mvrnorm(nsim, tht, vcov(object, sandwich = sandwich))
    }
    out <- matrix(0.0, nrow(X), nsim)
    for (i in seq_len(nsim)) {
        if (acount_MLEs_var) {
            new_thetas_i <- relist(new_thetas[i, ], skeleton = list_thetas)
            betas <- new_thetas_i$betas
            phis <- new_thetas_i$phis
            gammas <- new_thetas_i$gammas
            D <- if (diag_D) diag(exp(new_thetas_i$D), length(new_thetas_i$D)) 
            else chol_transf(new_thetas_i$D)
        }
        b_i <- if (new_RE) MASS::mvrnorm(n, rep(0, nRE), D) else ranef(object)
        if (type == "mean_subject")
            b_i <- b_i * 0
        eta_y <- c(X %*% betas) + rowSums(Z * b_i[id, ind, drop = FALSE])
        if (!is.null(offset))
            eta_y <- eta_y + offset
        mu <- object$Funs$mu_fun(eta_y)
        if (has_X_Zi)
            eta_zi <- c(X_zi %*% gammas)
        if (has_Z_Zi)
            eta_zi <- eta_zi + rowSums(Z_zi * b_i[id, !ind, drop = FALSE])
        if (has_X_Zi && !is.null(offset_zi))
            eta_zi <- eta_zi + offset_zi
        out[, i] <- sim_fun(nrow(X), mu, phis, eta_zi)
    }
    out
}

# model.matrix.MixMod / model.frame.MixMod / terms.MixMod / formula.MixMod
# Accessor methods for MixMod objects that extract specific model components:
#   model.matrix.MixMod: returns one of four design matrices (type = "fixed", "random",
#                        "zi_fixed", "zi_random")
#   model.frame.MixMod:  returns one of four model frames
#   terms.MixMod:        returns one of four terms objects
#   formula.MixMod:      returns one of the four formulas from the original call
# All support type = c("fixed", "random", "zi_fixed", "zi_random").
model.matrix.MixMod <- function (object, type = c("fixed", "random", "zi_fixed", "zi_random"), ...) {
    type <- match.arg(type)
    switch(type,
           "fixed" = model.matrix(object$Terms$termsX, object$model_frames$mfX),
           "random" = {
               id <- object$id[[1]]
               id <- match(id, unique(id))
               Z <- mapply(constructor_Z, object$Terms$termsZ, object$model_frames$mfZ, 
                           MoreArgs = list(id = id), SIMPLIFY = FALSE)
               do.call("cbind", Z)
               },
           "zi_fixed" = model.matrix(object$Terms$termsX_zi, object$model_frames$mfX_zi),
           "zi_random" = {
               id <- object$id[[1]]
               id <- match(id, unique(id))
               Z <- mapply(constructor_Z, object$Terms$termsZ_zi, object$model_frames$mfZ_zi, 
                           MoreArgs = list(id = id), SIMPLIFY = FALSE)
               do.call("cbind", Z)
           }
    )
}

model.frame.MixMod <- function (formula, type = c("fixed", "random", "zi_fixed", 
                                                  "zi_random"), ...) {
    type <- match.arg(type)
    switch(type, "fixed" = formula$model_frames$mfX, "random" = formula$model_frames$mfZ,
           "zi_fixed" = formula$model_frames$mfX_zi, 
           "zi_random" = formula$model_frames$mfZ_zi)
}

terms.MixMod <- function (x, type = c("fixed", "random", "zi_fixed", "zi_random"), ...) {
    type <- match.arg(type)
    switch(type, "fixed" = x$Terms$termsX, "random" = x$Terms$termsZ,
           "zi_fixed" = x$Terms$termsX_zi, "zi_random" = x$Terms$termsZ_zi)
}

formula.MixMod <- function (x, type = c("fixed", "random", "zi_fixed", "zi_random"), ...) {
    type <- match.arg(type)
    switch(type, "fixed" = eval(x$call$fixed), "random" = eval(x$call$random),
           "zi_fixed" = eval(x$call$zi_fixed), "zi_random" = eval(x$call$zi_random))
}

# family.MixMod
# Returns the family object from a fitted MixMod. Allows generic functions that dispatch
# on the family (e.g., link functions) to work with MixMod objects.
family.MixMod <- function (object, ...) {
    object$family
}

# nobs.MixMod
# Returns the number of observations. With level = 0, returns the number of
# groups/clusters; with level = 1 (default), returns the total number of observations.
# This is used by AIC() and BIC() via logLik.MixMod().
nobs.MixMod <- function (object, level = 1,...) {
    if (level == 0) {
        length(unique(object$id[[1]]))
    } else {
        length(object$id[[1]])
    }
}

# recover_data.MixMod
# emmeans integration: recovers the data and predictor information needed by emmeans to
# set up the reference grid for estimated marginal means. Dispatches to the appropriate
# terms object based on the mode argument:
#   "fixed-effects" or "marginal": uses the main fixed-effects terms
#   "zero_part":                    uses the zero-inflation terms
# This method is registered via .onLoad() when emmeans is available.
recover_data.MixMod <- function (object, mode = c("fixed-effects", "zero_part", "marginal"),
                                 ...) {
    fcall <- object$call
    mode <- match.arg(mode)
    if (mode == "fixed-effects" || mode == "marginal") {
        emmeans::recover_data(fcall, delete.response(terms(object)), object$na.action, 
                              ...)
    } else {
        emmeans::recover_data(fcall, delete.response(terms(object, type = "zi_fixed")), 
                              object$na.action, ...)
    }
}

# emm_basis.MixMod
# emmeans integration: sets up the basis for computing estimated marginal means (EMMs).
# Constructs the model matrix X for the prediction grid and returns the appropriate
# coefficients (betas or marginal_coefs()) and variance-covariance matrix.
# Three modes:
#   "fixed-effects": uses fixed effects and their inverse-Hessian variance
#   "marginal":      uses marginal_coefs() (population-averaged)
#   "zero_part":     uses zero-inflation coefficients and their variance
# Sets the link and inverse-link labels for emmeans to display properly.
# This method is registered via .onLoad() when emmeans is available.
emm_basis.MixMod <- function (object, trms, xlev, grid,
                              mode = c("fixed-effects", "zero_part", "marginal"),
                              ...) {
    mode <- match.arg(mode)
    if (mode == "fixed-effects" || mode == "marginal") {
        m <- model.frame(trms, grid, na.action = na.pass, xlev = xlev)
        X <- model.matrix(trms, m, contrasts.arg = object$contrasts)
        if (mode == "marginal") {
            mcoefs <- marginal_coefs(object, std_errors = TRUE)
            bhat <- mcoefs$betas
            V <- mcoefs$var_betas
        } else {
            bhat <- fixef(object, sub_model = "main") 
            V <- vcov(object, parm = "fixed-effects")
        }
        nbasis <- matrix(NA) 
        dfargs <- list(df = Inf)
        dffun <- function (k, dfargs) dfargs$df
    } else {
        trms_zi <- terms(object, type = "zi_fixed")
        m <- model.frame(trms_zi, grid, na.action = na.pass, xlev = xlev)
        X <- model.matrix(trms_zi, m, contrasts.arg = object$contrasts) 
        bhat <- fixef(object, sub_model = "zero_part") 
        V <- vcov(object, parm = "zero_part")
        nbasis <- matrix(NA) 
        dfargs <- list(df = Inf)
        dffun <- function (k, dfargs) dfargs$df
    }
    .std.link.labels <- function (fam, misc) {
        if (is.null(fam) || !is.list(fam)) 
            return(misc)
        if (fam$link == "identity") 
            return(misc)
        misc$tran = fam$link
        misc$inv.lbl = "response"
        if (length(grep("binomial", fam$family)) == 1) 
            misc$inv.lbl = "prob"
        else if (length(grep("poisson", fam$family)) == 1) 
            misc$inv.lbl = "rate"
        misc
    }
    misc <- .std.link.labels(object$family, list())
    list(X = X, bhat = bhat, nbasis = nbasis, V = V, dffun = dffun, 
         dfargs = dfargs, misc = misc)
}

# Effect.MixMod
# effects package integration: computes effects for a MixMod object by extracting the
# fixed effects, variance-covariance matrix, and family/link information, then delegating
# to the default Effect implementation. This method is registered via .onLoad() when
# the effects package is available.
#
# Arguments:
#   focal.predictors: character vector of predictor names to compute effects for
#   mod:              a fitted MixMod object
#   ...:              additional arguments passed to effects::Effect.default()
Effect.MixMod <- function (focal.predictors, mod, ...) {
    args <- list(call = mod$call, formula = formula(mod), family = mod$family,
                 coefficients = fixef(mod), vcov = vcov(mod, parm = "fixed-effects"))
    effects::Effect.default(focal.predictors, mod, ..., sources = args)
}

# scoring_rules
# Computes proper scoring rules to assess predictive performance of a fitted MixMod.
# Two scoring rules are computed for each observation:
#   - Log score:      log(p(y | predicted distribution))
#   - Brier score:    sum_k (P(Y=k) - I(Y=k))^2, summed over the support of Y
# These are evaluated using the predicted mean from the "mean_subject" or
# "subject_specific" predictions (depending on whether newdata2 is provided).
# The marginal predicted probabilities are computed by summing over a grid of values
# 0:max_count for count/binomial outcomes.
#
# Arguments:
#   object:          a fitted MixMod object
#   newdata:         data.frame with covariate values (and optionally the outcome)
#   newdata2:        data.frame for new subjects (uses subject_specific predictions)
#   max_count:       maximum count value to sum over for discrete distributions
#   return_newdata:  logical; if TRUE returns newdata with scoring rule columns appended
#
# Returns:
#   A data.frame with columns "log_score" and "brier_score" for each observation,
#   or the augmented newdata if return_newdata = TRUE.
scoring_rules <- function (object, newdata, newdata2 = NULL, max_count = 2000,
                           return_newdata = FALSE) {
    termsX <- object$Terms$termsX
    ND <- if (is.null(newdata2)) newdata else newdata2
    y <- model.response(model.frame(termsX, data = ND,
                                    xlev = .getXlevels(termsX, object$model_frames$mfX)))
    if (is.null(y)) {
        stop("the outcome variable is not in 'newdata' and/or 'newdata2'.")
    }
    n <- length(y)
    if (object$family$family == "binomial" && NCOL(y) == 2) {
        N <- max_count <- y[, 1] + y[, 2]
        y <- y[, 1]
    } else if (object$family$family == "binomial" && NCOL(y) == 1) {
        N <- max_count <- rep(1, n)
    } else {
        N <- NULL
    }
    max_count <- rep(max_count, length.out = n)
    prob_fun <- if (object$family$family == "binomial") {
        function (x, mean, pis, N) dbinom(x, size = N, prob = mean)
    } else if (object$family$family == "poisson") {
        function (x, mean, pis, N) dpois(x, lambda = mean)
    } else if (object$family$family == "negative binomial") {
        function (x, mean, pis, N) dnbinom(x, mu = mean, size = exp(object$phis))
    } else if (object$family$family == "zero-inflated poisson") {
        function (x, mean, pis, N) {
            ind0 <- x == 0
            out <- (1 - pis) * dpois(x, lambda = mean / (1 - pis))
            out[ind0] <- pis + out[ind0]
            out
        }
    } else if (object$family$family == "zero-inflated negative binomial") {
        function (x, mean, pis, N) {
            ind0 <- x == 0
            out <- (1 - pis) * dnbinom(x, mu = mean / (1 - pis), size = exp(object$phis))
            out[ind0] <- pis + out[ind0]
            out
        }
    } else if (object$family$family == "hurdle poisson") {
        function (x, mean, pis, N) {
            ind0 <- x == 0
            trunc_zero <- dpois(x, lambda = mean) / 
                ppois(0, lambda = mean, lower.tail = FALSE)
            out <- (1 - pis) * trunc_zero
            out[ind0] <- pis
            out
        }
    } else if (object$family$family == "hurdle negative binomial") {
        function (x, mean, pis, N) {
            ind0 <- x == 0
            trunc_zero <- dnbinom(x, mu = mean, size = exp(object$phis)) / 
                pnbinom(0, mu = mean, size = exp(object$phis), lower.tail = FALSE)
            out <- (1 - pis) * trunc_zero
            out[ind0] <- pis
            out
        }
    }
    max_count_seq <- lapply(max_count, seq, from = 0)
    pred <- predict(object, newdata = newdata, newdata2 = newdata2, 
                    type = "subject_specific")
    pred_zi <- if (!is.null(object$gammas)) attr(pred, "zi_probs")
    if (!is.null(newdata2)) {
        pred <- pred$pred2
        pred_zi <- attr(pred, "zi_probs")
    }
    logarithmic <- quadratic <- spherical <- numeric(n)
    for (i in seq_len(n)) {
        p_y <- prob_fun(y[i], mean = pred[i], pis = pred_zi[i], N[i])
        quadrat_p <- sum(prob_fun(max_count_seq[[i]], mean = pred[i], 
                                  pis = pred_zi[i], N[i])^2)
        logarithmic[i] <- log(p_y)
        quadratic[i] <- 2 * p_y - quadrat_p
        spherical[i] <- p_y / sqrt(quadrat_p)
    }
    result <- data.frame(logarithmic = logarithmic, quadratic = quadratic, 
                         spherical = spherical)
    if (return_newdata) cbind(ND, result) else result
}

# VIF (generic)
# Generic function for variance inflation factors. Dispatches to VIF.MixMod().
VIF <- function (object, ...) UseMethod("VIF")

# VIF.MixMod
# Computes generalized variance inflation factors (GVIF) for the fixed effects of a
# fitted MixMod object. Uses the Fox & Monette (1992) formula:
#   GVIF = det(R_j) * det(R_{-j}) / det(R)
# where R is the correlation matrix of the fixed-effects variance-covariance, R_j is the
# submatrix for term j's columns, and R_{-j} is the remaining submatrix.
# For multi-df terms, GVIF^(1/(2*df)) is also reported (comparable to the single-df VIF).
#
# Arguments:
#   object: a fitted MixMod object
#   type:   "fixed" (default) for main fixed effects, or "zi_fixed" for ZI fixed effects
#   ...:    currently ignored
#
# Returns:
#   A named numeric vector (if all Df = 1) or a matrix with columns GVIF, Df, GVIF^(1/(2*Df)).
VIF.MixMod <- function (object, type = c("fixed", "zi_fixed"), ...) {
    type <- match.arg(type)
    if (any(is.na(fixef(object, sub_model = if (type == "fixed") "main" else "zero_part")))) 
        stop ("there are aliased coefficients in the model.")
    v <- vcov(object, parm = if (type == "fixed") "fixed-effects" else "zero_part")
    assign <- attr(model.matrix(object, type = type), "assign")
    if (names(fixef(object)[1L]) == "(Intercept)") {
        v <- v[-1, -1]
        assign <- assign[-1]
    } else {
        warning("No intercept: VIFs may not be sensible.")
    }
    terms <- labels(terms(object, type = type))
    n.terms <- length(terms)
    if (n.terms < 2) 
        stop("model contains fewer than 2 terms")
    R <- cov2cor(v)
    detR <- det(R)
    result <- matrix(0.0, n.terms, 3)
    rownames(result) <- terms
    colnames(result) <- c("GVIF", "Df", "GVIF^(1/(2*Df))")
    for (term in seq_len(n.terms)) {
        subs <- which(assign == term)
        result[term, 1] <- det(R[subs, subs, drop = FALSE]) * 
            det(R[-subs, -subs, drop = FALSE]) / detR
        result[term, 2] <- length(subs)
    }
    if (all(result[, 2] == 1)) {
        result <- result[, 1]
    } else {
        result[, 3] <- result[, 1]^(1/(2 * result[, 2]))
    }
    result
}

# cooks.distance.MixMod
# Computes Cook's distances for each group/cluster in a fitted MixMod object. Cook's
# distance measures the influence of each group on the parameter estimates by refitting
# the model with each group excluded and measuring the change in the parameter vector:
#   D_i = (theta_hat - theta_hat(-i))' * H * (theta_hat - theta_hat(-i)) / p
# where H is the Hessian at the full-model estimates and p is the number of parameters.
# Refitting is done in parallel (using the parallel package) when cores > 1.
#
# Arguments:
#   model: a fitted MixMod object
#   cores: number of parallel cores (default: max(detected cores - 1, 1))
#   ...:   currently ignored
#
# Returns:
#   A named numeric vector of Cook's distances (one per group/cluster).
cooks.distance.MixMod <- function (model, cores = max(parallel::detectCores() - 1, 1),
                                   ...) {
    data <- model$data
    id <- data[[model$id_name]]
    unq_id <- unique(id)
    n <- length(unq_id)
    cook_distance <- function (exclude, model, data, id) {
        data_i <- data[id != exclude, ]
        model_i <- update(model, data = data_i)
        list(betas = fixef(model_i), Vbetas = vcov(model_i, parm = "fixed-effects"),
             gammas = if (!is.null(model_i$gammas)) fixef(model_i, "zero_part"), 
             Vgammas = if (!is.null(model_i$gammas)) vcov(model_i, parm = "zero_part"))
    }
    cl <- parallel::makeCluster(cores)
    res <- parallel::parLapply(cl, unq_id, cook_distance, model = model, 
                               data = data, id = id)
    parallel::stopCluster(cl)
    betas <- fixef(model)
    betas_i <- do.call(rbind, lapply(res, "[[", "betas"))
    ss <- rep(betas, each = nrow(betas_i)) - betas_i
    invCov <- solve(vcov(model, parm = "fixed-effects"))
    CooksD_betas <- rowSums((ss %*% invCov) * ss)
    if (!is.null(model$gammas)) {
        gammas <- fixef(model, "zero_part")
        gammas_i <- do.call(rbind, lapply(res, "[[", "gammas"))
        ss <- rep(gammas, each = nrow(gammas_i)) - gammas_i
        invCov <- solve(vcov(model, parm = "zero_part"))
        CooksD_gammas <- rowSums((ss %*% invCov) * ss)
        list(betas = CooksD_betas, gammas = CooksD_gammas)
    } else {
        CooksD_betas
    }
}



