# find_modes
# Finds the posterior modes (empirical Bayes estimates) of the random effects for each
# subject/group, using BFGS optimization of the negative log-posterior. The posterior is
# proportional to p(y_i | b_i) * p(b_i), where p(b_i) is the normal prior with covariance D.
# The Hessian at each mode is also computed, providing the local curvature needed for the
# adaptive Gauss-Hermite quadrature.
#
# Arguments:
#   b:               n x nRE matrix of starting values for the random effects (one row per group)
#   y_lis:           list of response vectors/matrices, one element per group
#   N_lis:           list of binomial totals (NULL if not applicable)
#   X_lis:           list of fixed-effects design matrices, one per group
#   Z_lis:           list of random-effects design matrices, one per group
#   offset_lis:      list of offset vectors for the main part (NULL if no offset)
#   X_zi_lis:        list of zero-inflation fixed-effects design matrices (NULL if not ZI)
#   Z_zi_lis:        list of zero-inflation random-effects design matrices (NULL if not ZI)
#   offset_zi_lis:   list of zero-inflation offset vectors (NULL if not applicable)
#   betas:           numeric vector of fixed-effects coefficients
#   invD:            inverse of the random-effects covariance matrix D
#   phis:            numeric vector of dispersion/shape parameters (NULL if not applicable)
#   gammas:          numeric vector of zero-part fixed effects (NULL if not ZI)
#   canonical:       logical; TRUE if the model uses a canonical link (binomial/logit, poisson/log)
#   user_defined:    logical; TRUE if a user-defined family is used
#   Zty_lis:         list of Z'y cross-products per group (used for canonical link score)
#   log_dens:        function computing log-density of the response
#   mu_fun:          inverse-link function (linkinv)
#   var_fun:         variance function of the family
#   mu.eta_fun:      derivative of the mean w.r.t. the linear predictor
#   score_eta_fun:   analytic score w.r.t. eta (NULL triggers numerical differentiation)
#   score_phis_fun:  analytic score w.r.t. phis (NULL triggers numerical differentiation)
#   score_eta_zi_fun: analytic score w.r.t. zi linear predictor (NULL triggers numerical diff)
#
# Returns:
#   A list with:
#     post_modes:    n x nRE matrix of posterior mode estimates for each group
#     post_hessians: list of n Hessian matrices (nRE x nRE) at the posterior mode for each group
find_modes <- function (b, y_lis, N_lis, X_lis, Z_lis, offset_lis, X_zi_lis, Z_zi_lis,
                        offset_zi_lis, betas, invD, phis, gammas,
                        canonical, user_defined, Zty_lis, log_dens, mu_fun, var_fun,
                        mu.eta_fun, score_eta_fun, score_phis_fun, score_eta_zi_fun) {
    log_post_b <- function (b_i, y_i, N_i, X_i, Z_i, offset_i, X_zi_i, Z_zi_i, offset_zi_i, 
                            betas, invD, phis, gammas, canonical,
                            user_defined, Zty_i, log_dens, mu_fun, var_fun, mu.eta_fun,
                            score_eta_fun, score_phis_fun, score_eta_zi_fun) {
        ind_Z <- seq_len(ncol(Z_i))
        eta_y <- as.vector(X_i %*% betas + Z_i %*% b_i[ind_Z])
        if (!is.null(offset_i))
            eta_y <- eta_y + offset_i
        eta_zi <- if (!is.null(X_zi_i)) as.vector(X_zi_i %*% gammas)
        if (!is.null(Z_zi_i))
            eta_zi <- eta_zi + as.vector(Z_zi_i %*% b_i[-ind_Z])
        if (!is.null(offset_zi_i))
            eta_zi <- eta_zi + offset_zi_i
        - sum(log_dens(y_i, eta_y, mu_fun, phis, eta_zi), na.rm = TRUE) +
                c(0.5 * crossprod(b_i, invD) %*% b_i)
    }
    score_log_post_b <- function (b_i, y_i, N_i, X_i, Z_i, offset_i,X_zi_i, Z_zi_i, offset_zi_i,
                                  betas, invD, phis, gammas,
                                  canonical, user_defined, Zty_i, log_dens, mu_fun,
                                  var_fun, mu.eta_fun, score_eta_fun, score_phis_fun,
                                  score_eta_zi_fun) {
        eta_y <- as.vector(X_i %*% betas + Z_i %*% b_i[seq_len(ncol(Z_i))])
        if (!is.null(offset_i))
            eta_y <- eta_y + offset_i
        eta_zi <- if (!is.null(X_zi_i)) as.vector(X_zi_i %*% gammas)
        if (!is.null(Z_zi_i))
            eta_zi <- eta_zi + as.vector(Z_zi_i %*% b_i[-seq_len(ncol(Z_i))])
        if (!is.null(offset_zi_i))
            eta_zi <- eta_zi + offset_zi_i
        mu_y <- mu_fun(eta_y)
        log_dens_part <- if (user_defined) {
            out <- if (!is.null(score_eta_fun)) {
                - crossprod(Z_i, score_eta_fun(y_i, mu_y, phis, eta_zi))
            } else {
                l1 <- log_dens(y_i, eta_y + 1e-04, mu_fun, phis, eta_zi)
                l2 <- log_dens(y_i, eta_y - 1e-04, mu_fun, phis, eta_zi)
                - crossprod(Z_i, (l1 - l2) / (2 * 1e-04))
            }
            if (!is.null(Z_zi_i)) {
                out <- if (!is.null(score_eta_zi_fun)) {
                    c(out, - crossprod(Z_zi_i, score_eta_zi_fun(y_i, mu_y, phis, eta_zi)))
                } else {
                    l1 <- log_dens(y_i, eta_y, mu_fun, phis, eta_zi + 1e-04)
                    l2 <- log_dens(y_i, eta_y, mu_fun, phis, eta_zi - 1e-04)
                    c(out, - crossprod(Z_zi_i, (l1 - l2) / (2 * 1e-04)))
                }
            }
            out
        } else {
            if (canonical) {
                if (!is.null(N_i))- Zty_i + crossprod(Z_i, N_i * mu_y) else 
                    - Zty_i + crossprod(Z_i, mu_y)
            } else {
                var <- var_fun(mu_y)
                deriv <- mu.eta_fun(eta_y)
                if (!is.null(N_i)) - crossprod(Z_i, (y_i[, 1] - N_i * mu_y) * deriv / var) else
                - crossprod(Z_i, (y_i - mu_y) * deriv / var)
            }
        }
        drop(log_dens_part + invD %*% b_i)
    }
    n <- length(y_lis)
    post_modes <- b
    post_hessians <- vector("list", n)
    for (i in seq_len(n)) {
        y_i <- y_lis[[i]]
        N_i <- if (!is.null(N_lis)) N_lis[[i]]
        X_i <- X_lis[[i]]
        Z_i <- Z_lis[[i]]
        offset_i <- if (!is.null(offset_lis)) offset_lis[[i]]
        Zty_i <- Zty_lis[[i]]
        X_zi_i <- if (!is.null(X_zi_lis)) X_zi_lis[[i]]
        Z_zi_i <- if (!is.null(Z_zi_lis)) Z_zi_lis[[i]]
        offset_zi_i <- if (!is.null(offset_zi_lis)) offset_zi_lis[[i]]
        b_i <- b[i, ]
        opt_i <- optim(par = b_i, fn = log_post_b, gr = score_log_post_b, method = "BFGS",
                       y_i = y_i, N_i = N_i, X_i = X_i, Z_i = Z_i, offset_i = offset_i,
                       X_zi_i = X_zi_i, Z_zi_i = Z_zi_i, offset_zi_i = offset_zi_i,
                       betas = betas, invD = invD, phis = phis, gammas = gammas, 
                       canonical = canonical,
                       user_defined = user_defined, Zty_i = Zty_i, log_dens = log_dens,
                       mu_fun = mu_fun, var_fun = var_fun, mu.eta_fun = mu.eta_fun,
                       score_eta_fun = score_eta_fun, score_phis_fun = score_phis_fun,
                       score_eta_zi_fun = score_eta_zi_fun)
        post_modes[i, ] <- opt_i$par
        post_hessians[[i]] <- cd_vec(post_modes[i, ], score_log_post_b,
                                     y_i = y_i, N_i = N_i, X_i = X_i, Z_i = Z_i, 
                                     offset_i = offset_i, X_zi_i = X_zi_i, Z_zi_i = Z_zi_i, 
                                     offset_zi_i = offset_zi_i, betas = betas, invD = invD, 
                                     phis = phis, gammas = gammas, canonical = canonical, 
                                     user_defined = user_defined,
                                     Zty_i = Zty_i, log_dens = log_dens, mu_fun = mu_fun,
                                     var_fun = var_fun, mu.eta_fun = mu.eta_fun,
                                     score_eta_fun = score_eta_fun,
                                     score_phis_fun = score_phis_fun, 
                                     score_eta_zi_fun = score_eta_zi_fun)
    }
    list(post_modes = post_modes, post_hessians = post_hessians)
}

# GHfun
# Sets up the adaptive Gauss-Hermite (AGH) quadrature by:
#   1. Computing standard GH nodes/weights via gauher().
#   2. Finding the posterior mode and Hessian for each group via find_modes().
#   3. Transforming the standard GH nodes to be centered at the posterior mode and scaled
#      by the inverse Cholesky of the Hessian (i.e., the Laplace approximation to the
#      posterior covariance). This "adaptation" concentrates quadrature points near the
#      region of high posterior mass.
#   4. Computing the corrected quadrature weights (accounting for the change of variables).
#   5. Pre-computing Z*b and Z_zi*b for efficiency inside the EM loop.
#
# Arguments:
#   b:               current n x nRE matrix of random-effect values (used as starting values)
#   y_lis, N_lis, X_lis, Z_lis, offset_lis,
#   X_zi_lis, Z_zi_lis, offset_zi_lis: per-group data lists (see find_modes)
#   betas:           current fixed-effects coefficients
#   inv_D:           inverse of the random-effects covariance matrix
#   phis:            current dispersion parameters (NULL if not applicable)
#   gammas:          current zero-part fixed effects (NULL if not ZI)
#   k:               number of quadrature points per dimension (nAGQ)
#   q:               total number of random effects (nRE)
#   canonical, user_defined, Zty_lis, log_dens, mu_fun, var_fun,
#   mu.eta_fun, score_eta_fun, score_phis_fun, score_eta_zi_fun: passed to find_modes
#
# Returns:
#   A list with:
#     b:          (n * k^q) x q matrix of adapted quadrature nodes stacked across groups
#     b2:         outer-product form of b (for computing E[b b'] in the EM M-step for D)
#     Ztb:        Z * b^T pre-computed across quadrature points for efficiency
#     Z_zitb:     Z_zi * b^T (NULL if no ZI random effects)
#     wGH:        vector of k^q corrected quadrature weights (common across groups after adaptation)
#     log_dets:   log-determinant corrections from the Cholesky of the Hessian (one per group)
#     post_modes: n x nRE matrix of posterior modes (passed through from find_modes)
#     post_vars:  list of posterior variance matrices (inverse Hessians, one per group)
GHfun <- function (b, y_lis, N_lis, X_lis, Z_lis, offset_lis, X_zi_lis, Z_zi_lis, offset_zi_lis,
                   betas, inv_D, phis, gammas, k, q,
                   canonical, user_defined, Zty_lis, log_dens, mu_fun, var_fun, mu.eta_fun,
                   score_eta_fun, score_phis_fun, score_eta_zi_fun) {
    GH <- gauher(k)
    aGH <- find_modes(b, y_lis, N_lis, X_lis, Z_lis, offset_lis, X_zi_lis, Z_zi_lis, 
                      offset_zi_lis, betas, inv_D, phis, gammas,
                      canonical, user_defined, Zty_lis, log_dens, mu_fun, var_fun, 
                      mu.eta_fun, score_eta_fun, score_phis_fun, score_eta_zi_fun)
    modes <- aGH$post_modes
    chol_hessians <- lapply(aGH$post_hessian, chol)
    b <- as.matrix(expand.grid(lapply(seq_len(q), function (k, u) u$x, u = GH)))
    n <- nrow(modes)
    b_new <- vector("list", n)
    log_dets <- numeric(n)
    for (i in seq_len(n)) {
        b_new[[i]] <- t(sqrt(2) * solve(chol_hessians[[i]], t(b)) + modes[i, ])
        log_dets[i] <- - determinant.matrix(chol_hessians[[i]], logarithm = TRUE)$modulus
    }
    wGH <- as.matrix(expand.grid(lapply(seq_len(q), function (k, u) u$w, u = GH)))
    wGH <- 2^(q/2) * apply(wGH, 1, prod) * exp(rowSums(b * b))
    b2 <- lapply(b_new, function (b) if (q == 1) b * b else
        t(apply(b, 1, function (x) x %o% x)))
    ind_Z <- seq_len(ncol(Z_lis[[1]]))
    Ztb <- do.call('rbind', mapply(function (z, b) z %*% t(b[, ind_Z, drop = FALSE]), 
                                   Z_lis, b_new, SIMPLIFY = FALSE))
    Z_zitb <- if (!is.null(Z_zi_lis[[1]])) {
        do.call('rbind', mapply(function (z, b) z %*% t(b[, -ind_Z, drop = FALSE]), 
                                Z_zi_lis, b_new, SIMPLIFY = FALSE))  
    } 
    list(b = do.call('rbind', b_new), b2 = do.call('rbind', b2), Ztb = Ztb, Z_zitb = Z_zitb,
               wGH = wGH, log_dets = log_dets, post_modes = modes, 
         post_vars = lapply(aGH$post_hessian, solve))
}

# chol_transf
# Converts between an unconstrained parameter vector and a positive-definite covariance
# matrix using the Cholesky parameterization. This ensures D stays positive-definite
# during optimization.
#
# When x is a matrix (i.e., a PD matrix D):
#   - Computes the upper-triangular Cholesky factor U such that D = U'U.
#   - Log-transforms the diagonal of U (so diagonal entries are unconstrained real numbers).
#   - Returns the upper-triangle of U (including the transformed diagonal) as a vector.
#
# When x is a vector (i.e., the unconstrained parameter vector):
#   - Reconstructs the upper-triangular U by filling in the upper-triangle.
#   - Exponentiates the diagonal elements to enforce positivity.
#   - Returns D = U'U (positive-definite) and attaches the lower-triangle of U as an
#     attribute "L" (used for the Jacobian computation in score_mixed).
#
# Arguments:
#   x: either a square positive-definite matrix (to be vectorized) or a numeric vector
#      (to be reconstructed as a PD matrix)
#
# Returns:
#   When x is a matrix: a numeric vector of the upper-triangle of the log-Cholesky factor.
#   When x is a vector: the reconstructed PD covariance matrix with attribute "L".
chol_transf <- function (x) {
    if (any(is.na(x) | !is.finite(x)))
        stop("NA or infinite values in 'x'.\n")
    if (is.matrix(x)) {
        k <- nrow(x)
        U <- chol(x)
        U[cbind(1:k, 1:k)] <- log(U[cbind(1:k, 1:k)])
        U[upper.tri(U, TRUE)]
    } else {
        nx <- length(x)
        k <- round((-1 + sqrt(1 + 8 * nx))/2)
        mat <- matrix(0, k, k)
        mat[upper.tri(mat, TRUE)] <- x
        mat[cbind(1:k, 1:k)] <- exp(mat[cbind(1:k, 1:k)])
        res <- crossprod(mat)
        attr(res, "L") <- t(mat)[lower.tri(mat, TRUE)]
        res
    }
}

# deriv_D
# Computes the partial derivatives of D with respect to each element of its lower triangle.
# These are used in score_mixed() to compute the score with respect to the unconstrained
# Cholesky parameters via the chain rule (i.e., dlogL/d(Chol params) = dlogL/dD * dD/d(Chol params)).
#
# The derivative of D w.r.t. the (i,j) lower-triangle element is a matrix that is 1 at
# positions (i,j) and (j,i), and 0 elsewhere (since D is symmetric).
#
# Arguments:
#   D: a square covariance matrix of the random effects
#
# Returns:
#   A list of (ncz*(ncz+1)/2) matrices, each of size ncz x ncz, representing
#   dD/d(theta_k) for each unconstrained parameter theta_k in the lower triangle of D.
deriv_D <- function (D) {
    ncz <- nrow(D)
    ind <- which(lower.tri(D, TRUE), arr.ind = TRUE)
    dimnames(ind) <- NULL
    nind <- nrow(ind)
    svD <- solve(D)
    lapply(seq_len(nind), function (x, ind) {
        mat <- matrix(0, ncz, ncz)
        ii <- ind[x, , drop = FALSE]
        mat[ii[1], ii[2]] <- mat[ii[2], ii[1]] <- 1
        mat
    }, ind = ind[, 2:1, drop = FALSE])
}

# jacobian2
# Computes the Jacobian matrix of the transformation from the unconstrained log-Cholesky
# parameters (the upper-triangle of U with log-diagonal, stored as a vector) back to the
# lower-triangle of D = U'U.
#
# This Jacobian is needed to transform the score with respect to D's lower-triangle elements
# into the score with respect to the unconstrained optimization parameters. Specifically,
# if theta is the unconstrained parameter vector (log-Cholesky), then:
#   dlogL/d(theta) = dlogL/d(vech(D)) * J
# where J = jacobian2(L, ncz), and vech() extracts the lower-triangle.
#
# Arguments:
#   L:   the lower-triangle of the Cholesky factor U (as a vector, including diagonal)
#   ncz: the dimension of the covariance matrix D (number of random effects)
#
# Returns:
#   A (ncz*(ncz+1)/2) x (ncz*(ncz+1)/2) Jacobian matrix.
jacobian2 <- function (L, ncz) {
    ind <- which(lower.tri(matrix(0, ncz, ncz), TRUE), arr.ind = TRUE)
    dimnames(ind) <- NULL
    nind <- nrow(ind)
    id <- 1:nind
    rind <- which(ind[, 1] == ind[, 2])
    lind <- vector("list", length(rind))
    for (i in seq_along(rind)) {
        tt <- matrix(0, ncz - i + 1, ncz - i + 1)
        tt[lower.tri(tt, TRUE)] <- seq(rind[i], nind)
        tt <- tt + t(tt)
        diag(tt) <- diag(tt)/2
        lind[[i]] <- tt
    }
    out <- matrix(0, nind, nind)
    for (g in 1:ncz) {
        gind <- id[g == ind[, 2]]
        vals <- L[gind]
        for (j in gind) {
            k <- which(j == gind)
            out[cbind(lind[[g]][k, ], j)] <- if (j %in% rind) vals[1] * vals else vals
        }
    }
    out[rind, ] <- 2 * out[rind, ]
    col.ind <- matrix(0, ncz, ncz)
    col.ind[lower.tri(col.ind, TRUE)] <- seq(1, length(L))
    col.ind <- t(col.ind)
    out[, col.ind[upper.tri(col.ind, TRUE)]]
}

# fd
# Computes the gradient of a scalar-valued function f at point x using forward differences.
# For each component i of x, the partial derivative is approximated as:
#   (f(x + eps_i * e_i) - f(x)) / eps_i
# where eps_i is a step size scaled to the magnitude of x[i].
#
# Arguments:
#   x:   numeric vector at which to evaluate the gradient
#   f:   scalar function to differentiate (may accept additional arguments via ...)
#   ...: additional arguments passed to f
#   eps: base step size (default: machine epsilon to the 1/4 power)
#
# Returns:
#   A numeric vector of length n = length(x) containing the gradient approximation.
fd <- function (x, f, ..., eps = .Machine$double.eps^0.25) {
    n <- length(x)
    res <- numeric(n)
    ex <- eps * (abs(x) + eps)
    f0 <- f(x, ...)
    for (i in seq_len(n)) {
        x1 <- x
        x1[i] <- x[i] + ex[i]
        diff.f <- c(f(x1, ...) - f0)
        diff.x <- x1[i] - x[i]
        res[i] <- diff.f / diff.x
    }
    res
}

# fd_vec
# Approximates the Hessian (second-derivative matrix) of a scalar function f using forward
# differences applied to the gradient (i.e., Jacobian of the gradient vector). For a vector-
# valued gradient function f, each column j of the result approximates df/dx_j.
# The result is symmetrized by averaging the matrix with its transpose.
#
# This is used to compute the Hessian of the score functions (score_betas, score_phis,
# score_gammas) during the EM Newton-Raphson updates.
#
# Arguments:
#   x:   numeric vector at which to evaluate the Hessian
#   f:   vector-valued function (typically a score/gradient) to differentiate
#   ...: additional arguments passed to f
#   eps: base step size (default: machine epsilon to the 1/4 power)
#
# Returns:
#   A symmetric n x n matrix approximating the Hessian of the original objective,
#   or equivalently the Jacobian of f (symmetrized).
fd_vec <- function (x, f, ..., eps = .Machine$double.eps^0.25) {
    n <- length(x)
    res <- matrix(0, n, n)
    ex <- pmax(abs(x), 1)
    f0 <- f(x, ...)
    for (i in 1:n) {
        x1 <- x
        x1[i] <- x[i] + eps * ex[i]
        diff.f <- c(f(x1, ...) - f0)
        diff.x <- x1[i] - x[i]
        res[, i] <- diff.f / diff.x
    }
    0.5 * (res + t(res))
}

# cd
# Computes the gradient of a scalar function f at point x using central differences.
# Central differences are more accurate than forward differences (O(eps^2) vs O(eps)):
#   (f(x + eps_i * e_i) - f(x - eps_i * e_i)) / (2 * eps_i)
# The step size eps_i is scaled to max(|x[i]|, 1) to handle parameters near zero.
#
# Arguments:
#   x:   numeric vector at which to evaluate the gradient
#   f:   scalar function to differentiate
#   ...: additional arguments passed to f
#   eps: relative step size (default: 0.001)
#
# Returns:
#   A numeric vector of length n = length(x) containing the gradient approximation.
cd <- function (x, f, ..., eps = 0.001) {
    n <- length(x)
    res <- numeric(n)
    ex <- pmax(abs(x), 1)
    for (i in seq_len(n)) {
        x1 <- x2 <- x
        x1[i] <- x[i] + eps * ex[i]
        x2[i] <- x[i] - eps * ex[i]
        diff.f <- c(f(x1, ...) - f(x2, ...))
        diff.x <- x1[i] - x2[i]
        res[i] <- diff.f / diff.x
    }
    res
}

# cd_vec
# Approximates the Hessian of a scalar function (or the Jacobian of a vector-valued
# function) using central differences. Each column j approximates df(x)/dx_j via:
#   (f(x + eps_j * e_j) - f(x - eps_j * e_j)) / (2 * eps_j)
# The result is symmetrized by averaging the matrix with its transpose.
#
# Used for computing the Hessian of the log-likelihood (via score_mixed) at convergence.
#
# Arguments:
#   x:   numeric vector at which to evaluate the Hessian
#   f:   vector-valued function (typically score_mixed) to differentiate
#   ...: additional arguments passed to f
#   eps: relative step size (default: 0.001)
#
# Returns:
#   A symmetric n x n matrix approximating the Hessian (Jacobian of f, symmetrized).
cd_vec <- function (x, f, ..., eps = 0.001) {
    n <- length(x)
    res <- matrix(0, n, n)
    ex <- pmax(abs(x), 1)
    for (i in seq_len(n)) {
        x1 <- x2 <- x
        x1[i] <- x[i] + eps * ex[i]
        x2[i] <- x[i] - eps * ex[i]
        diff.f <- c(f(x1, ...) - f(x2, ...))
        diff.x <- x1[i] - x2[i]
        res[, i] <- diff.f / diff.x
    }
    0.5 * (res + t(res))
}

# dmvnorm
# Evaluates the multivariate normal density N(mu, Sigma) at each row of x.
# Handles three special cases efficiently:
#   - Univariate (p == 1): delegates to dnorm().
#   - Diagonal Sigma (or Sigma given as a vector of variances): uses independent normals.
#   - General Sigma: uses the eigendecomposition for the log-determinant and inverse.
#
# Arguments:
#   x:     numeric matrix (each row is an observation) or vector (treated as single row)
#   mu:    numeric vector of length p (mean)
#   Sigma: p x p positive-definite covariance matrix, OR a length-p vector of variances
#          (interpreted as a diagonal covariance)
#   log:   logical; if TRUE returns the log-density (default: FALSE)
#
# Returns:
#   A numeric vector of length nrow(x) with densities (or log-densities if log = TRUE).
dmvnorm <- function (x, mu, Sigma, log = FALSE)  {
    if (!is.matrix(x))
        x <- rbind(x)
    p <- length(mu)
    if (p == 1) {
        dnorm(x, mu, sqrt(Sigma), log = log)
    } else {
        t1 <- length(mu) == length(Sigma)
        t2 <- all(abs(Sigma[lower.tri(Sigma)]) < sqrt(.Machine$double.eps))
        if (t1 || t2) {
            if (!t1)
                Sigma <- diag(Sigma)
            nx <- nrow(x)
            ff <- rowSums(dnorm(x, rep(mu, each = nx),
                                sd = rep(sqrt(Sigma), each = nx), log = TRUE))
            if (log) ff else exp(ff)
        } else {
            ed <- eigen(Sigma, symmetric = TRUE)
            ev <- ed$values
            evec <- ed$vectors
            if (!all(ev >= -1e-06 * abs(ev[1])))
                stop("'Sigma' is not positive definite")
            ss <- x - rep(mu, each = nrow(x))
            inv.Sigma <- evec %*% (t(evec)/ev)
            quad <- 0.5 * rowSums((ss %*% inv.Sigma) * ss)
            fact <- - 0.5 * (p * log(2 * pi) + sum(log(ev)))
            if (log) as.vector(fact - quad) else as.vector(exp(fact - quad))
        }
    }
}

# unattr
# Strips all attributes from an object (e.g., names, dimnames, class) while preserving
# its dimensions if it is a matrix. This is used to clean up design matrices and response
# vectors before the core fitting computations, avoiding unexpected behavior from
# lingering attributes (e.g., "assign", "contrasts") that could interfere with operations.
#
# Arguments:
#   x: any R object (typically a matrix or vector)
#
# Returns:
#   x with all attributes removed; if x was a matrix, its dim attribute is restored.
unattr <- function (x) {
    if (is_mat <- is.matrix(x)) {
        d <- dim(x)
    }
    attributes(x) <- NULL
    if (is_mat) {
        dim(x) <- d
    }
    x
}

# gauher
# Computes the nodes (abscissas) and weights for Gauss-Hermite quadrature of order n.
# These are used to approximate integrals of the form integral(f(x) * exp(-x^2) dx).
# The nodes are the roots of the n-th Hermite polynomial H_n(x), computed iteratively
# using Newton-Raphson with initial approximations from Abramowitz & Stegun (1972).
# Exploits the symmetry of the Hermite polynomial: only the n/2 positive roots need to
# be found, and their negatives give the remaining nodes.
#
# Arguments:
#   n: the number of quadrature points (positive integer)
#
# Returns:
#   A list with:
#     x: numeric vector of n nodes (in ascending order)
#     w: numeric vector of n weights corresponding to the nodes
gauher <- function (n) {
    m <- trunc((n + 1) / 2)
    x <- w <- rep(-1, n)
    for (i in seq_len(m)) {
        z <- if (i == 1) {
            sqrt(2 * n + 1) - 1.85575 * (2 * n + 1)^(-0.16667)
        } else if (i == 2) {
            z - 1.14 * n^0.426/z
        } else if (i == 3) {
            1.86 * z - 0.86 * x[1]
        } else if (i == 4) {
            1.91 * z - 0.91 * x[2]
        } else {
            2 * z - x[i - 2]
        }
        for (its in seq_len(10)) {
            p1 <- 0.751125544464943
            p2 <- 0
            for (j in seq_len(n)) {
                p3 <- p2
                p2 <- p1
                p1 <- z * sqrt(2 / j) * p2 - sqrt((j - 1) / j) * p3
            }
            pp <- sqrt(2 * n) * p2
            z1 <- z
            z <- z1 - p1/pp
            if (abs(z - z1) <= 3e-14)
                break
        }
        x[i] <- z
        x[n + 1 - i] <- -z
        w[i] <- 2 / (pp * pp)
        w[n + 1 - i] <- w[i]
    }
    list(x = x, w = w)
}

# nearPD
# Finds the nearest positive-definite (PD) matrix to a given symmetric matrix M, using
# the algorithm of Higham (2002). This is used to ensure that approximate Hessian matrices
# (computed via finite differences) remain positive-definite so that Newton-Raphson updates
# yield descent directions.
#
# The algorithm iterates:
#   1. Project the current estimate onto the cone of symmetric PD matrices (eigenvalue clipping).
#   2. Apply a Dykstra correction (U) to maintain closeness to M.
# Until convergence (change in infinity-norm < conv.tol) or maxits is reached.
# A final eigenvalue-rescaling step ensures strict positive-definiteness with minimum
# eigenvalue >= posd.tol * |lambda_max|.
#
# Arguments:
#   M:        a square symmetric numeric matrix (typically an approximate Hessian)
#   eig.tol:  eigenvalues below eig.tol * lambda_max are set to zero in the projection step
#   conv.tol: convergence criterion on the relative change in infinity-norm (default: 1e-07)
#   posd.tol: minimum eigenvalue as a fraction of the largest eigenvalue (default: 1e-08)
#   maxits:   maximum number of iterations (default: 100)
#
# Returns:
#   A symmetric positive-definite matrix near M.
nearPD <- function (M, eig.tol = 1e-06, conv.tol = 1e-07, posd.tol = 1e-08,
                    maxits = 100) {
    if (!(is.numeric(M) && is.matrix(M) && identical(M, t(M))))
        stop("Input matrix M must be square and symmetric.\n")
    inorm <- function(x) max(rowSums(abs(x)))
    n <- ncol(M)
    U <- matrix(0.0, n, n)
    X <- M
    iter <- 0
    converged <- FALSE
    while (iter < maxits && !converged) {
        Y <- X
        T <- Y - U
        e <- eigen(Y, symmetric = TRUE)
        Q <- e$vectors
        d <- e$values
        D <- if (length(d) > 1) diag(d) else as.matrix(d)
        p <- (d > eig.tol * d[1])
        QQ <- Q[, p, drop = FALSE]
        X <- QQ %*% D[p, p, drop = FALSE] %*% t(QQ)
        U <- X - T
        X <- (X + t(X)) / 2
        conv <- inorm(Y - X)/inorm(Y)
        iter <- iter + 1
        converged <- conv <= conv.tol
    }
    X <- (X + t(X)) / 2
    e <- eigen(X, symmetric = TRUE)
    d <- e$values
    Eps <- posd.tol * abs(d[1L])
    if (d[n] < Eps) {
        d[d < Eps] <- Eps
        Q <- e$vectors
        o.diag <- diag(X)
        X <- Q %*% (d * t(Q))
        D <- sqrt(pmax(Eps, o.diag) / diag(X))
        X[] <- D * X * rep(D, each = n)
    }
    (X + t(X)) / 2
}

# getRE_Formula
# Extracts the random-effects structure from a random-effects formula of the form
# ~ <re_terms> | <grouping_factor> or ~ <re_terms> || <grouping_factor>.
# Returns a one-sided formula containing only the random-effects terms (without the
# grouping factor), e.g., for ~ time | id it returns ~time.
# Used to build model matrices for the random-effects design matrix Z.
#
# Arguments:
#   form: a formula of the form ~ <terms> | <group> or ~ <terms> || <group>
#
# Returns:
#   A one-sided formula containing only the random-effects part (e.g., ~time).
getRE_Formula <- function (form) {
    if (!(inherits(form, "formula"))) {
        stop("formula(object) must return a formula")
    }
    form <- form[[length(form)]]
    if (length(form) == 3 && (form[[1]] == as.name("|") || form[[1]] == as.name("||"))) {
        form <- form[[2]]
    }
    eval(substitute(~form))
}

# getID_Formula
# Extracts the grouping factor(s) from a random-effects formula or a named list.
# For a formula of the form ~ <terms> | <group>, returns a one-sided formula ~<group>.
# For a list (nested grouping), returns a formula combining the nested group names
# as ~name1/name2.
# Used to identify which column in the data represents the subject/cluster identifier.
#
# Arguments:
#   form: either a formula ~ <terms> | <group>, or a named list for nested grouping
#
# Returns:
#   A one-sided formula identifying the grouping variable(s).
getID_Formula <- function (form) {
    if (is.list(form)) {
        nams <- names(form)
        as.formula(paste0("~", nams[1L], "/", nams[2L]))
    } else {
        form <- form[[length(form)]]
        asOneSidedFormula(form[[3]])
    }
}

# printCall
# Formats a function call object as a character string for display. If the deparsed call
# has more than 3 lines, only the first 3 lines are shown with "..." appended to indicate
# truncation. Used by print.MixMod() and print.summary.MixMod() to display the model call.
#
# Arguments:
#   call: a call object (typically object$call from a fitted model)
#
# Returns:
#   A single character string with newlines, showing up to 3 lines of the deparsed call.
printCall <- function (call) {
    d <- deparse(call)
    if (length(d) <= 3) {
        paste(d, sep = "\n", collapse = "\n")
    } else {
        d <- d[1:3]
        d[3] <- paste0(d[3], "...")
        paste(d, sep = "\n", collapse = "\n")
    }
}

# dgt
# Evaluates the density of a generalized (location-scale) t distribution with location mu,
# scale sigma, and df degrees of freedom. The density is given by:
#   f(x) = (1/sigma) * dt((x - mu) / sigma, df)
# This is used as a penalty/prior for the fixed effects when penalized = TRUE.
#
# Arguments:
#   x:   numeric vector of quantiles
#   mu:  location parameter (default: 0)
#   sigma: scale parameter (default: 1)
#   df:  degrees of freedom (required; no default)
#   log: logical; if TRUE returns the log-density (default: FALSE)
#
# Returns:
#   Density (or log-density) values at x.
dgt <- function (x, mu = 0, sigma = 1, df = stop("no df argument."), log = FALSE) {
    if (log) {
        dt(x = (x - mu) / sigma, df = df, log = TRUE) - log(sigma)
    } else {
        dt(x = (x - mu) / sigma, df = df) / sigma
    }
}

# dmvt
# Evaluates the multivariate t density with location mu, scale matrix Sigma (or its
# inverse invSigma), and df degrees of freedom. Supports both the full density and the
# proportional version (prop = TRUE omits the normalizing constant). Can accept Sigma as
# either a matrix or a list with $values (eigenvalues) and $vectors (eigenvectors) for
# efficiency when reusing the eigendecomposition.
#
# Used as the penalty/prior log-density for fixed effects when penalized = TRUE.
#
# Arguments:
#   x:         numeric vector or matrix (each row is an observation)
#   mu:        location vector of length p
#   Sigma:     p x p positive-definite scale matrix (provide either Sigma or invSigma)
#   invSigma:  inverse of Sigma (more efficient when Sigma^-1 is already available)
#   df:        degrees of freedom
#   log:       logical; if TRUE returns log-density (default: TRUE)
#   prop:      logical; if TRUE omits the normalizing constant (default: TRUE)
#
# Returns:
#   A numeric vector of densities (or log-densities) at each row of x.
dmvt <- function (x, mu, Sigma = NULL, invSigma = NULL, df, log = TRUE, prop = TRUE) {
    if (!is.numeric(x)) 
        stop("'x' must be a numeric matrix or vector")
    if (!is.matrix(x)) 
        x <- rbind(x)
    p <- length(mu)
    if (is.null(Sigma) && is.null(invSigma))
        stop("'Sigma' or 'invSigma' must be given.")
    if (!is.null(Sigma)) {
        if (is.list(Sigma)) {
            ev <- Sigma$values
            evec <- Sigma$vectors
        } else {
            ed <- eigen(Sigma, symmetric = TRUE)
            ev <- ed$values
            evec <- ed$vectors
        }
        if (!all(ev >= -1e-06 * abs(ev[1]))) 
            stop("'Sigma' is not positive definite")
        invSigma <- evec %*% (t(evec)/ev)
        if (!prop)
            logdetSigma <- sum(log(ev))
    } else {
        if (!prop)
            logdetSigma <- c(-determinant(invSigma)$modulus)
    }
    ss <- x - rep(mu, each = nrow(x))
    quad <- rowSums((ss %*% invSigma) * ss)/df
    if (!prop)
        fact <- lgamma((df + p)/2) - lgamma(df/2) - 
        0.5 * (p * (log(pi) + log(df)) + logdetSigma)
    if (log) {
        if (!prop) as.vector(fact - 0.5 * (df + p) * log(1 + quad)) else 
            as.vector(- 0.5 * (df + p) * log(1 + quad))
    } else {
        if (!prop) as.vector(exp(fact) * ((1 + quad)^(-(df + p)/2))) else 
            as.vector(((1 + quad)^(-(df + p)/2)))
    }
}

# rmvt
# Generates random samples from a multivariate t distribution with location mu,
# scale matrix Sigma, and df degrees of freedom. Uses the representation:
#   X = mu + Z / sqrt(chi^2(df) / df)
# where Z ~ N(0, Sigma). Sigma can be provided as a matrix or as a list with
# $values and $vectors (eigendecomposition) for efficiency.
#
# Used in simulate.MixMod() when the family is Student's-t.
#
# Arguments:
#   n:     number of random vectors to generate
#   mu:    location vector of length p
#   Sigma: p x p positive-definite scale matrix, or a list with $values/$vectors
#   df:    degrees of freedom
#
# Returns:
#   If n == 1: a numeric vector of length p.
#   If n > 1: an n x p matrix with one sample per row.
rmvt <- function (n, mu, Sigma, df) {
    p <- length(mu)
    if (is.list(Sigma)) {
        ev <- Sigma$values
        evec <- Sigma$vectors
    } else {
        ed <- eigen(Sigma, symmetric = TRUE)
        ev <- ed$values
        evec <- ed$vectors
    }
    X <- drop(mu) + tcrossprod(evec * rep(sqrt(pmax(ev, 0)), each = p), 
                               matrix(rnorm(n * p), n)) / rep(sqrt(rchisq(n, df)/df), each = p)
    if (n == 1L) drop(X) else t.default(X)
}

# register_s3_method
# Registers an S3 method for a generic function defined in another package (pkg) without
# requiring that package to be loaded at install time. This pattern avoids hard dependencies:
# if pkg is already loaded, the method is registered immediately; additionally, a hook is
# set to register it whenever pkg is loaded in the future.
#
# Used in .onLoad() to register MixMod methods for emmeans and effects packages.
#
# Arguments:
#   pkg:     character string; the name of the package that owns the generic
#   generic: character string; the name of the generic function (e.g., "recover_data")
#   class:   character string; the class for which the method is defined (e.g., "MixMod")
#
# Returns:
#   NULL (invisibly); called for its side-effect of registering the S3 method.
register_s3_method <- function (pkg, generic, class) {
    fun <- get(paste0(generic, ".", class), envir = parent.frame())
    if (isNamespaceLoaded(pkg))
        registerS3method(generic, class, fun, envir = asNamespace(pkg))
    # Also ensure registration is done if pkg is loaded later:
    setHook(
        packageEvent(pkg, "onLoad"),
        function (...)
            registerS3method(generic, class, fun, envir = asNamespace(pkg))
    )
}

# .onLoad
# Package load hook that registers S3 methods for optional (Suggested) packages:
#   - emmeans: registers recover_data.MixMod and emm_basis.MixMod so that
#              emmeans() can compute estimated marginal means for MixMod objects.
#   - effects: registers Effect.MixMod so that the effects package can compute
#              and plot effects for MixMod objects.
# Methods are only registered if the respective packages are available, allowing
# GLMMadaptive to function without these dependencies.
.onLoad <- function (libname, pkgname) {
    if (requireNamespace("emmeans", quietly = TRUE)) {
        register_s3_method("emmeans", "recover_data", "MixMod")
        register_s3_method("emmeans", "emm_basis", "MixMod")
    }
    if (requireNamespace("effects", quietly = TRUE)) {
        register_s3_method("effects", "Effect", "MixMod")
    }
}

# constructor_form_random
# Constructs a list of random-effects formulas (one per grouping level) from the user-
# supplied random formula. For a simple formula like ~ time | id, it returns a one-element
# list named "id" with formula ~time. For nested grouping (multiple levels), it creates
# interaction terms so that the random effects at each level are nested within those at
# higher levels (e.g., a two-level model with subjects within centers).
#
# For nested grouping (formula is a list or has multiple grouping variables), the inner
# groups are parameterized using dummy coding via the nesting() helper:
#   ~ 0 + group + group:re_terms
# which creates separate random effects for each level of the outer group.
#
# Arguments:
#   formula: either a standard random formula ~ <terms> | <group>, or a named list
#            for nested grouping structures
#   data:    the data.frame (used only to evaluate the formula)
#
# Returns:
#   A named list of one-sided formulas, one per grouping level, to be used with
#   model.frame() and constructor_Z() to build the Z design matrix.
constructor_form_random <- function (formula, data) {
    groups <- all.vars(getID_Formula(formula))
    ngroups <- length(groups)
    formula <- if (!is.list(formula)) {
        form_random <- vector("list", ngroups)
        names(form_random) <- groups
        form_random[] <- lapply(form_random, function (x) getRE_Formula(formula))
    } else formula
    if (ngroups > 1) {
        nesting <- function (form, group_name) {
            terms_form <- attr(terms(form), "term.labels")
            if (length(terms_form)) {
                interaction_terms <- paste0(group_name, ":", terms_form, collapse = " + ")
                as.formula(paste0("~ 0 + ", group_name, " + ", interaction_terms))
            } else {
                as.formula(paste0("~ 0 + ", group_name))
            }
        }
        formula[-1] <- mapply(nesting, formula[-1], groups[-1], SIMPLIFY = FALSE)
    }
    formula
}

# constructor_Z
# Builds the random-effects design matrix Z by constructing separate model matrices for
# each subject/group and then row-binding them. For each group i, a model matrix is
# computed from the random-effects terms and model frame subset. The columns are reordered
# to match the "assign" attribute ordering, ensuring consistent column ordering across
# groups (important when groups have different numbers of observations but the same terms).
#
# Arguments:
#   termsZ_i: terms object for the random-effects formula (from constructor_form_random)
#   mfZ_i:    model frame for the random-effects formula (all subjects combined)
#   id:       integer vector of subject/group indices (length = number of observations)
#
# Returns:
#   The random-effects design matrix Z (n_obs x ncz), constructed by stacking the per-
#   group model matrices, with columns in the same order as the terms assignment.
constructor_Z <- function (termsZ_i, mfZ_i, id) {
    n <- length(unique(id))
    Zmats <- vector("list", n)
    for (i in seq_len(n)) {
        #mf <- model.frame(termsZ_i, mfZ_i[id == i, , drop = FALSE],
        #                  drop.unused.levels = TRUE)
        mf <- mfZ_i[id == i, , drop = FALSE]
        mm <- model.matrix(termsZ_i, mf)
        assign <- attr(mm, "assign")
        assgn <- sapply(unique(assign), function (x) which(assign == x))
        if (is.list(assgn))
            assgn <- unlist(assgn, use.names = FALSE)
        Zmats[[i]] <- mm[, c(t(assgn)), drop = FALSE]
    }
    do.call("rbind", Zmats)
}

# cr_setup
# Prepares the data for fitting a continuation ratio (CR) model for an ordinal response.
# A CR model decomposes the ordinal response into a series of binary comparisons:
#   - Forward direction: P(Y = j | Y >= j) for j = 1, ..., K-1
#   - Backward direction: P(Y = j | Y <= j) for j = K-1, ..., 1
#
# This is achieved by expanding each observation into multiple "pseudo-observations",
# one for each applicable binary comparison (cohort). The resulting binary outcome Y
# and cohort indicator can then be used with a standard binary mixed model.
#
# Arguments:
#   y:         factor or numeric vector of ordinal responses (with at least 3 levels)
#   direction: "forward" (default) for P(Y=j|Y>=j), or "backward" for P(Y=j|Y<=j)
#
# Returns:
#   A list with:
#     y:      binary response vector (0/1) for each pseudo-observation
#     cohort: factor indicating which comparison/cohort each pseudo-obs belongs to
#     subs:   integer vector of original observation indices (for sub-setting covariates)
#     reps:   integer vector giving the replication count for each original observation
cr_setup <- function (y, direction = c("forward", "backward")) {
    direction <- match.arg(direction)
    yname <- as.character(substitute("y"))
    if (!is.factor(y)) {
        y <- factor(y)
    }
    ylevels <- levels(y)
    ncoefs <- length(ylevels) - 1
    if (ncoefs < 2) {
        stop("it seems that variable ", yname, " has two levels; use a mixed effects ", 
             "logistic regression instead.\n")
    }
    y <- as.numeric(unclass(y) - 1)
    if (direction == "forward") {
        reps <- ifelse(is.na(y), 1, ifelse(y < ncoefs - 1, y + 1, ncoefs))
        subs <- rep(seq_along(y), reps)
        cuts <- vector("list", ncoefs + 2)
        cuts[[1]] <- NA
        for (j in seq(0, ncoefs)) {
            cuts[[j + 2]] <- seq(0, if (j < ncoefs - 1) j else ncoefs - 1)
        }
        cuts <- unlist(cuts[ifelse(is.na(y), 1, y + 2)], use.names = FALSE)
        labels <- c("all", paste0(yname, ">=", ylevels[2:ncoefs]))
        y <- rep(y, reps)
        Y <- as.numeric(y == cuts)
    } else {
        reps <- ifelse(is.na(y), 1, ifelse(y > 1, 2 + (ncoefs - 1 - y) , ncoefs))
        subs <- rep(seq_along(y), reps)
        cuts <- vector("list", ncoefs + 2)
        cuts[[ncoefs + 2]] <- NA
        for (j in seq(ncoefs, 0)) {
            cuts[[j + 1]] <- seq(0, ncoefs - if (j > 1) j else 1)
        }
        cuts <- unlist(cuts[ifelse(is.na(y), ncoefs + 2, y + 1)], use.names = FALSE)
        labels <- c("all", paste0(yname, "<=", ylevels[ncoefs:2]))
        y <- rep(y, reps)
        Y <- as.numeric(y == (ncoefs - cuts))
    }
    cohort <- factor(cuts, levels = seq(0, ncoefs - 1), labels = labels)
    list(y = Y, cohort = cohort, subs = subs, reps = reps)
}

# cr_marg_probs
# Computes the marginal (category) probabilities P(Y = j) for j = 0, ..., K from the
# linear predictors eta of a fitted continuation ratio model. Converts the conditional
# probabilities (from the CR model) to marginal probabilities using the chain rule:
#
#   Forward:  P(Y = j) = P(Y = j | Y >= j) * prod_{k=0}^{j-1} P(Y > k | Y >= k)
#   Backward: P(Y = j) = P(Y = j | Y <= j) * prod_{k=j+1}^{K} P(Y < k | Y <= k)
#
# The last (or first) category probability is obtained as 1 - sum of all other probabilities.
# Uses log-scale computations for numerical stability.
#
# Arguments:
#   eta:       n x (K-1) matrix of linear predictors, one column per cohort/comparison
#   direction: "forward" (default) or "backward", matching the direction used in cr_setup()
#
# Returns:
#   An n x K matrix of marginal category probabilities (each row sums to 1).
cr_marg_probs <- function (eta, direction = c("forward", "backward")) {
    direction <- match.arg(direction)
    ncoefs <- ncol(eta)
    if (direction == "forward") {
        cumsum_1_minus_p <- apply(plogis(eta[, -ncoefs, drop = FALSE], log.p = TRUE, 
                                           lower.tail = FALSE), 1L, cumsum)
        if (is.matrix(cumsum_1_minus_p)) cumsum_1_minus_p <- t(cumsum_1_minus_p)
        probs <- exp(plogis(eta, log.p = TRUE) + cbind(0, cumsum_1_minus_p))
        cbind(probs, 1 - rowSums(probs))
    } else {
        cumsum_1_minus_p <- apply(plogis(eta[, seq(ncoefs, 2), drop = FALSE], log.p = TRUE, 
                                           lower.tail = FALSE), 1L, cumsum)
        cumsum_1_minus_p <- if (is.matrix(cumsum_1_minus_p)) t(cumsum_1_minus_p) else as.matrix(cumsum_1_minus_p)
        probs <- exp(plogis(eta, log.p = TRUE) + 
                         cbind(cumsum_1_minus_p[, seq(ncoefs - 1, 1)], 0))
        cbind(1 - rowSums(probs), probs)
    }
}
