# JAX Integration Roadmap

## Motivation

The current Python port uses finite-difference numerical differentiation
(central differences, `cd_grad` / `cd_hess`) to compute gradients and Hessians
of the marginal log-likelihood and the per-group log-posterior.  This mirrors
the R implementation exactly, but it has two limitations:

1. **Accuracy**: finite differences introduce truncation error (O(h²) for central
   differences vs. machine precision for automatic differentiation).
2. **Speed**: each gradient evaluation requires O(p) or O(p²) forward passes
   through the likelihood.

[JAX](https://jax.readthedocs.io) can replace all finite-difference calls with
exact gradients via forward-mode (`jax.jacfwd`) or reverse-mode (`jax.grad`)
automatic differentiation.

## Migration Path

### Phase 1 — Optional JAX backend (non-breaking)

Add an `engine` parameter to `MixedModel.fit()`:

```python
result = model.fit(engine="jax")  # new; default remains "numpy"
```

The JAX engine would:
1. Re-implement `log_dens` for each family using `jax.numpy` primitives.
2. Wrap `loglik_mixed` / `score_betas` in `jax.jit` for compilation.
3. Replace `cd_grad(neg_log_post, ...)` in `find_posterior_mode` with
   `jax.grad(neg_log_post)`.
4. Replace `cd_hess(neg_ll_final, ...)` in `mixed_fit` with
   `jax.hessian(neg_ll_final)`.

### Phase 2 — First-class JAX family objects

Define a `JaxBaseFamily` subclass whose `log_dens` is a pure JAX function.
Family authors can subclass `JaxBaseFamily` instead of `BaseFamily` to opt in
to exact gradients automatically.

```python
import jax.numpy as jnp
from glmmadaptive.families.jax_base import JaxBaseFamily

class JaxBinomial(JaxBaseFamily):
    def log_dens(self, y, eta, phis=None, eta_zi=None):
        mu = jax.nn.sigmoid(eta)
        return y * jnp.log(mu) + (1 - y) * jnp.log(1 - mu)
```

### Phase 3 — GPU / TPU support

Because JAX transparently maps to XLA, phase 2 models automatically benefit
from GPU/TPU acceleration for large datasets with many groups.

## Dependencies

```toml
[project.optional-dependencies]
jax = ["jax>=0.4", "jaxlib>=0.4"]
```

Install with: `pip install glmmadaptive[jax]`

## Notes

- JAX requires pure functions (no in-place mutation, no Python side-effects in
  hot loops).  The `GHQuadrature` node construction would need to be refactored
  to use `jax.lax.scan` or `vmap` instead of Python for-loops.
- `nearPD` (Higham 2002 iteration) is inherently sequential and may not
  benefit much from JAX unless the iteration is unrolled with `lax.while_loop`.
- The `find_posterior_mode` BFGS optimisation can be replaced with
  `jaxopt.LBFGS` for a fully-JAX end-to-end pipeline.
