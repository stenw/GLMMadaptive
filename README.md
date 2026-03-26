# GLMMadaptive (Python)

This repository hosts a **Python adaptation** of the R package
[GLMMadaptive](https://drizopoulos.github.io/GLMMadaptive/) by Dimitris
Rizopoulos. It fits generalized linear mixed models for clustered /
repeated-measures data using **adaptive Gauss-Hermite quadrature**.

This Python port is **not yet feature-complete** relative to the R package.
Coverage is growing, but some families and utilities are still missing.

## What works today

- Core model fitting via `MixedModel` with formula strings for fixed and
  random effects (e.g. `~ 1 | id`, `~ time | id`, diagonal via `||`).
- Implemented families:
  - `Binomial`, `Poisson`, `NegativeBinomial`
  - `Gamma`, `Beta`
  - `Gaussian`, `StudentsT`
  - Zero-inflated `ZIPoisson`, `ZINegativeBinomial`
- Results and utilities: `summary()`, `fixef()`, `ranef()`, `vcov()`,
  `confint()`, `predict()`, `fitted()`, `residuals()`, `anova()`, and
  dynamic prediction via `predict_dynamic()`.

## Basic usage

```python
import pandas as pd
from glmmadaptive import MixedModel
from glmmadaptive.families import Binomial

model = MixedModel(
    fixed="y ~ time + treatment",
    random="~ 1 | id",
    data=df,
    family=Binomial(),
)
res = model.fit()
print(res.summary())
```

## Installation (from source)

```bash
pip install -e "python"
```

The Python package lives under `python/`. See `python/README.md` for more
details, tests, and vignettes.
