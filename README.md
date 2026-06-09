# renew-glm

**Streaming generalized linear models with bounded memory** -- a Python
port of the renewable-estimation algorithm of Luo & Song (2020).

![Cost trade-off at n=128M gaussian](https://raw.githubusercontent.com/tommycarstensen/renew-glm/main/docs/pareto.png)

```python
from renew_glm import RenewGLM

# Streaming (recommended): one chunk at a time, never stores prior chunks.
# Peak RAM is O(p^2 + chunk_size * p), independent of n.
def chunk_fn():
    for X_chunk, y_chunk in source_iter():   # your generator
        yield X_chunk, y_chunk               # X must include intercept column

model = RenewGLM(family="binomial").fit_streaming(chunk_fn)
print(model.coef_, model.n_iter_)

# Or chunk-buffered (matches the original R API):
model = RenewGLM(family="poisson")
for X_chunk, y_chunk in chunks:
    model.partial_fit(X_chunk, y_chunk)
model.fit()
print(model.coef_, model.se_, model.pvalue_)
```

`chunk_fn` is a zero-argument callable returning an iterator of `(X, y)`
tuples. `fit_streaming` consumes the iterator once and discards each chunk
after use; `partial_fit` + `fit` buffers all chunks in memory so it can
compute `se_` and `pvalue_` like the R reference.

Supports gaussian, binomial, and poisson families. Coefficients converge to
the maximum-likelihood point of the full data; agreement with
`statsmodels.GLM` is verified to ~1e-3 on the test suite.

## Why

Standard in-memory tools (`statsmodels.GLM`, R's `glm()`) load the
entire design matrix before fitting. At n = 16 M rows, `statsmodels.GLM`
allocates ~8 GB and OOMs above n ~ 20 M on a 16 GB laptop. This package
fits the same model in **bounded memory** -- one chunk at a time,
O(p^2) state regardless of n.

Other Python GLM options exist with different trade-offs:

- **`glum`** (Quantco) -- fast in-memory; bit-identical to the
  closed-form MLE we get, but caps out where the full design matrix
  fits in RAM, same as statsmodels. Recommended *if* your `n` fits.
- **`dask-glm`** -- distributed; useful at multi-machine scale, has
  scheduler overhead at single-host scale.
- **`pyglmnet`** -- gradient-based, regularization-focused; supports
  several families but the unregularized path is slower than IRLS.
- **`scikit-learn` SGD** -- online SGD, approximate (not exact MLE).

This package targets the gap: **exact-MLE streaming on a single host,
no full-design-matrix RAM**. It carries the same Wald inference the
R reference (`biglm`, `RenewGLM_pkg`) exposes.

## Install

```bash
pip install renew-glm
```

Requires only NumPy and SciPy. Pure Python -- no C extension, no compile
step, no platform-specific wheels.

## Correctness

Across seven independent implementations spanning Python (NumPy / Numba
/ JAX), an in-memory Python competitor (glum), a distributed Python
competitor (dask-glm), and a cross-language reference (R's biglm), the
coefficient estimates agree to floating-point machine epsilon
(`~1e-15`) on the largest workload we test.

![Cross-method coefficient agreement at n=128M](https://raw.githubusercontent.com/tommycarstensen/renew-glm/main/docs/heatmap.png)

Each cell shows `log10(max|beta_i - beta_j|)`. The renew-glm /
dask-glm corners hit FP epsilon at -15; glum and biglm sit one order
out at -10/-11 (algorithm-decomposition path, not convergence). The
JAX entry uses `jax_enable_x64=True` -- the silent-float32 default
would land an order of magnitude looser.

## Scaling

Wall time and peak RAM as `n` grows from 4M to 128M rows (cold runs;
single 16 GB laptop). The in-memory baselines (statsmodels) terminate
where they OOM; renew-glm and biglm keep going at bounded RAM.

![Scaling: wall time and peak RAM vs n](https://raw.githubusercontent.com/tommycarstensen/renew-glm/main/docs/scaling.png)

## Algorithm

For each chunk:

1. Compute Fisher information `H_b = X' diag(W_b) X` at the current
   coefficient.
2. Inner Newton-Raphson against `H_b + sum_of_prior_Fishers`, with a
   penalty term that pulls toward the previous coefficient.
3. Update `sum_of_prior_Fishers += H_b` and move to the next chunk.

One outer pass over the chunks suffices because the penalty term lets each
chunk contribute meaningfully without revisiting prior data. The docstring
in `_irls.py` matches the paper notation.

## Differences from the original R package

- **No SE in the streaming path.** `fit_streaming(chunk_fn)` returns only
  `coef_` and `n_iter_`. The chunk-buffered `partial_fit` + `fit()` path
  computes `se_` and `pvalue_` like the R version.
- **Pure NumPy/SciPy.** No C extension; portable and easy to install.
- **Convergence tolerance** uses `|g' d_beta| < tol` (same as the R
  version's `df_beta` criterion).

## Roadmap (post v0.1.0)

Deferred to keep the first release minimal; pull requests welcome.

- **Formula API** (`patsy` / `formulaic`) -- a `from_formula("y ~ x1 +
  C(category)", chunk_source=...)` constructor so users coming from
  `statsmodels.GLM.from_formula(...)` or R's `glm(y ~ ...)` don't have
  to build the design matrix themselves. The streaming twist: category
  levels must be discovered before fitting, so the API will require
  either a first-pass `discover_levels(source)` step or an explicit
  `levels={...}` argument. First-chunk-lock-in (reject any chunk
  introducing a new level, with a clear error) is the most likely
  default. Workaround today: use `patsy.dmatrices(...)` per-chunk
  and feed the resulting arrays to `partial_fit` / `fit_streaming`.
- **Gamma + inverse-Gaussian families** -- the algorithm generalises
  (any exponential-dispersion family with a known link works), but
  the test suite only covers gaussian / binomial / poisson. Adding a
  family is ~10 LOC of weight + mu functions in `_irls.py` plus a
  test case.
- **Optional Cholesky -> Givens-QR path** -- the current code uses
  `scipy.linalg.cho_factor` on `X' W X + sum_prior_Fisher`. For
  ill-conditioned designs a Givens-QR path on `[W^{1/2} X ; previous-R]`
  would be more numerically stable; the bench's R reference (biglm)
  already does this and our coefficients agree to ~1e-6, suggesting
  the Cholesky path is sufficient for typical inputs. Track this if
  a real user hits a conditioning problem.

## Credit

- **Algorithm** by Lan Luo and Peter X.-K. Song. Please cite the original
  paper when using this package:

  > Luo L. & Song R. (2020). "Renewable Estimation and Incremental
  > Inference in Generalized Linear Models with Streaming Data Sets."
  > *Journal of the Royal Statistical Society: Series B*, **82**(1),
  > 69-97. <https://doi.org/10.1111/rssb.12352>

- **Reference R implementation** by Luo & Song:
  <https://github.com/luolsph/RenewGLM_pkg>

- **Python port** by Tommy Carstensen.
  ORCID: <https://orcid.org/0000-0002-3672-9931>

## License

GPL-2.0-or-later, matching the license of the original R package.
