"""Smoke tests for renew_glm.

Each test:
  1. Simulates a small dataset with known coefficients.
  2. Fits via RenewGLM (both partial_fit + fit AND fit_streaming).
  3. Verifies coefficients match statsmodels.GLM to ~1e-3.

statsmodels is a test-only dependency (pip install .[test])."""

import numpy as np
import pytest
import statsmodels.api as sm

from renew_glm import RenewGLM


N = 5_000
P = 6  # number of predictors (excluding intercept)
SEED = 42
TOL = 5e-3  # MLE vs MLE: tighter than sampling noise (~1/sqrt(N) = 0.014)


def _make_design():
    rng = np.random.default_rng(SEED)
    X_inner = rng.standard_normal((N, P))
    X = np.column_stack([np.ones(N), X_inner])
    return rng, X


def _chunks_of(X, y, n_chunks=5):
    idx = np.array_split(np.arange(len(y)), n_chunks)
    return [(X[i], y[i]) for i in idx]


def _statsmodels_coef(y, X, family_name):
    fam = {"binomial": sm.families.Binomial(),
           "poisson": sm.families.Poisson(),
           "gaussian": sm.families.Gaussian()}[family_name]
    return sm.GLM(y, X, family=fam).fit().params


# ---------------------------------------------------------------------------
# Chunk-buffered path (partial_fit + fit)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family,beta_true", [
    ("binomial", np.array([-0.5, 1.5, -1.2, 0.5, 0.3, -0.2, 0.4])),
    ("poisson", np.array([0.12, 0.16, 0.12, 0.08, 0.04, -0.04, 0.0])),
    ("gaussian", np.array([1.0, 0.5, -0.3, 0.2, 0.4, -0.1, 0.3])),
])
def test_partial_fit_then_fit_matches_statsmodels(family, beta_true):
    rng, X = _make_design()
    if family == "binomial":
        mu = 1.0 / (1.0 + np.exp(-X @ beta_true))
        y = rng.binomial(1, mu).astype(np.float64)
    elif family == "poisson":
        mu = np.exp(X @ beta_true)
        y = rng.poisson(mu).astype(np.float64)
    else:
        y = X @ beta_true + rng.normal(0.0, 1.0, N)

    model = RenewGLM(family=family, tol=1e-8, max_iter=100)
    for X_c, y_c in _chunks_of(X, y, n_chunks=5):
        model.partial_fit(X_c, y_c)
    model.fit()

    ref = _statsmodels_coef(y, X, family)
    assert np.max(np.abs(model.coef_ - ref)) < TOL, (
        f"{family}: |coef - statsmodels| max = "
        f"{np.max(np.abs(model.coef_ - ref)):.2e}"
    )
    assert model.coef_.shape == (P + 1,)
    assert model.se_.shape == (P + 1,)
    assert model.pvalue_.shape == (P + 1,)


# ---------------------------------------------------------------------------
# Streaming path (fit_streaming)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family,beta_true", [
    ("binomial", np.array([-0.5, 1.5, -1.2, 0.5, 0.3, -0.2, 0.4])),
    ("poisson", np.array([0.12, 0.16, 0.12, 0.08, 0.04, -0.04, 0.0])),
    ("gaussian", np.array([1.0, 0.5, -0.3, 0.2, 0.4, -0.1, 0.3])),
])
def test_fit_streaming_matches_statsmodels(family, beta_true):
    rng, X = _make_design()
    if family == "binomial":
        mu = 1.0 / (1.0 + np.exp(-X @ beta_true))
        y = rng.binomial(1, mu).astype(np.float64)
    elif family == "poisson":
        mu = np.exp(X @ beta_true)
        y = rng.poisson(mu).astype(np.float64)
    else:
        y = X @ beta_true + rng.normal(0.0, 1.0, N)

    chunks = _chunks_of(X, y, n_chunks=5)
    state = {"i": 0}

    def chunk_fn():
        if state["i"] >= len(chunks):
            return None
        c = chunks[state["i"]]
        state["i"] += 1
        return c

    model = RenewGLM(family=family, tol=1e-8, max_iter=100)
    model.fit_streaming(chunk_fn)

    ref = _statsmodels_coef(y, X, family)
    assert np.max(np.abs(model.coef_ - ref)) < TOL, (
        f"{family} (streaming): |coef - statsmodels| max = "
        f"{np.max(np.abs(model.coef_ - ref)):.2e}"
    )
    assert model.coef_.shape == (P + 1,)
    assert model.n_iter_ == len(chunks)


# ---------------------------------------------------------------------------
# API contract: invalid family raises early
# ---------------------------------------------------------------------------

def test_invalid_family_raises():
    with pytest.raises(ValueError, match="family"):
        RenewGLM(family="gamma")


# ---------------------------------------------------------------------------
# Streaming and chunk-buffered paths agree on the SAME data
# ---------------------------------------------------------------------------

def test_streaming_matches_partial_fit_on_same_data():
    rng, X = _make_design()
    beta_true = np.array([-0.5, 1.5, -1.2, 0.5, 0.3, -0.2, 0.4])
    mu = 1.0 / (1.0 + np.exp(-X @ beta_true))
    y = rng.binomial(1, mu).astype(np.float64)
    chunks = _chunks_of(X, y, n_chunks=5)

    m_buf = RenewGLM(family="binomial", tol=1e-8, max_iter=100)
    for X_c, y_c in chunks:
        m_buf.partial_fit(X_c, y_c)
    m_buf.fit()

    state = {"i": 0}

    def chunk_fn():
        if state["i"] >= len(chunks):
            return None
        c = chunks[state["i"]]
        state["i"] += 1
        return c
    m_str = RenewGLM(family="binomial", tol=1e-8, max_iter=100).fit_streaming(chunk_fn)

    assert np.max(np.abs(m_buf.coef_ - m_str.coef_)) < 1e-10, (
        "streaming and chunk-buffered fits should converge to the same point"
    )
