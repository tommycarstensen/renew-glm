"""
Renewable (incremental) GLM -- Python reimplementation of Luo & Song (2020).

Luo L. & Song R. (2020). Renewable Estimation and Incremental Inference in
Generalized Linear Models with Streaming Data Sets. JRSS-B 82(1), 69-97.
DOI: https://doi.org/10.1111/rssb.12352
Original R package: https://github.com/luolsph/RenewGLM_pkg

Algorithm (inloop variant)
--------------------------
Maintain accumulated Fisher information sum2 (p x p) across batches.
For each batch b:
  1. betahat_old <- betahat from previous batch (or 0 initially).
  2. Compute IRLS weights W at betahat_old.
  3. Compute batch Fisher H = X' diag(W) X.
  4. Inner Newton-Raphson: solve (sum2 + H) d = g until |g'd| < tol, where
       g = X'(y - mu)  -  sum2 (beta - betahat_old).
  5. Update sum2 += X' diag(W_new) X evaluated at converged beta.
After all batches: SE = sqrt(diag(sum2^{-1})) [* sqrt(phi) for Gaussian].

Supports: "gaussian" (identity link), "binomial" (logit link),
          "poisson" (log link).

Usage
-----
    model = RenewGLM(family="binomial")
    for chunk_X, chunk_y in chunks:   # X already includes intercept column
        model.partial_fit(chunk_X, chunk_y)
    model.fit()
    print(model.coef_, model.se_)
"""

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.stats import norm

__all__ = ["RenewGLM"]


def _eta(X, beta):
    return X @ beta


def _mu(eta, family):
    if family == "gaussian":
        return eta
    if family == "binomial":
        return np.where(
            eta >= 0,
            1.0 / (1.0 + np.exp(-eta)),
            np.exp(eta) / (1.0 + np.exp(eta)),
        )
    # poisson
    return np.exp(np.clip(eta, -500, 500))


def _weights(eta, family):
    """IRLS weights: d(mu)/d(eta)^2 / Var(mu)  =  1 / g'(mu)^2 / V(mu)."""
    if family == "gaussian":
        return np.ones(len(eta))
    if family == "binomial":
        mu = _mu(eta, "binomial")
        return np.clip(mu * (1.0 - mu), 1e-15, None)
    # poisson
    return np.exp(np.clip(eta, -500, 500))


def _xtWx(X, w):
    """X' diag(w) X, vectorised."""
    return (X * w[:, None]).T @ X


def _solve_spd(A, g, cf):
    """Solve A x = g where A is symmetric positive-definite.

    Uses the precomputed Cholesky factor `cf` (from `cho_factor`) when
    available; falls back to LU and finally to least-squares if both
    fail. Centralised so the inner Newton loop doesn't need a nested
    closure to capture A / cf / use_cho -- pyright couldn't infer the
    type of that closure cleanly."""
    if cf is not None:
        try:
            return cho_solve(cf, g, check_finite=False)
        except (LinAlgError, np.linalg.LinAlgError):
            pass
    try:
        return np.linalg.solve(A, g)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, g, rcond=None)[0]


class RenewGLM:
    """
    Incremental GLM via accumulated Fisher information (Luo & Song 2020).

    Parameters
    ----------
    family : {"gaussian", "binomial", "poisson"}
    tol : float
        Inner NR convergence threshold: stop when |g' d_beta| < tol.
    max_iter : int
        Maximum inner NR iterations per batch.
    """

    def __init__(
        self,
        family: str = "gaussian",
        tol: float = 1e-6,
        max_iter: int = 100,
    ):
        if family not in ("gaussian", "binomial", "poisson"):
            raise ValueError(
                "family must be 'gaussian', 'binomial', or 'poisson'"
            )
        self.family = family
        self.tol = tol
        self.max_iter = max_iter
        self._chunks: list[tuple] = []

    def partial_fit(self, X, y):
        """Accumulate one data chunk. X must include the intercept column.

        NOTE: this is the in-memory path -- chunks are retained in self._chunks
        until fit() consumes them, so peak RAM is O(n*p). For true streaming
        (O(p^2) RAM regardless of n) use fit_streaming(chunk_fn) instead.
        """
        self._chunks.append((
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64).ravel(),
        ))
        return self

    def fit_streaming(self, chunk_fn):
        """Streaming variant: never stores chunks. Memory is O(p^2).

        chunk_fn() returns (X, y) for the next chunk, or None when done.
        The returned chunk is consumed and released before the next call,
        so peak RAM is one chunk + sum2 (p x p) + A (p x p). Matches the
        biglm::bigglm callback pattern.

        Sets coef_ and n_iter_. SE / dispersion are not computed in the
        streaming path (would need additional accumulators; the
        chunk-buffered fit() above provides them).
        """
        beta = sum2 = None
        p = None
        n_chunks = 0
        while True:
            chunk = chunk_fn()
            if chunk is None:
                break
            X, y = chunk
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64).ravel()
            if p is None:
                p = X.shape[1]
                beta = np.zeros(p)
                sum2 = np.zeros((p, p))
            assert beta is not None and sum2 is not None
            n_chunks += 1
            beta_old = beta.copy()

            # Batch Fisher at beta_old.
            eta_old = _eta(X, beta_old)
            W_old = _weights(eta_old, self.family)
            H = _xtWx(X, W_old)

            A = sum2 + H
            try:
                cf = cho_factor(A, lower=False, check_finite=False)
            except (LinAlgError, np.linalg.LinAlgError):
                cf = None

            # Inner Newton-Raphson.
            for _ in range(self.max_iter):
                mu = _mu(_eta(X, beta), self.family)
                g0 = X.T @ (y - mu)
                g1 = -(sum2 @ (beta - beta_old))
                g = g0 + g1
                d = _solve_spd(A, g, cf)
                if abs(float(g @ d)) < self.tol:
                    break
                beta = beta + d

            # Accumulate Fisher at converged beta; chunk released next iter.
            W_new = _weights(_eta(X, beta), self.family)
            sum2 = sum2 + _xtWx(X, W_new)

        if beta is None:
            raise ValueError("fit_streaming(): no chunks were provided by chunk_fn")
        self.coef_ = beta
        self.n_iter_ = n_chunks
        return self

    def fit(self):
        """
        Run renewable estimation over all accumulated chunks.
        Sets coef_, se_, pvalue_.
        """
        if not self._chunks:
            raise RuntimeError("No data: call partial_fit() before fit().")

        p = self._chunks[0][0].shape[1]
        beta = np.zeros(p)
        sum2 = np.zeros((p, p))
        phi = 0.0
        s = 0          # cumulative sample count (Gaussian only)

        for X, y in self._chunks:
            n = len(y)
            beta_old = beta.copy()

            # Batch Fisher at beta_old
            eta_old = _eta(X, beta_old)
            W_old = _weights(eta_old, self.family)
            H = _xtWx(X, W_old)

            A = sum2 + H
            try:
                cf = cho_factor(A, lower=False, check_finite=False)
            except (LinAlgError, np.linalg.LinAlgError):
                cf = None

            # Inner NR
            for _ in range(self.max_iter):
                mu = _mu(_eta(X, beta), self.family)
                g0 = X.T @ (y - mu)                   # score
                g1 = -(sum2 @ (beta - beta_old))       # accumulated penalty
                g = g0 + g1
                d = _solve_spd(A, g, cf)
                if abs(float(g @ d)) < self.tol:
                    break
                beta = beta + d

            # Gaussian dispersion -- running MSE estimate (eq. 5 Luo & Song)
            if self.family == "gaussian":
                s_old = s
                s = s + n
                if s > p:
                    penalty = float(beta_old @ (sum2 @ (beta_old - beta)))
                    rss = float(y @ (y - X @ beta))
                    phi = (
                        max(0, s_old - p) / (s - p) * phi
                        + penalty / (s - p)
                        + rss / (s - p)
                    )

            # Accumulate Fisher at converged beta
            W_new = _weights(_eta(X, beta), self.family)
            sum2 = sum2 + _xtWx(X, W_new)

        sum2_inv = np.linalg.inv(sum2)
        if self.family == "gaussian":
            se = np.sqrt(np.diag(sum2_inv) * max(phi, 0.0))
        else:
            se = np.sqrt(np.diag(sum2_inv))

        self.coef_ = beta
        self.se_ = se
        self.pvalue_ = 2.0 * norm.cdf(-np.abs(beta / np.where(se > 0, se, 1)))
        self.n_iter_ = len(self._chunks)
        return self
