#!/usr/bin/env python3.13
"""Generate the three README figures for renew-glm-py:

  pareto.png       -- RAM vs wall time @ n=128M, hybrid 7-method set
  heatmap.png      -- pairwise coefficient distance @ n=128M
  scaling.png      -- wall time + RAM vs n, hybrid 7-method set

Hybrid 7-method set:
  renew_glm                 (our package, the headline)
  renew_glm_numba           (our package, alternative backend -- bench-only)
  renew_glm_jax_x64         (our package, alternative backend -- bench-only)
  glum                      (Python competitor, in-memory)
  dask_glm                  (Python competitor, distributed)
  statsmodels_full          (Python in-memory baseline; OOMs at scale)
  biglm                     (R cross-language reference)

Standalone -- no imports from src/glm/gen_handout.py (which the
handout-owning agent is iterating on). Pulls data straight from
results/competing_glm_cold.json.

Usage:
    python3.13 packages/renew-glm-py/docs/gen_readme_figures.py

Writes PNG files to packages/renew-glm-py/docs/ for embedding in
README.md via GitHub-raw URLs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm   # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RESULTS_COLD = _REPO_ROOT / "results" / "competing_glm_cold.json"
_RESULTS_WARM = _REPO_ROOT / "results" / "competing_glm.json"
_OUT_DIR = Path(__file__).resolve().parent

_N_REF = 128_000_000
_FAMILY = "gaussian"

# Region Hovedstaden palette, also used elsewhere on the project's
# charts. No red.
_NAV = "#002555"
_BLUE = "#007dbb"
_DGREY = "#646c6f"
_NEAR_BLACK = "#333333"

# Hybrid 7 method set + labels. "renew_glm" is our package's
# canonical name; the two alt backends are clearly marked as such.
_METHODS = [
    "renew_glm",
    "renew_glm_numba",
    "renew_glm_jax_x64",
    "glum",
    "dask_glm",
    "statsmodels_full",
    "biglm",
]
_LABELS = {
    "renew_glm":        "renew-glm (NumPy) -- this package",
    "renew_glm_numba":  "renew-glm (Numba backend)",
    "renew_glm_jax_x64": "renew-glm (JAX backend)",
    "glum":             "glum (Quantco)",
    "dask_glm":         "dask-glm",
    "statsmodels_full": "statsmodels.GLM",
    "biglm":            "biglm (R, cross-language ref)",
}
# Compact labels for the heatmap (axis ticks are tight); the .split()
# trick in the previous version collapsed the three renew-glm rows
# into "renew-glm / renew-glm / renew-glm" -- explicitly disambiguate.
_HEATMAP_LABELS = {
    "renew_glm":        "renew-glm (NumPy)",
    "renew_glm_numba":  "renew-glm (Numba)",
    "renew_glm_jax_x64": "renew-glm (JAX x64)",
    "glum":             "glum",
    "dask_glm":         "dask-glm",
    "statsmodels_full": "statsmodels",
    "biglm":            "biglm (R)",
}
_TEXT_STROKE = [pe.withStroke(linewidth=2.0, foreground="white")]


def _load_rows():
    rows = json.loads(_RESULTS_COLD.read_text())["results"]
    # bigglm-style methods sometimes only have warm-cache rows; fall
    # back per-method below if needed.
    warm = (json.loads(_RESULTS_WARM.read_text())["results"]
            if _RESULTS_WARM.exists() else [])
    return rows, warm


def _coef_at_n(rows, method, n):
    for r in rows:
        if (r.get("method") == method
                and r.get("family") == _FAMILY
                and r.get("n_rows") == n
                and r.get("coef")):
            return np.asarray(r["coef"], dtype=float)
    return None


def _best_at_ref(rows, method, n):
    for r in rows:
        if (r.get("method") == method
                and r.get("family") == _FAMILY
                and r.get("n_rows") == n
                and r.get("n_reps")):
            return r
    return None


def _color_for(m):
    """Distinct colours, with renew-glm getting the brand navy. The
    alt backends get blue / dark grey to read as 'same family'."""
    if m == "renew_glm":
        return _NAV
    if m == "renew_glm_numba":
        return _BLUE
    if m == "renew_glm_jax_x64":
        return "#5588cc"
    if m == "glum":
        return "#117733"          # green
    if m == "dask_glm":
        return "#ddcc77"          # sand
    if m == "statsmodels_full":
        return "#cc6677"          # rose
    if m == "biglm":
        return "#882255"          # plum
    return _DGREY


def _clean(ax):
    ax.grid(True, which="major", axis="both",
            alpha=0.12, linewidth=0.3, color="#000000")
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# 1) Pareto: RAM vs wall @ n=128M
# ---------------------------------------------------------------------------

def render_pareto(rows):
    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None
    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor="white")
    xs, ys, texts = [], [], []
    missing_oom = []  # methods with no n=128M data (OOM'd out)
    for m in _METHODS:
        r = _best_at_ref(rows, m, _N_REF)
        if r is None:
            missing_oom.append(m)
            continue
        x = r["mean_rss_peak_mb"]
        y = r["mean_elapsed_s"]
        col = _color_for(m)
        ax.scatter(x, y, s=130, color=col, edgecolors="#222222",
                   linewidths=0.7, marker="s", zorder=5)
        label = _LABELS[m]
        weight = "bold" if m == "renew_glm" else "normal"
        texts.append(ax.text(x, y, label, fontsize=10.5,
                             fontweight=weight, color=col,
                             path_effects=_TEXT_STROKE))
        xs.append(x)
        ys.append(y)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Peak RAM (MB)", fontsize=11)
    ax.set_ylabel("Wall time (s)", fontsize=11)
    ax.set_title(
        f"GLM cost trade-off at $\\mathit{{n}}$ = {_N_REF // 1_000_000}M "
        "(gaussian, cold)",
        color=_NAV, fontsize=12, fontweight="bold", pad=10)
    if xs and ys:
        ax.set_xlim(min(xs) * 0.5, max(xs) * 1.8)
        ax.set_ylim(min(ys) * 0.4, max(ys) * 2.5)
    _clean(ax)
    # If statsmodels (or other in-memory baselines) didn't make it to
    # n=128M, explain WHY -- otherwise the reader thinks all the named
    # alternatives fit at this scale.
    if "statsmodels_full" in missing_oom:
        ax.text(0.98, 0.02,
                "statsmodels.GLM not shown -- OOMs at $n>20M$ on a "
                "16 GB laptop\n(largest cached fit: $n=16M$, peak RAM "
                "$\\approx 8$ GB).",
                transform=ax.transAxes, fontsize=8.5, color=_DGREY,
                ha="right", va="bottom", style="italic")
    if adjust_text is not None:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color=_DGREY, lw=0.4, alpha=0.6),
            expand_points=(1.4, 1.6),
            expand_text=(1.1, 1.3),
            force_text=(0.5, 0.7),
            force_points=(0.3, 0.4),
        )
    fig.tight_layout()
    out = _OUT_DIR / "pareto.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# 2) Heatmap: pairwise |beta_i - beta_j| @ n=128M
# ---------------------------------------------------------------------------

def render_heatmap(rows):
    coefs = {m: _coef_at_n(rows, m, _N_REF) for m in _METHODS}
    present = [m for m in _METHODS if coefs[m] is not None]
    n = len(present)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 1e-16
                continue
            M[i, j] = max(
                float(np.max(np.abs(coefs[present[i]] - coefs[present[j]]))),
                1e-16,
            )
    fig, ax = plt.subplots(figsize=(8.5, 6.0), facecolor="white")
    cmap = LinearSegmentedColormap.from_list(
        "regionh_blue", ["#002555", "#007dbb", "#ccd3dd"])
    norm = LogNorm(vmin=1e-15, vmax=1e-2)
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="equal")
    short = [_HEATMAP_LABELS[m] for m in present]
    row_labels = [f"{i + 1}. {s}" for i, s in enumerate(short)]
    col_labels = [str(i + 1) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = M[i, j]
            if not np.isfinite(v):
                continue
            exp = int(np.floor(np.log10(v)))
            colour = "white" if v < 1e-7 else "#222222"
            ax.text(j, i, str(exp), ha="center", va="center",
                    fontsize=8, color=colour)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("$\\log_{10}\\, \\max_k\\,|\\,\\beta_i - \\beta_j\\,|$",
                   fontsize=9)
    ax.set_title(
        f"Cross-method coefficient agreement at $\\mathit{{n}}$ = "
        f"{_N_REF // 1_000_000}M",
        color=_NAV, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    out = _OUT_DIR / "heatmap.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# 3) Scaling: wall + RAM vs n, side-by-side
# ---------------------------------------------------------------------------

def render_scaling(rows):
    scales = [4, 8, 16, 32, 64, 128]
    fig, (ax_t, ax_r) = plt.subplots(
        1, 2, figsize=(11.5, 5.0), facecolor="white")
    for m in _METHODS:
        col = _color_for(m)
        ts, rs, ns = [], [], []
        for n in scales:
            r = _best_at_ref(rows, m, n * 1_000_000)
            if r is None:
                continue
            ts.append(r["mean_elapsed_s"])
            rs.append(r["mean_rss_peak_mb"])
            ns.append(n)
        if not ns:
            continue
        lw = 2.0 if m == "renew_glm" else 1.4
        ax_t.plot(ns, ts, "-o", color=col, label=_LABELS[m],
                  linewidth=lw, markersize=5)
        ax_r.plot(ns, rs, "-o", color=col, label=_LABELS[m],
                  linewidth=lw, markersize=5)
    for ax, ylabel, title in [
        (ax_t, "Wall time (s)", "Wall time vs $\\mathit{n}$"),
        (ax_r, "Peak RAM (MB)", "Peak RAM vs $\\mathit{n}$"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$\\mathit{n}$ (millions of rows)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, color=_NAV, fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(scales)
        ax.set_xticklabels([str(s) for s in scales])
        _clean(ax)
    # Single shared legend below both axes.
    handles, labels = ax_t.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=4, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    # Annotation calling out the OOM cliff for statsmodels / dask_glm
    # / glum (in-memory methods truncate before n=128M on the test box).
    fig.suptitle(
        "Scaling: streaming methods (renew-glm, biglm) keep going; "
        "in-memory baselines OOM",
        color=_DGREY, fontsize=10.5, y=0.995)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = _OUT_DIR / "scaling.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    if not _RESULTS_COLD.exists():
        sys.exit(f"missing {_RESULTS_COLD}")
    rows, _warm = _load_rows()
    print(f"[gen_readme_figures] {len(rows)} rows loaded from cold cache")
    render_pareto(rows)
    render_heatmap(rows)
    render_scaling(rows)
    print(f"[gen_readme_figures] done; figures in {_OUT_DIR}")


if __name__ == "__main__":
    main()
