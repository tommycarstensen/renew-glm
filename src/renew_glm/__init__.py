"""renew_glm: Python port of Luo & Song (2020) renewable streaming GLM.

A renewable estimator for generalized linear models with streaming data
sets. Algorithm by Lan Luo and Peter X.-K. Song:

  Luo L. & Song P.X.-K. (2020). "Renewable Estimation and Incremental
  Inference in Generalized Linear Models with Streaming Data Sets."
  JRSS-B 82(1), 69-97. https://doi.org/10.1111/rssb.12352

Original R reference implementation:
  https://github.com/luolsph/RenewGLM_pkg

Python port: Tommy Carstensen (ORCID 0000-0002-3672-9931).
License: GPL-2.0-or-later (matches the original R package).

Public API:
  RenewGLM -- pure NumPy implementation. The only backend in this
              minimalist release; carry your own JAX/Numba/Dask
              wrapper if you need them.
"""

from importlib.metadata import PackageNotFoundError, version

from renew_glm._irls import RenewGLM

__all__ = ["RenewGLM"]

try:
    __version__ = version("renew-glm")
except PackageNotFoundError:
    # Imported from a source tree that was never installed, e.g. a checkout on
    # sys.path. Without this guard the import itself raises.
    __version__ = "0.0.0.dev0"
