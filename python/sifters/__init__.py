"""sifters - spectral feature selection, powered by Rust (native PyO3 binding).

**`sifters` picks, for every size K, the best-conditioned subset of K variables** -
the one whose correlation matrix has the largest smallest eigenvalue
``lambda_min(R_S)``, i.e. the subset that stays as far as possible from
singularity. In optimal design theory this criterion is called *E-optimal*
(maximize the minimum eigenvalue of the information matrix).

The numerical core is written in safe, multi-threaded Rust; this module is a
native extension: no data crosses a subprocess boundary, numpy arrays are read
through the buffer protocol, and the GIL is released during the computation.

Example
-------
>>> import numpy as np, sifters
>>> X = np.random.default_rng(0).normal(size=(500, 300))   # (observations, variables)
>>> r = sifters.select(X, k=30)
>>> r.subset[:5]
[3, 7, 11, 29, 41]
>>> r.lambda_min
0.123...
>>> r.curve[10]          # lambda_min of the family at K = 10
0.29...

The family is **nested**: ``r.subset_at(k)`` returns the subset of any visited
size ``k`` without recomputation (mode ``"backward"``) or after completion
(mode ``"forward"``).

The criterion is not restricted to the linear link: because the engine only
needs a symmetric PSD matrix with a unit diagonal, any **dependence matrix**
(distance correlation, HSIC, normalized mutual information, Gaussian copula)
can replace the correlation matrix through :func:`select_from_dependence`, and
:mod:`sifters.dopt` offers the D-optimal criterion (maximization of
``log det``, i.e. minimization of the Gaussian total correlation). See
:func:`dependence_matrix` and :func:`select_dependence`.
"""

from __future__ import annotations

import array
import os
from typing import Any, Callable, Optional

from . import _sifters
from ._sifters import SiftersError, Selection, Step, __version__

__all__ = [
    "select",
    "path",
    "curve",
    "as_matrix",
    "dependence_matrix",
    "select_from_dependence",
    "select_dependence",
    "dopt",
    "dependence",
    "Selection",
    "Step",
    "SiftersError",
    "version",
    "__version__",
]

#: Submodules loaded on demand (they need numpy, which is optional for the
#: core engine).
_LAZY_SUBMODULES = ("dependence", "dopt")

#: Progress callback signature: receives a :class:`Step` and returns ``False``
#: (or any falsy object) to interrupt the walk.  ``None`` keeps going.
ProgressCallback = Callable[[Step], Optional[bool]]

REMOVED = {
    "binary_path": (
        "binary_path() was removed: the package no longer calls the `sifters` "
        "executable through a subprocess, the engine is a native extension."
    ),
    "Result": (
        "Result was replaced by sifters.Selection (same main attributes: "
        "k, subset, lambda_min, curve, seconds, order, initial_subset, direction, "
        "m_total, subset_at())."
    ),
}


def __getattr__(name: str) -> Any:
    if name in REMOVED:
        raise AttributeError(REMOVED[name])
    if name in _LAZY_SUBMODULES:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'sifters' has no attribute '{name}'")


def version() -> str:
    """Engine version (same as ``sifters.__version__``)."""
    return __version__


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _as_buffer(X: Any) -> tuple[Any, Optional[tuple[int, int]], int]:
    """Convert ``X`` into a usable buffer and return ``(buffer, shape, M)``.

    numpy is used when available (it absorbs lists, DataFrames, tensors,
    ``float32``, non-contiguous views...).  Without numpy, ``X`` must already
    expose a 2D ``float64`` buffer, or be a sequence of sequences (flattened
    here).
    """
    if isinstance(X, (str, bytes, os.PathLike)):
        raise TypeError(
            "passing a file path is not supported: load the data yourself "
            "(numpy.loadtxt, pandas.read_csv, ...) and pass the array"
        )
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        a = np.asarray(X)
        if a.ndim != 2:
            raise ValueError(
                f"X must be 2D (observations, variables), got {a.ndim}D"
            )
        a = np.ascontiguousarray(a, dtype=np.float64)
        return a, None, int(a.shape[1])

    if isinstance(X, memoryview) and X.ndim == 2:
        return X, None, int(X.shape[1])

    # numpy-free fallback: sequence of sequences -> 1D buffer (row-major).
    try:
        rows = [list(r) for r in X]  # type: ignore[union-attr]
    except TypeError:
        return X, None, 0
    if not rows:
        raise ValueError("X is empty")
    n, m = len(rows), len(rows[0])
    if any(len(r) != m for r in rows):
        raise ValueError("all rows of X must have the same length")
    flat = array.array("d", [float(v) for r in rows for v in r])
    return flat, (n, m), m


def as_matrix(X: Any) -> Any:
    """Return a contiguous 2D ``float64`` view (C order) of ``X``.

    Useful to check or explicitly prepare an input; the selection functions
    already perform this conversion.  numpy is used when installed.
    """
    buf, shape, _ = _as_buffer(X)
    if shape is not None:  # numpy-free fallback: return the flat buffer
        return buf
    try:
        import numpy as np

        return np.asarray(buf, dtype=np.float64)
    except ImportError:  # pragma: no cover - environment dependent
        return buf


def _auto_direction(m: int, k: int) -> str:
    """``forward`` is both better and much cheaper than ``backward`` as soon as
    the requested K is small compared with M."""
    if m <= 0:
        return "forward"
    return "forward" if k * 3 < m else "backward"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select(
    X: Any,
    k: int,
    *,
    direction: str = "auto",
    center: bool = True,
    verify: bool = True,
    eval: str = "auto",
    representation: str = "auto",
    low_rank: int = 4,
    tol: float = 1e-10,
    iters_warm: int = 60,
    iters_cold: int = 400,
    max_exact: int = 0,
    batch: int = 8,
    prefilter: bool = False,
    forward_top: int = 0,
    forward_first: Optional[int] = None,
    forward_seeds: int = 1,
    exact: bool = False,
    exact_time_s: float = 30.0,
    exact_max_evals: int = 0,
    swap_passes: int = 0,
    swap_top: int = 8,
    swap_from: Optional[int] = None,
    mem_budget_mb: int = 4096,
    block_rows: int = 64,
    threads: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
) -> Selection:
    """Select ``k`` variables out of ``X`` (E-optimal criterion).

    Parameters
    ----------
    X : array-like, shape ``(n_observations, n_variables)``
        Converted to contiguous ``float64``; columns are centered then scaled to
        unit norm (``center=False`` gives the cosine matrix).
    k : int
        Number of variables to keep.
    direction : {"auto", "backward", "forward"}
        ``"auto"`` (default) picks ``"forward"`` when ``3k < M``, else
        ``"backward"``.  ``"backward"`` builds the full certified family (costly
        for large ``M``); ``"forward"`` grows greedily.
    center : bool
        Center the columns (correlation) or not (cosine).
    verify : bool
        Re-validate ``lambda_min`` of the retained subset with a strict cold
        Lanczos (tolerance ``1e-13``).  In ``"backward"`` mode, also re-validate
        every step of the curve.
    eval : {"auto", "direct", "inverse"}
        Candidate evaluation.  ``"inverse"`` maintains ``R^-1`` and converges in
        ~20 iterations instead of 100-150 (positive definite correlation).
    representation : {"auto", "packed", "implicit"}
        Materialized correlation (``packed``) or applied implicitly
        ``Z_S^T (Z_S x)`` (``implicit``, optimal when ``N << k``).
    low_rank : int
        Number of eigenpairs used by the Temple bound (default 4).
    tol, iters_warm, iters_cold : Lanczos settings.
    max_exact : int
        Cap on candidates evaluated exactly per step (0 = certified).
    forward_top : int
        In ``forward`` mode, width of the secular prefilter.  ``0`` (default)
        evaluates every candidate: the greedy walk is then exactly E-optimal.
        Ignored when ``prefilter=False``; otherwise sets the width (``0`` -> 16).
    prefilter : bool
        Enable the secular prefilter in ``forward`` mode: only the ``forward_top``
        (16 by default) best candidates of the secular score are evaluated
        exactly.  **Disabled by default**: the forward greedy is then exactly
        E-optimal.  Nothing is enabled automatically based on size.  Turning it
        on speeds up large ``M`` a lot, at the price of a few percent of
        ``lambda_min`` (heuristic) - see the README.
    forward_first : int, optional
        Imposed first variable (``forward`` mode).
    forward_seeds : int
        Number of starting variables tried in ``forward`` mode (default 1).
        The first start is the historical heuristic, then come the best
        variables in the sense of the **one-step lookahead** - ``lambda_min`` at
        ``k = 2``, i.e. ``1 - min_j |r_ij|`` - without duplicates.  The retained
        family is the one with the largest final ``lambda_min``.  ``1``
        reproduces the historical greedy, and since the set of starts is nested,
        increasing ``forward_seeds`` can never degrade the final result; ``M``
        (or more) tries every start.  The cost is multiplied by the number of
        starts, and ``forward_first`` disables multi-start.  No effect in
        ``direction="backward"``.
    exact : bool
        Prove optimality at the requested ``k`` with a branch and bound over all
        subsets of that exact size, after the greedy walk provided the incumbent
        (default ``False``).  The returned ``subset``/``lambda_min`` are then the
        optimum, and ``proved_optimal`` tells whether the proof completed within
        the budget; otherwise ``gap_certified`` bounds the remaining relative
        improvement.  The nested family (``curve``/``steps``) still comes from
        the greedy walk that seeded the search.  A good incumbent (raise
        ``forward_seeds``) is what makes the proof cheap.
    exact_time_s : float
        Wall-clock budget in seconds for the exact search (``<= 0`` = unlimited).
    exact_max_evals : int
        Cap on ``lambda_min`` evaluations for the exact search (``0`` = no cap).
    swap_passes, swap_top, swap_from : 1-for-1 local swaps (``backward`` mode),
        to correct the myopia of the greedy walk.
    mem_budget_mb, block_rows : performance / memory.
    threads : int, optional
        Size of the dedicated ``rayon`` pool.  ``None`` (or ``0``) uses the
        global pool, sized on the number of available cores.
    progress : callable, optional
        Called with a :class:`Step` after each step.  Returning ``False``
        interrupts the walk cleanly (the partial result is returned); returning
        ``None`` keeps going.  ``Ctrl-C`` is intercepted at the same moment and
        raises ``KeyboardInterrupt``.

    Returns
    -------
    :class:`Selection`

    Examples
    --------
    >>> r = sifters.select(X, k=30)
    >>> r.subset, r.lambda_min
    ([3, 7, ...], 0.123...)
    >>> r = sifters.select(X, k=30, exact=True)   # proven optimum at k = 30
    >>> r.proved_optimal, r.gap_certified
    (True, 0.0)
    """
    if not isinstance(k, (int,)) or isinstance(k, bool):
        try:
            k = int(k)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise TypeError("k must be an integer") from exc
    buf, shape, m = _as_buffer(X)
    if direction == "auto":
        direction = _auto_direction(m, int(k))
    return _sifters.run(
        buf,
        shape=shape,
        direction=direction,
        k=int(k),
        center=center,
        verify=verify,
        eval=eval,
        representation=representation,
        low_rank=low_rank,
        tol=tol,
        iters_warm=iters_warm,
        iters_cold=iters_cold,
        max_exact=max_exact,
        batch=batch,
        prefilter=prefilter,
        forward_top=forward_top,
        forward_first=forward_first,
        forward_seeds=forward_seeds,
        exact=exact,
        exact_time_s=exact_time_s,
        exact_max_evals=exact_max_evals,
        swap_passes=swap_passes,
        swap_top=swap_top,
        swap_from=swap_from,
        mem_budget_mb=mem_budget_mb,
        block_rows=block_rows,
        threads=threads,
        progress=progress,
    )


def path(
    X: Any,
    *,
    kmin: int = 1,
    kmax: Optional[int] = None,
    direction: str = "backward",
    center: bool = True,
    verify: bool = True,
    eval: str = "auto",
    representation: str = "auto",
    low_rank: int = 4,
    tol: float = 1e-10,
    iters_warm: int = 60,
    iters_cold: int = 400,
    max_exact: int = 0,
    batch: int = 8,
    prefilter: bool = False,
    forward_top: int = 0,
    forward_first: Optional[int] = None,
    forward_seeds: int = 1,
    swap_passes: int = 0,
    swap_top: int = 8,
    swap_from: Optional[int] = None,
    mem_budget_mb: int = 4096,
    block_rows: int = 64,
    threads: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
) -> Selection:
    """Build a nested family of subsets.

    In ``direction="backward"`` (default), it starts from all ``M`` variables and
    eliminates down to ``kmin``: the family ``S_M > ... > S_kmin`` is
    **certified** (every step is proven optimal).  In ``direction="forward"``, it
    starts from a singleton and adds up to ``kmax``.

    Returns a :class:`Selection` whose :attr:`~Selection.curve` gives
    ``lambda_min`` for every visited K and whose :meth:`~Selection.subset_at`
    rebuilds any subset.

    The parameters are those of :func:`select`.
    """
    buf, shape, _ = _as_buffer(X)
    return _sifters.run(
        buf,
        shape=shape,
        direction=direction,
        kmin=int(kmin),
        kmax=None if kmax is None else int(kmax),
        center=center,
        verify=verify,
        eval=eval,
        representation=representation,
        low_rank=low_rank,
        tol=tol,
        iters_warm=iters_warm,
        iters_cold=iters_cold,
        max_exact=max_exact,
        batch=batch,
        prefilter=prefilter,
        forward_top=forward_top,
        forward_first=forward_first,
        forward_seeds=forward_seeds,
        swap_passes=swap_passes,
        swap_top=swap_top,
        swap_from=swap_from,
        mem_budget_mb=mem_budget_mb,
        block_rows=block_rows,
        threads=threads,
        progress=progress,
    )


def curve(X: Any, **kwargs: Any) -> dict[int, float]:
    """Shortcut: return the ``{K: lambda_min}`` curve directly.

    Accepts the same keywords as :func:`path`.
    """
    return dict(path(X, **kwargs).curve)


# ---------------------------------------------------------------------------
# Nonlinear / non-Gaussian dependence (lazy: needs numpy)
# ---------------------------------------------------------------------------

def dependence_matrix(X: Any, *, method: str = "dcor", **kwargs: Any) -> Any:
    """Pairwise dependence matrix ``D`` (unit diagonal) for any kind of link.

    ``method`` is one of ``"linear"`` (Pearson), ``"rank"`` (Gaussian copula /
    normal scores, handles monotone nonlinear links), ``"dcor"`` (distance
    correlation), ``"hsic"`` (normalized RBF HSIC) or ``"nmi"`` (normalized
    mutual information).  All of them are ``0`` for independent variables, so
    ``D = I`` means "the joint is the product of the marginals".

    See :mod:`sifters.dependence`.
    """
    from .dependence import dependence_matrix as _dm

    return _dm(X, method=method, **kwargs)


def select_from_dependence(D: Any, k: int, *, criterion: str = "E", **kwargs: Any):
    """Select ``k`` variables from a dependence matrix ``D``.

    ``criterion="E"`` (default) maximizes ``lambda_min(D_S)`` with the certified
    Rust engine (returns a :class:`Selection`).  ``criterion="D"`` maximizes
    ``log det(D_S)``, i.e. minimizes the Gaussian total correlation, with the
    Python reference solver (returns a :class:`sifters.dopt.DOptResult`).

    See :mod:`sifters.dependence` and :mod:`sifters.dopt`.
    """
    if criterion == "E":
        from .dependence import select_from_dependence as _sd

        return _sd(D, k, **kwargs)
    if criterion == "D":
        from .dopt import select as _dopt

        return _dopt(D, k, **kwargs)
    raise ValueError("criterion must be 'E' or 'D'")


def select_dependence(
    X: Any,
    k: int,
    *,
    method: str = "dcor",
    criterion: str = "E",
    measure_kwargs: Optional[dict] = None,
    **kwargs: Any,
):
    """End-to-end selection robust to nonlinear links.

    Computes the dependence matrix of ``X`` with ``method`` (see
    :func:`dependence_matrix`) and selects ``k`` variables with the ``"E"``
    (``lambda_min``, certified) or ``"D"`` (``log det``) criterion.
    """
    from .dependence import dependence_matrix as _dm

    D = _dm(X, method=method, **(measure_kwargs or {}))
    return select_from_dependence(D, k, criterion=criterion, **kwargs)
