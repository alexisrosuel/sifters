"""Dependence matrices for nonlinear variable selection.

``sifters`` selects the subset ``S`` of ``k`` variables that maximizes
``lambda_min(R_S)``, the smallest eigenvalue of the (correlation) matrix of the
subset.  That criterion only sees **linear** dependence: two variables linked by
``y = x**2`` look independent to it.

The engine itself never needs a correlation matrix: it only needs a symmetric
positive semi-definite matrix with unit diagonal, i.e. the Gram matrix of
normalized columns.  This module builds such a *dependence matrix* ``D`` from any
pairwise dependence measure (``D_ij = 0`` meaning "independent"), factors it as
``D = B^T B`` and hands ``B`` to the existing engine.  Maximizing
``lambda_min(D_S)`` then means: make the sub-joint as close as possible to the
product of its marginals, in the E-optimal sense, for **any kind of link**.

Available measures (``dependence_matrix(X, method=...)``):

===========  ============================================================
``linear``   Pearson correlation (the historical behaviour).
``rank``     Normal scores / Gaussian copula: any *monotone* link.
``dcor``     Distance correlation: any link (0 iff independent).
``hsic``     Normalized HSIC with an RBF kernel (PSD by construction).
``nmi``      Normalized mutual information from 2D histograms.
===========  ============================================================

All of them return a symmetric matrix with a unit diagonal (up to
``project_psd``).  ``dcor`` and ``nmi`` are not guaranteed PSD, so
``select_from_dependence`` clips the negative eigenvalues of ``D`` first; HSIC
and the two linear measures are PSD by construction.

Costs: ``dcor`` and ``hsic`` are ``O(M^2 N^2)`` in time and ``O(M N^2)`` in
memory; ``nmi`` is ``O(M^2 N)``; ``rank`` and ``linear`` are ``O(M^2 N)``.

Invariance. With the default settings every measure is invariant to a
per-feature **positive affine rescaling** ``x_j -> a_j x_j + b_j`` (``a_j > 0``):
the selection therefore does not depend on the units or the offset of the
features. Two caveats. ``hsic`` with an explicit ``sigma`` loses that property
(the default median heuristic adapts the bandwidth to each feature). The two
*signed* measures (``linear`` and ``rank``) change when a feature is
**reflected** (``a_j < 0``), whereas ``dcor``, ``hsic`` and ``nmi`` measure the
magnitude of the dependence and are reflection-invariant.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

__all__ = [
    "METHODS",
    "dependence_matrix",
    "distance_correlation",
    "normalized_mutual_information",
    "hsic",
    "normal_scores",
    "project_psd",
    "factor",
    "select_from_dependence",
    "select_dependence",
]

METHODS = ("linear", "rank", "dcor", "hsic", "nmi")


# ---------------------------------------------------------------------------
# Small numerical helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _as_2d(X: Any) -> np.ndarray:
    a = np.asarray(X, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"X must be 2D (observations, variables), got {a.ndim}D")
    if a.shape[0] < 2:
        raise ValueError("X must have at least 2 observations")
    if not np.all(np.isfinite(a)):
        raise ValueError("X contains non-finite values")
    return a


def _average_rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), without scipy."""
    values, inv = np.unique(x, return_inverse=True)
    counts = np.bincount(inv, minlength=values.size)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = 0.5 * (start + 1 + cum)
    return avg[inv].astype(np.float64)


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)

    lo, hi = 0.02425, 1.0 - 0.02425
    q = np.where(p < lo, p, 1.0 - p)
    t = np.sqrt(-2.0 * np.log(q))
    num = ((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]
    den = (((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0
    tail = num / den
    tail = np.where(p < lo, tail, -tail)

    qm = p - 0.5
    r = qm * qm
    centre = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * qm
    den2 = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    centre = centre / den2
    return np.where((p >= lo) & (p <= hi), centre, tail)


def normal_scores(X: Any) -> np.ndarray:
    """Map every column to normal scores (Gaussian copula / rank transform).

    The result is the ``Phi^-1`` transform of the normalized ranks, then
    standardized.  Feeding it to :func:`sifters.select` handles any *monotone*
    nonlinear link (``log``, ``exp``, ``sqrt``, ...) exactly.
    """
    a = _as_2d(X)
    n, m = a.shape
    out = np.empty((n, m), dtype=np.float64)
    for j in range(m):
        r = _average_rank(a[:, j])
        z = _norm_ppf((r - 0.5) / n)
        sd = z.std()
        if sd > 0:
            z = (z - z.mean()) / sd
        out[:, j] = z
    return out


# ---------------------------------------------------------------------------
# Pairwise dependence measures
# ---------------------------------------------------------------------------

def _centered_distance(x: np.ndarray) -> np.ndarray:
    """Double-centered distance matrix ``A`` of one variable."""
    a = np.abs(x[:, None] - x[None, :])
    return a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean()


def distance_correlation(X: Any) -> np.ndarray:
    """Pairwise distance correlation matrix (Szekely et al., biased estimate).

    ``D_ij`` lies in ``[0, 1]``, equals ``0`` if and only if the two columns are
    independent, and is invariant to any (possibly nonlinear) link and to
    monotone rescaling.  This is the default recommendation for an unknown link.

    Complexity ``O(M^2 N^2)`` in time and ``O(M N^2)`` in memory.
    """
    a = _as_2d(X)
    m = a.shape[1]
    A = [_centered_distance(a[:, j]) for j in range(m)]
    dvar = np.array([float((Ai * Ai).mean()) for Ai in A])
    D = np.eye(m, dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            if dvar[i] <= 0.0 or dvar[j] <= 0.0:
                continue
            dcov2 = float((A[i] * A[j]).mean())
            value = dcov2 / np.sqrt(dvar[i] * dvar[j])
            D[i, j] = D[j, i] = float(np.clip(np.sqrt(max(value, 0.0)), 0.0, 1.0))
    return D


def hsic(
    X: Any,
    *,
    sigma: Optional[Any] = None,
    kernel: str = "rbf",
) -> np.ndarray:
    """Normalized HSIC matrix (RBF kernel by default).

    HSIC is the squared Hilbert-Schmidt norm of the centered cross-covariance
    operator.  With a characteristic kernel it is ``0`` if and only if the two
    columns are independent.  The matrix of the inner products of the centered
    kernel matrices is a Gram matrix, hence **PSD by construction**: no
    projection is needed.

    ``sigma`` is the RBF bandwidth; ``None`` uses the per-variable median
    heuristic (``median`` of the pairwise distances).
    """
    a = _as_2d(X)
    n, m = a.shape
    if kernel not in ("rbf", "linear"):
        raise ValueError("kernel must be 'rbf' or 'linear'")

    H = np.eye(n) - np.ones((n, n)) / n
    centred = []
    for j in range(m):
        x = a[:, j]
        if kernel == "linear":
            K = np.outer(x, x)
        else:
            d = np.abs(x[:, None] - x[None, :])
            s = float(np.median(d)) if sigma is None else float(sigma)
            if s <= 0:
                s = 1.0
            K = np.exp(-(d * d) / (2.0 * s * s))
        C = H @ K @ H
        centred.append(C)

    diag = np.array([float((C * C).sum()) for C in centred])
    D = np.eye(m, dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            if diag[i] <= 0.0 or diag[j] <= 0.0:
                continue
            value = float((centred[i] * centred[j]).sum()) / np.sqrt(diag[i] * diag[j])
            D[i, j] = D[j, i] = float(np.clip(value, 0.0, 1.0))
    return D


def normalized_mutual_information(
    X: Any,
    *,
    bins: Any = "auto",
    normalize: str = "sqrt",
) -> np.ndarray:
    """Pairwise normalized mutual information from 2D histograms.

    ``I(X; Y) / sqrt(H(X) H(Y))`` (``normalize="sqrt"``, the default) or
    ``I(X; Y) / min(H(X), H(Y))`` (``normalize="min"``); both lie in ``[0, 1]``.
    Cheap (``O(M^2 N)``) but binning-dependent; prefer ``dcor`` or ``hsic`` when
    the sample is large enough.
    """
    a = _as_2d(X)
    n, m = a.shape
    if normalize not in ("sqrt", "min"):
        raise ValueError("normalize must be 'sqrt' or 'min'")

    edges, marg = [], []
    for j in range(m):
        e = np.histogram_bin_edges(a[:, j], bins=bins)
        if e.size < 2:
            e = np.array([a[:, j].min(), a[:, j].min() + 1.0])
        p = np.histogram(a[:, j], bins=e)[0].astype(np.float64)
        p /= p.sum()
        p = p[p > 0]
        edges.append(e)
        marg.append(float(-(p * np.log(p)).sum()))

    D = np.eye(m, dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            joint, _, _ = np.histogram2d(a[:, i], a[:, j], bins=[edges[i], edges[j]])
            joint /= joint.sum()
            pi = joint.sum(axis=1, keepdims=True)
            pj = joint.sum(axis=0, keepdims=True)
            nz = joint > 0
            mi = float((joint[nz] * np.log(joint[nz] / (pi @ pj)[nz])).sum())
            mi = max(mi, 0.0)
            if normalize == "sqrt":
                den = np.sqrt(marg[i] * marg[j])
            else:
                den = min(marg[i], marg[j])
            value = mi / den if den > 0 else 0.0
            D[i, j] = D[j, i] = float(np.clip(value, 0.0, 1.0))
    return D


def dependence_matrix(X: Any, *, method: str = "dcor", **kwargs: Any) -> np.ndarray:
    """Build the pairwise dependence matrix ``D`` (unit diagonal).

    ``method`` is one of :data:`METHODS`.  Extra keywords go to the underlying
    measure (``bins`` for ``nmi``, ``sigma`` / ``kernel`` for ``hsic``).
    """
    if method == "linear":
        a = _as_2d(X)
        Z = a - a.mean(axis=0, keepdims=True)
        nrm = np.linalg.norm(Z, axis=0)
        nrm[nrm == 0] = 1.0
        Z = Z / nrm
        return Z.T @ Z
    if method == "rank":
        Z = normal_scores(X)
        return Z.T @ Z / Z.shape[0]
    if method == "dcor":
        return distance_correlation(X, **kwargs)
    if method == "hsic":
        return hsic(X, **kwargs)
    if method == "nmi":
        return normalized_mutual_information(X, **kwargs)
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


# ---------------------------------------------------------------------------
# Dependence matrix -> ordinary sifters selection
# ---------------------------------------------------------------------------

def project_psd(D: Any, *, eps: float = 1e-10) -> np.ndarray:
    """Nearest PSD matrix with a unit diagonal (eigenvalue clipping).

    ``D`` is symmetrized, its negative eigenvalues are raised to
    ``eps * max_eigenvalue`` and its diagonal is rescaled back to ``1``.  The
    rescaling is a positive-diagonal congruence, so it preserves the PSD
    property.  Matrices that are already PSD with a unit diagonal (``hsic``,
    ``linear``, ``rank``) are returned almost unchanged.
    """
    A = np.asarray(D, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("D must be a square matrix")
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    floor = eps * max(float(w[-1]), 1.0) if w.size else eps
    w = np.maximum(w, floor)
    Ap = (V * w) @ V.T
    d = np.sqrt(np.clip(np.diag(Ap).copy(), 1e-300, None))
    Ap = Ap / np.outer(d, d)
    np.fill_diagonal(Ap, 1.0)
    return 0.5 * (Ap + Ap.T)


def factor(D: Any, *, eps: float = 1e-10, project: bool = True) -> np.ndarray:
    """Factor ``D = B^T B`` with unit-norm columns of ``B``.

    ``B`` is ``M x M`` and can be passed directly to :func:`sifters.select` with
    ``center=False``: the engine then sees exactly ``D`` as its Gram matrix.
    With ``project=False`` (a matrix already known to be PSD) the eigenvalues
    are only clipped at zero.
    """
    A = np.asarray(D, dtype=np.float64)
    A = project_psd(A, eps=eps) if project else 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 0.0)
    return np.sqrt(w)[:, None] * V.T


def select_from_dependence(D: Any, k: int, *, project: bool = True, **kwargs: Any):
    """E-optimal selection on an arbitrary dependence matrix ``D``.

    ``D`` is projected to the nearest PSD unit-diagonal matrix (unless
    ``project=False``), factored as ``D = B^T B``, and ``B`` is passed to
    :func:`sifters.select` with ``center=False``.  All other keywords are
    forwarded to :func:`sifters.select` (``direction``, ``exact``, ...).

    Returns the usual :class:`sifters.Selection`.  ``lambda_min`` is then the
    smallest eigenvalue of ``D_S`` and is still certified by the Rust engine.
    """
    from . import select as _select

    kwargs.pop("center", None)
    B = factor(D, project=project)
    return _select(B, k, center=False, **kwargs)


def select_dependence(
    X: Any,
    k: int,
    *,
    method: str = "dcor",
    measure_kwargs: Optional[dict] = None,
    project: bool = True,
    **kwargs: Any,
):
    """End-to-end shortcut: ``X`` -> dependence matrix -> E-optimal selection.

    Equivalent to ``select_from_dependence(dependence_matrix(X, method=...),
    k, ...)``.  See :func:`dependence_matrix` for the measures and
    :func:`select_from_dependence` for the selection keywords.
    """
    D = dependence_matrix(X, method=method, **(measure_kwargs or {}))
    return select_from_dependence(D, k, project=project, **kwargs)
