"""Classical column-subset-selection baselines, for comparison with `sifters`.

Why this module exists
----------------------
`sifters` maximizes ``lambda_min(R_S)`` over the ``k``-subsets ``S`` of the
columns of ``X``.  That objective is the *E-optimal* criterion of optimal design
theory, and it has classical competitors; a quality claim is only meaningful
against them.  The heuristics that ``scripts/compare_baseline.py`` used
("min-max correlation" greedy, threshold filter) are the weak ones.  This module
adds the ones a numerical-linear-algebra or applied-statistics reviewer reaches
for first:

============================  ====================================================  ==========================
baseline                      idea                                                  cost
============================  ====================================================  ==========================
``vif_topk``                  smallest variance inflation factors                   ``O(M^3)``
``qr_pivot``                  QR with column pivoting                               ``O(NMk)``
``pivoted_cholesky``          pivoted Cholesky on the Gram = **D-optimal greedy**   ``O(Mk^2)``
``rrqr_strong``               strong RRQR (Gu & Eisenstat swaps) + **guarantee**    ``O(NMk + k^2 M)``
``fedorov_exchange``          Fedorov 1-for-1 exchange on ``lambda_min``            ``O(passes k (M-k) k^3)``
``eig_greedy_forward/back``   exhaustive E-optimal greedy (eigenvalue oracle)       ``O(k M k^3)``
``uniform_sample``            uniform random subset (the degenerate CSSP sampler)   ``O(k)``
============================  ====================================================  ==========================

Two facts worth knowing before reading the numbers
--------------------------------------------------
**1. Pivoted Cholesky *is* the D-optimal greedy.**  Pivoting on the largest
diagonal of the Schur complement picks, at each step, the column maximizing
``det(D_{S u j}) / det(D_S) = 1 - c_j^T D_S^{-1} c_j``.  That is exactly what
``sifters.dopt.greedy`` computes, and what maximum-determinant / DPP map
inference computes.  On the Gram matrix it is also exactly QR with column
pivoting, because the squared residual norm after projecting out ``S`` *is* that
Schur diagonal.  So "max det" is a 1960s algorithm, available in LAPACK, free.

**2. Strong RRQR carries a guarantee on the very quantity `sifters` optimizes.**
Gu & Eisenstat's strong rank-revealing QR factorization returns ``k`` columns
``S`` with

    ``sigma_i(Z_S) >= sigma_i(Z) / sqrt(1 + f^2 k (M - k))``,  ``1 <= i <= k``

(Golub, Mahoney, Drineas & Lim, *Bridging the gap between numerical and
theoretical linear algebra*; the classical version has ``f = 1``).  With
``i = k`` and ``lambda_min(R_S) = sigma_min(Z_S)^2``, this is a *global,
deterministic, multiplicative* guarantee on the objective -- the honest yardstick
for `sifters`, whose certificate only covers the current greedy **step**.

The scale-free metric used throughout
-------------------------------------
Cauchy interlacing bounds the objective from above, independently of any
algorithm: for ``|S| = k``,

    ``lambda_min(R_S) <= lambda_{M-k+1}(R) = sigma_k(Z)^2``.

So define the **efficiency** ``eta(S) = lambda_min(R_S) / sigma_k(Z)^2 <= 1``.
``eta`` is comparable across datasets, and the RRQR guarantee becomes
``eta >= 1 / (1 + f^2 k (M-k))``.

Only numpy is required.  Nothing here is specific to `sifters`: the module is
importable on its own and is exercised by ``tests/test_cssp_baselines.py``.

Return-value convention (please respect it when adding a baseline)
------------------------------------------------------------------
* functions that answer "give me a subset" (``vif_topk``, ``uniform_sample``,
  ``rrqr_strong``, ``fedorov_exchange``) return the indices **sorted increasing**,
  so they can be fed directly to :func:`lam_min`;
* functions that answer "give me a nested path" (``qr_pivot``,
  ``pivoted_cholesky``, ``eig_greedy_forward``) return the **order of selection**,
  which is the informative output and is deliberately *not* sorted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "BaselineResult",
    "standardize",
    "correlation",
    "lam_min",
    "ceiling",
    "efficiency",
    "vif_topk",
    "uniform_sample",
    "qr_pivot",
    "pivoted_cholesky",
    "rrqr_bound",
    "rrqr_strong",
    "rrqr_growth",
    "fedorov_exchange",
    "eig_greedy_forward",
    "eig_greedy_backward",
]


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class BaselineResult:
    """Subset returned by a baseline, plus what it cost and what it guarantees.

    Attributes
    ----------
    name : str
        Identifier of the baseline.
    subset : list[int]
        Selected column indices, increasing.
    k : int
        Size of ``subset``.
    lambda_min : float
        Exact ``lambda_min(R_S)`` of the returned subset (numpy oracle).
    efficiency : float
        ``lambda_min / sigma_k(Z)^2``, in ``(0, 1]`` (see module docstring).
    seconds : float
        Wall-clock time of the selection itself.
    bound : float, optional
        Certified lower bound on ``lambda_min`` returned by the algorithm
        (only strong RRQR provides one).
    extra : dict
        Anything algorithm-specific (``f``, number of swaps, ...).
    """

    name: str
    subset: list = field(default_factory=list)
    k: int = 0
    lambda_min: float = float("nan")
    efficiency: float = float("nan")
    seconds: float = 0.0
    bound: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "k": int(self.k),
            "subset": [int(i) for i in self.subset],
            "lambda_min": float(self.lambda_min),
            "efficiency": float(self.efficiency),
            "seconds": float(self.seconds),
            "bound": None if self.bound is None else float(self.bound),
            "extra": {str(a): b for a, b in self.extra.items()},
        }


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def standardize(X: np.ndarray, center: bool = True) -> np.ndarray:
    """Center (optional) and scale each column to unit norm, as `sifters` does."""
    Z = np.asarray(X, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("X must be 2D (observations, variables)")
    if center:
        Z = Z - Z.mean(axis=0, keepdims=True)
    else:
        Z = Z.copy()
    nrm = np.linalg.norm(Z, axis=0)
    nrm[nrm == 0.0] = 1.0
    return Z / nrm


def correlation(X: np.ndarray, center: bool = True) -> np.ndarray:
    """Correlation matrix ``R = Z^T Z`` of the standardized ``X``."""
    Z = standardize(X, center=center)
    return Z.T @ Z


def lam_min(R: np.ndarray, idx: "Sequence[int] | np.ndarray") -> float:
    """Exact ``lambda_min`` of the principal submatrix ``R[S, S]``."""
    idx = list(idx)
    if not idx:
        return float("nan")
    return float(np.linalg.eigvalsh(R[np.ix_(idx, idx)])[0])


def ceiling(R: np.ndarray, k: int) -> float:
    """``lambda_{M-k+1}(R) = sigma_k(Z)^2``: the largest ``lambda_min`` any
    ``k``-subset can reach (Cauchy interlacing).  ``R`` is the ``(M, M)``
    correlation matrix, ``k`` the requested size."""
    m = R.shape[0]
    if not 1 <= k <= m:
        raise ValueError("k must be in 1..M")
    return float(np.linalg.eigvalsh(R)[m - k])


def efficiency(value: float, ceil: float) -> float:
    """``value / ceil``, guarded against a null ceiling."""
    return float(value / ceil) if ceil > 0.0 else float("nan")


# --------------------------------------------------------------------------- #
# Classical cheap heuristics
# --------------------------------------------------------------------------- #
def vif_topk(R: np.ndarray, k: int, *, ridge: float = 0.0) -> list[int]:
    """The ``k`` columns of smallest variance inflation factor, in increasing index
    order.

    ``VIF_j = (R^-1)_{jj}`` is the textbook multicollinearity diagnostic, included
    because it is what the applied-statistics audience actually runs.

    Be precise about the guarantee, because the tempting claim is **false in both
    directions**.  For PSD ``A``, ``lambda_max(A) >= max_i A_ii``, so with
    ``A = R^-1``::

        lambda_min(R) = 1 / lambda_max(R^-1) <= 1 / max_j VIF_j

    That is an **upper** bound: a large ``VIF`` *certifies* that ``R`` is badly
    conditioned, which is exactly what a collinearity diagnostic is for.  A small
    ``VIF`` certifies nothing -- ``lambda_min(R)`` can be far below ``1/max VIF``
    (measured: 0.289 against 0.676 on an AR(1) draw).  And the statement is about
    the *whole* ``R``: since ``(R_S)^-1`` is not the submatrix of ``R^-1``, no
    VIF says anything about ``lambda_min(R_S)``.

    So VIF is a one-sided screen, not a selector with guarantees.  It is included
    precisely because its weakness is instructive, and because it is dominated by
    column-pivoted QR in ``scripts/compare_cssp.py``.

    ``ridge`` is added to the diagonal before inverting (``M >= N`` makes ``R``
    singular); default ``0``.
    """
    m = R.shape[0]
    A = R if ridge == 0.0 else R + ridge * np.eye(m)
    try:
        inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(A)
    order = np.argsort(np.clip(np.diag(inv), 0.0, None), kind="stable")[:k]
    return sorted(int(i) for i in order)


def uniform_sample(m: int, k: int, rng: np.random.Generator) -> list[int]:
    """Uniform random ``k``-subset, **without** replacement.

    Worth stating explicitly: the standard randomized CSSP sampler draws columns
    with probability ``||A^(j)||^2 / ||A||_F^2`` (Drineas, Mahoney &
    Muthukrishnan).  Because `sifters` normalizes every column to unit norm
    *before* looking at the Gram, those probabilities are all equal and that
    sampler degenerates exactly into this one.  In this geometry, this is the
    CSSP sampler -- not a strawman.
    """
    k = int(min(k, m))
    return sorted(int(i) for i in rng.choice(m, size=k, replace=False))


# --------------------------------------------------------------------------- #
# QR with column pivoting, and pivoted Cholesky (the D-optimal greedy)
# --------------------------------------------------------------------------- #
def qr_pivot(Z: np.ndarray, k: int) -> list[int]:
    """Pivot order of QR with column pivoting on ``Z`` (modified Gram-Schmidt
    with a second orthogonalization pass for numerical safety).

    At each step the column of largest residual norm is chosen.  The squared
    residual norm is the Schur complement diagonal of the Gram, so this is the
    same pivot rule as :func:`pivoted_cholesky`.
    """
    _, m = Z.shape
    k = int(min(k, m))
    res = np.array(Z, dtype=np.float64, copy=True)
    colsq = np.einsum("ij,ij->j", res, res)
    avail = np.ones(m, dtype=bool)
    order: list[int] = []
    for _ in range(k):
        p = int(np.argmax(np.where(avail, colsq, -1.0)))
        if colsq[p] <= 0.0:
            break
        order.append(p)
        avail[p] = False
        v = res[:, p] / np.sqrt(colsq[p])
        for _pass in range(2):  # twice is enough
            res -= np.outer(v, v @ res)
        colsq = np.einsum("ij,ij->j", res, res)
        colsq[~avail] = 0.0
    return order


def pivoted_cholesky(R: np.ndarray, k: int) -> list[int]:
    """Pivot order of the Cholesky factorization of ``R`` with complete pivoting.

    The pivot is the largest Schur complement diagonal, i.e. the column
    maximizing ``det(R_{S u j}) / det(R_S)``: this **is** the D-optimal greedy,
    the same rule as :func:`sifters.dopt.greedy`, and the same rule as QR with
    column pivoting on a factor ``B`` with ``B^T B = R``.

    Note the genuine **degeneracy at the first step**, the exact analogue of the
    one documented for the E-optimal forward greedy: the Schur complement of the
    empty set is ``diag(R) = 1`` for *every* column, so the objective does not
    decide the first pick in exact arithmetic.  In floating point the tie is
    broken by rounding noise (``argmax`` returns whichever diagonal came out a
    few ulps larger), which is why ``sifters.dopt`` installs an explicit
    Gershgorin tie-break.  From step two on, the rule is well defined.
    """
    m = R.shape[0]
    k = int(min(k, m))
    A = np.asarray(R, dtype=np.float64)
    L = np.zeros((k, m), dtype=np.float64)
    d = np.diag(A).astype(np.float64).copy()  # Schur complement diagonal
    avail = np.ones(m, dtype=bool)
    order: list[int] = []
    for i in range(k):
        p = int(np.argmax(np.where(avail, d, -np.inf)))
        piv = float(d[p])
        if piv <= 0.0:
            break
        order.append(p)
        avail[p] = False
        row = A[p].copy()
        if i:
            row = row - L[:i, p] @ L[:i, :]
        L[i, :] = row / np.sqrt(piv)
        d = d - L[i, :] ** 2
        d[~avail] = -np.inf
        if not np.any(d > 0.0):
            break
    return order


# --------------------------------------------------------------------------- #
# Strong RRQR: the competitor that carries a global guarantee
# --------------------------------------------------------------------------- #
def rrqr_bound(R: np.ndarray, k: int, f: float = 1.0,
               sigma_k: Optional[float] = None) -> float:
    """Certified floor on ``lambda_min(R_S)`` for strong RRQR with parameter ``f``::

        sigma_min(Z_S) >= sigma_k(Z) / sqrt(1 + f^2 k (M - k))
        => lambda_min(R_S) >= sigma_k(Z)^2 / (1 + f^2 k (M - k))

    ``sigma_k`` (the ``k``-th largest singular value of ``Z``, i.e.
    ``sqrt(ceiling)``) is recomputed from ``R`` when not supplied.
    """
    m = R.shape[0]
    if not 1 <= k <= m:
        raise ValueError("k must be in 1..M")
    if sigma_k is None:
        sigma_k = float(np.sqrt(max(ceiling(R, k), 0.0)))
    return float(sigma_k**2 / (1.0 + f * f * k * (m - k)))


def rrqr_growth(Z: np.ndarray, selected: Sequence[int]) -> float:
    """Growth factor ``max |(R11^-1 R12)_{ij}|`` of the selection ``selected``.

    This is the quantity the Gu & Eisenstat bound is *conditional* on: the
    guarantee ``sigma_min(Z_S) >= sigma_k(Z)/sqrt(1 + f^2 k (M-k))`` holds only
    once this is ``<= f``.  Reporting it is what makes :func:`rrqr_bound`
    honest, since a swap loop given an unreachable ``f`` stops early.
    """
    sel = list(selected)
    _, m = Z.shape
    k = len(sel)
    rest = [j for j in range(m) if j not in set(sel)]
    if k == 0 or not rest or k > Z.shape[0]:
        return 0.0
    Q, R11 = np.linalg.qr(Z[:, sel], mode="reduced")
    try:
        W = np.linalg.solve(R11, Q.T @ Z[:, rest])
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.max(np.abs(W)))


def rrqr_strong(Z: np.ndarray, k: int, *, f: float = 1.0,
                max_swaps: int = 0) -> tuple[list[int], int, float, dict]:
    """Strong rank-revealing QR: column-pivoted QR, then Gu & Eisenstat swaps.

    The classical algorithm (Gu & Eisenstat, *Efficient algorithms for computing
    a strong rank-revealing QR factorization*, SIAM J. Sci. Comput. 17(4), 1996)
    refines a column-pivoted QR by swapping a selected column with an unselected
    one whenever the growth factor ``|(R11^-1 R12)_{ij}|`` exceeds ``f``; this
    drives the smallest singular value of the selected block above the floor of
    :func:`rrqr_bound`.  ``R11`` is re-factorized with ``numpy.linalg.qr`` after
    each candidate swap, i.e. ``O(N k^2)`` -- negligible beside the initial
    pivoted QR as long as the swap count stays small.

    A swap is accepted only when it **strictly increases** ``|det R11|``.  That
    is the termination argument (the determinant is bounded above by
    ``sigma_k(Z)^k``) and it is what the growth-factor rule is trying to achieve;
    the bare ``|W| > f`` test is not monotone and can cycle.  ``max_swaps``
    (``0`` -> ``4 k``) is a second safety net.

    The returned bound is made **unconditionally valid**: the growth factor
    actually achieved, ``g = max |(R11^-1 R12)|``, is measured on the final
    subset and the bound is evaluated at ``f_eff = max(f, g)``.  If the swap loop
    could not reach a small ``f`` (it can fail, since a swap must also increase
    the determinant), the bound degrades honestly instead of silently lying.

    Returns
    -------
    (subset, swaps, bound, extra)
        ``extra`` carries ``f``, ``f_eff``, ``growth`` and ``det_r11``.

    Note on the observed behaviour: on the synthetic regimes of
    ``scripts/compare_cssp.py`` the condition ``max |W| <= 1`` already holds for
    plain column-pivoted QR, so no swap is accepted and this returns the QRCP
    subset with the (very loose) guarantee it inherits.  The swap loop is what
    makes the bound *rigorous* below ``f = 1``; it is not a routine quality
    improvement.
    """
    _, m = Z.shape
    k = int(min(k, m))
    R = Z.T @ Z
    selected = qr_pivot(Z, k)
    if k >= m or k > Z.shape[0]:
        # No room to swap, or the selected block cannot be full rank.
        return (sorted(selected), 0, float(rrqr_bound(R, k, f=f)),
                {"f": float(f), "f_eff": float(f), "growth": 0.0, "det_r11": float("nan")})

    in_set = np.zeros(m, dtype=bool)
    in_set[selected] = True
    swaps = 0
    cap = max_swaps if max_swaps else 4 * k
    while swaps < cap:
        rest = np.nonzero(~in_set)[0]
        if rest.size == 0:
            break
        Q, R11 = np.linalg.qr(Z[:, selected], mode="reduced")
        det_cur = abs(float(np.prod(np.diag(R11))))
        try:
            W = np.linalg.solve(R11, Q.T @ Z[:, rest])
        except np.linalg.LinAlgError:
            break
        row, col = np.unravel_index(int(np.argmax(np.abs(W))), W.shape)
        i, j = int(row), int(col)
        if abs(W[i, j]) <= f:
            break
        in_col = int(rest[j])
        cand = list(selected)
        cand[i] = in_col
        _, R11c = np.linalg.qr(Z[:, cand], mode="reduced")
        if abs(float(np.prod(np.diag(R11c)))) <= det_cur * (1.0 + 1e-12):
            break  # not an improvement: stop rather than cycle
        in_set[selected[i]] = False
        in_set[in_col] = True
        selected = cand
        swaps += 1

    growth = rrqr_growth(Z, selected)
    f_eff = max(float(f), growth)
    _, R11 = np.linalg.qr(Z[:, selected], mode="reduced")
    extra = {
        "f": float(f),
        "f_eff": float(f_eff),
        "growth": float(growth),
        "det_r11": abs(float(np.prod(np.diag(R11)))),
    }
    return sorted(selected), swaps, float(rrqr_bound(R, k, f=f_eff)), extra


# --------------------------------------------------------------------------- #
# Exchange algorithm (the classical attack on E-optimality)
# --------------------------------------------------------------------------- #
def fedorov_exchange(R: np.ndarray, k: int, rng: np.random.Generator,
                     *, init: Optional[Sequence[int]] = None, passes: int = 20,
                     restarts: int = 1, tol: float = 1e-12,
                     on_step: Optional[Callable[[int, float], None]] = None,
                     ) -> tuple[list[int], dict]:
    """Fedorov 1-for-1 exchange on ``lambda_min``: repeatedly swap the pair
    ``(i in S, j not in S)`` giving the largest improvement, until no pair
    improves.

    This is the classical exchange heuristic of optimal design theory (Fedorov
    1972; it is what the ``swap_passes`` option of ``sifters.select`` implements).
    It reaches a local optimum by construction -- a fair "classical competitor",
    uncertified, at the cost of one ``lambda_min`` per pair.

    ``restarts`` random initial subsets are tried and the best local optimum is
    kept; ``init`` overrides the first start (when it has exactly ``k`` distinct
    elements).
    """
    m = R.shape[0]
    k = int(min(k, m))
    best_subset: list[int] = []
    best_value = -np.inf
    stats = {"evaluations": 0, "passes": 0, "restarts": 0, "improvements": 0}

    for r in range(max(1, restarts)):
        current: list[int]
        if init is not None and r == 0 and len(set(int(i) for i in init)) == k:
            current = sorted(int(i) for i in init)
        else:
            current = uniform_sample(m, k, rng)
        current_value = lam_min(R, current)
        stats["evaluations"] += 1
        for _pass in range(passes):
            stats["passes"] += 1
            in_set = np.zeros(m, dtype=bool)
            in_set[current] = True
            rest = np.nonzero(~in_set)[0]
            best_pair: Optional[tuple[int, int]] = None
            best_pair_value = current_value
            for i in current:
                base = [c for c in current if c != i]
                for jj in rest:
                    j = int(jj)
                    value = lam_min(R, base + [j])
                    stats["evaluations"] += 1
                    if value > best_pair_value + tol:
                        best_pair_value = value
                        best_pair = (int(i), j)
            if best_pair is None:
                break
            i, j = best_pair
            current = sorted([c for c in current if c != i] + [j])
            current_value = best_pair_value
            stats["improvements"] += 1
            if on_step is not None:
                on_step(_pass, current_value)
        stats["restarts"] += 1
        if current_value > best_value + tol:
            best_value = current_value
            best_subset = list(current)
    return sorted(best_subset), stats


# --------------------------------------------------------------------------- #
# Exhaustive E-optimal greedy: the eigen-oracle reference
# --------------------------------------------------------------------------- #
def eig_greedy_forward(R: np.ndarray, kmax: int) -> tuple[list[int], dict]:
    """Nested forward E-optimal greedy with an exact ``eigvalsh`` oracle at every
    candidate step: the "naive numpy" reference of ``scripts/bench_numpy.py``,
    made reusable.

    Returns the order of addition and a dict with ``evaluations`` and the
    ``{K: lambda_min}`` curve.  Step 1 uses the Gershgorin rule of `sifters`
    (smallest maximum absolute correlation), because every singleton has
    ``lambda_min = 1`` and the objective is degenerate there.
    """
    m = R.shape[0]
    kmax = int(min(kmax, m))
    off = np.abs(R - np.eye(m))
    first = int(np.argmin(off.max(axis=1)))
    selected = [first]
    curve = {1: 1.0}
    evaluations = 1
    rest = [j for j in range(m) if j != first]
    for _ in range(1, kmax):
        values = np.array([lam_min(R, selected + [j]) for j in rest], dtype=np.float64)
        evaluations += len(rest)
        pos = int(np.argmax(values))
        selected.append(rest.pop(pos))
        curve[len(selected)] = float(values[pos])
    return selected, {"evaluations": evaluations, "curve": curve}


def eig_greedy_backward(R: np.ndarray, kmin: int) -> tuple[list[int], dict]:
    """Nested backward E-optimal greedy with an exact ``eigvalsh`` oracle: from
    all ``M`` columns, repeatedly remove the one whose removal maximizes
    ``lambda_min`` of what remains.

    Returns the removal order and a dict with ``evaluations`` and the
    ``{K: lambda_min}`` curve.
    """
    m = R.shape[0]
    kmin = int(max(1, min(kmin, m)))
    current = list(range(m))
    curve = {m: lam_min(R, current)}
    evaluations = 1
    order: list[int] = []
    while len(current) > kmin:
        values = np.array([lam_min(R, [c for c in current if c != i])
                           for i in current], dtype=np.float64)
        evaluations += len(current)
        removed = current.pop(int(np.argmax(values)))
        order.append(removed)
        curve[len(current)] = float(np.max(values))
    return order, {"evaluations": evaluations, "curve": curve}
