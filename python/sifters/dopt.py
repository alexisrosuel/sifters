"""D-optimal selection on a dependence matrix (Gaussian total correlation).

:mod:`sifters.dependence` maximizes ``lambda_min(D_S)`` -- the *E-optimal*
criterion, a hard surrogate: for ``trace(D_S) = k`` and unit diagonal,
``lambda_min(D_S) >= 1 - eps`` forces every eigenvalue into
``[1 - eps, 1 + (k - 1) eps]``, hence ``det(D_S)`` close to ``1``.

This module optimizes the quantity the E-optimal criterion only controls,
namely the determinant itself.  For a Gaussian (or Gaussian copula) dependence
matrix the **total correlation** (multi-information) of the subset is exactly

    ``TC(S) = sum_i H(X_i) - H(X_S) = -0.5 * log det(D_S)``

so maximizing ``log det(D_S)`` is the literal "make the sub-joint as close as
possible to the product of its marginals" objective.  The relation to
E-optimality: if ``lambda_min(D_S) >= 1 - eps`` then
``TC(S) <= -k/2 * log(1 - eps)``, so E-optimality *implies* a small total
correlation, but not conversely (a single bad direction can leave the
determinant large while ``lambda_min`` is tiny).

Two solvers are provided:

* :func:`greedy` -- forward selection, ``O(k^2 M)`` matrix-vector work per step
  with a rank-1 update of the inverse; the standard workhorse for D-optimal
  design.  Nested family, no optimality guarantee.
* :func:`optimal` -- exact branch and bound.  It is valid because for a PSD
  matrix with unit diagonal the determinant of a principal submatrix is
  non-increasing as the set grows (``det(D_{S u j}) = det(D_S) * s_j`` with the
  Schur complement ``s_j <= 1``); ``det(D_S)`` therefore upper-bounds all its
  completions and the incumbent prunes the tree.

This is a pure-Python reference implementation on top of the same dependence
matrices; the certified Rust engine is reserved for the E-optimal criterion.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

from .dependence import dependence_matrix, project_psd

__all__ = [
    "DOptResult",
    "logdet",
    "total_correlation",
    "greedy",
    "optimal",
    "select",
    "select_from_data",
]

_EPS = 1e-300


def _logdet_slice(D: np.ndarray, idx: Iterable[int]) -> float:
    idx = list(idx)
    if not idx:
        return 0.0
    _sign, ld = np.linalg.slogdet(D[np.ix_(idx, idx)])
    return float(ld)


def logdet(D: Any, subset: Iterable[int]) -> float:
    """``log det(D_S)`` (``0.0`` for the empty set)."""
    A = np.asarray(D, dtype=np.float64)
    return _logdet_slice(A, subset)


def total_correlation(D: Any, subset: Iterable[int]) -> float:
    """Gaussian total correlation ``-0.5 * log det(D_S)`` (``>= 0`` up to noise)."""
    return -0.5 * logdet(D, subset)


@dataclass
class DOptResult:
    """Result of a D-optimal selection.

    Attributes
    ----------
    subset : list[int]
        Selected variable indices.
    k : int
        Requested size.
    logdet : float
        ``log det(D_S)`` of the selected subset (``<= 0``).
    det : float
        ``det(D_S)`` of the selected subset, in ``[0, 1]``.
    curve : dict[int, float]
        ``{size: log det}`` of the nested greedy family (when available).
    proved_optimal : bool
        True when the branch and bound exhausted the search space.
    nodes : int
        Number of branch-and-bound nodes visited.
    seconds : float
        Wall-clock time of the call.
    """

    subset: list = field(default_factory=list)
    k: int = 0
    logdet: float = 0.0
    det: float = 1.0
    curve: dict = field(default_factory=dict)
    proved_optimal: bool = False
    nodes: int = 0
    seconds: float = 0.0
    criterion: str = "D"

    def __len__(self) -> int:
        return len(self.subset)

    def to_dict(self) -> dict:
        return {
            "criterion": self.criterion,
            "k": int(self.k),
            "subset": [int(i) for i in self.subset],
            "logdet": float(self.logdet),
            "det": float(self.det),
            "total_correlation": float(-0.5 * self.logdet),
            "proved_optimal": bool(self.proved_optimal),
            "nodes": int(self.nodes),
            "seconds": float(self.seconds),
        }


def _prepare(D: Any, project: bool) -> np.ndarray:
    A = np.asarray(D, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("D must be a square dependence matrix")
    return project_psd(A) if project else 0.5 * (A + A.T)


def _first_index(D: np.ndarray) -> int:
    """Most "isolated" variable (smallest largest dependence), as in sifters."""
    m = D.shape[0]
    off = np.abs(D - np.eye(m))
    return int(np.argmin(off.max(axis=1)))


def _schur(inv: np.ndarray, c: np.ndarray) -> tuple:
    u = inv @ c
    return float(1.0 - c @ u), u


def _bordered_inverse(inv: np.ndarray, u: np.ndarray, s: float) -> np.ndarray:
    s = max(s, 1e-300)
    n = inv.shape[0]
    out = np.empty((n + 1, n + 1), dtype=np.float64)
    out[:n, :n] = inv + np.outer(u, u) / s
    out[:n, n] = -u / s
    out[n, :n] = -u / s
    out[n, n] = 1.0 / s
    return out


def greedy(D: Any, k: int, *, project: bool = True, family: bool = False):
    """Forward greedy for ``max log det(D_S)`` with ``|S| = k``.

    At each step the candidate with the largest Schur complement
    ``s_j = 1 - c_j^T D_S^{-1} c_j`` is added (it maximizes the new
    determinant).  The inverse is maintained by a rank-1 (bordered) update.

    Returns a :class:`DOptResult`; with ``family=True`` the nested
    ``{size: log det}`` curve is filled in as well.
    """
    A = _prepare(D, project)
    m = A.shape[0]
    k = int(min(max(k, 0), m))
    if k == 0:
        return DOptResult(subset=[], k=0, logdet=0.0, det=1.0)
    if m == 0:
        raise ValueError("D is empty")

    first = _first_index(A)
    S = [first]
    inv = np.array([[1.0 / A[first, first]]], dtype=np.float64)
    cumulative = 0.0
    curve = {1: 0.0} if family else {}

    while len(S) < k:
        in_set = np.zeros(m, dtype=bool)
        in_set[S] = True
        rest = np.nonzero(~in_set)[0]
        C = A[np.ix_(S, rest)]
        U = inv @ C
        schur = 1.0 - np.einsum("ij,ij->j", C, U)
        j = int(np.argmax(schur))
        s = float(schur[j])
        inv = _bordered_inverse(inv, U[:, j], s)
        cumulative = float(cumulative + np.log(max(s, 1e-300)))
        S.append(int(rest[j]))
        if family:
            curve[len(S)] = cumulative

    return DOptResult(
        subset=S, k=k, logdet=cumulative, det=float(np.exp(cumulative)), curve=curve
    )


class _Stop(Exception):
    pass


def optimal(
    D: Any,
    k: int,
    *,
    project: bool = True,
    time_limit_s: float = 30.0,
    node_limit: int = 0,
    incumbent: Optional[Iterable[int]] = None,
    tol: float = 1e-12,
) -> DOptResult:
    """Exact branch and bound for ``max log det(D_S)`` with ``|S| = k``.

    The bound is the monotonicity of the determinant of principal submatrices:
    every completion of a partial set ``S`` has ``det <= det(D_S)``, so a node is
    pruned as soon as ``log det(D_S) <= best``.  Candidates are tried in
    decreasing Schur complement order (the greedy order), which finds a strong
    incumbent on the first leaf.  The greedy solution seeds the search.

    ``time_limit_s`` (``<= 0`` for none) and ``node_limit`` (``0`` for none)
    bound the search.  If a limit is hit, ``proved_optimal`` is False; the
    returned subset is the best incumbent found so far.
    """
    A = _prepare(D, project)
    m = A.shape[0]
    k = int(min(max(k, 0), m))
    t0 = time.perf_counter()
    if k == 0:
        return DOptResult(subset=[], k=0, proved_optimal=True, seconds=time.perf_counter() - t0)

    greedy_result = greedy(A, k, project=False, family=True)
    best_subset = list(incumbent) if incumbent is not None else list(greedy_result.subset)
    best_subset = sorted(set(int(i) for i in best_subset))[:k]
    if len(best_subset) != k:
        best_subset = list(greedy_result.subset)
    best_logdet = _logdet_slice(A, best_subset)
    if greedy_result.logdet > best_logdet:
        best_logdet, best_subset = greedy_result.logdet, list(greedy_result.subset)

    state = {"nodes": 0, "proved": True}
    deadline = t0 + time_limit_s if time_limit_s and time_limit_s > 0 else None

    def rec(S: list, inv: np.ndarray, logdet_s: float, cand: list) -> None:
        nonlocal best_logdet, best_subset
        state["nodes"] += 1
        if node_limit and state["nodes"] >= node_limit:
            state["proved"] = False
            raise _Stop
        if deadline is not None and (state["nodes"] & 1023) == 0 and time.perf_counter() > deadline:
            state["proved"] = False
            raise _Stop
        if len(S) == k:
            if logdet_s > best_logdet:
                best_logdet, best_subset = logdet_s, list(S)
            return
        if not cand or len(S) + len(cand) < k:
            return
        if logdet_s <= best_logdet + tol:
            return

        scored = []
        Sidx = np.array(S, dtype=int)
        for j in cand:
            c = A[Sidx, j]
            s, u = _schur(inv, c)
            scored.append((s, int(j), c, u))
        scored.sort(key=lambda t: -t[0])

        for pos, (s, j, _c, u) in enumerate(scored):
            if s <= 0.0:
                continue
            if logdet_s <= best_logdet + tol:
                return
            nxt = [t[1] for t in scored[pos + 1:]]
            rec(S + [j], _bordered_inverse(inv, u, s), logdet_s + np.log(s), nxt)

    try:
        rec([], np.zeros((0, 0)), 0.0, list(range(m)))
    except _Stop:
        pass

    return DOptResult(
        subset=best_subset,
        k=k,
        logdet=best_logdet,
        det=float(np.exp(best_logdet)),
        curve=greedy_result.curve,
        proved_optimal=state["proved"],
        nodes=state["nodes"],
        seconds=time.perf_counter() - t0,
    )


def select(
    D: Any,
    k: int,
    *,
    exact: bool = False,
    project: bool = True,
    time_limit_s: float = 30.0,
    node_limit: int = 0,
) -> DOptResult:
    """D-optimal selection on a dependence matrix.

    ``exact=False`` runs the greedy forward selection; ``exact=True`` runs the
    branch and bound (seeded by the greedy) and sets ``proved_optimal`` when the
    space was exhausted within the budget.
    """
    if exact:
        return optimal(D, k, project=project, time_limit_s=time_limit_s, node_limit=node_limit)
    return greedy(D, k, project=project, family=True)


def select_from_data(
    X: Any,
    k: int,
    *,
    method: str = "dcor",
    exact: bool = False,
    measure_kwargs: Optional[dict] = None,
    project: bool = True,
    time_limit_s: float = 30.0,
    node_limit: int = 0,
) -> DOptResult:
    """Shortcut: ``X`` -> dependence matrix -> D-optimal selection."""
    D = dependence_matrix(X, method=method, **(measure_kwargs or {}))
    return select(
        D,
        k,
        exact=exact,
        project=project,
        time_limit_s=time_limit_s,
        node_limit=node_limit,
    )
