"""Compare `sifters` (Rust) with two naive heuristics in Python.

1. `minmax greedy`: we iteratively add the variable whose maximum correlation
   (in absolute value) with the already kept variables is the lowest
   -- this is exactly the "greedy" version of the threshold filter.
2. `threshold filter` (the requested approach): we scan the variables and drop
   those that are correlated above `x %` with an already selected variable.
   We vary `x` to obtain a curve (K, lambda_min).

Metrics: exact lambda_min (numpy) of the kept subset, and selection
time.

Usage:
    python3 compare_baseline.py --n 500 --m 400 --rho 0.6 --k 10,25,50,100
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np

import sifters


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def make_data(kind: str, n: int, m: int, rho: float, rho_out: float, blocks: int,
              rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "equi":
        g = rng.normal(size=(n, 1))
        return np.sqrt(rho) * g + np.sqrt(1.0 - rho) * rng.normal(size=(n, m))
    if kind == "ar":
        e = rng.normal(size=(n, m))
        x = np.empty_like(e)
        x[:, 0] = e[:, 0]
        for j in range(1, m):
            x[:, j] = rho * x[:, j - 1] + np.sqrt(1 - rho**2) * e[:, j]
        return x
    if kind == "blocks":
        common = rng.normal(size=(n, 1))
        fac = rng.normal(size=(n, blocks))
        b = np.minimum(np.arange(m) * blocks // m, blocks - 1)
        g = np.sqrt(rho_out) * common + np.sqrt(rho) * fac[:, b]
        return g + np.sqrt(max(0.0, 1.0 - rho)) * rng.normal(size=(n, m))
    if kind == "factor":
        load = rng.normal(size=(m, rank))
        return rng.normal(size=(n, rank)) @ load.T + rng.normal(size=(n, m))
    return rng.normal(size=(n, m))


def correlation(X: np.ndarray) -> np.ndarray:
    Z = X - X.mean(axis=0)
    nrm = np.linalg.norm(Z, axis=0)
    nrm[nrm == 0] = 1.0
    Z = Z / nrm
    return Z.T @ Z


def lambda_min(R: np.ndarray, idx) -> float:
    idx = list(idx)
    sub = R[np.ix_(idx, idx)]
    return float(np.linalg.eigvalsh(sub)[0])


# --------------------------------------------------------------------------- #
# Naive heuristics
# --------------------------------------------------------------------------- #
def minmax_greedy_path(R: np.ndarray, kmax: int) -> tuple[Callable[[int], list[int]], float]:
    """Nested path: at each step we add the variable least correlated with the
    already kept variables (min of the max of |correlation|)."""
    t0 = time.perf_counter()
    m = R.shape[0]
    A = np.abs(R).copy()
    np.fill_diagonal(A, 0.0)
    selected: list[int] = []
    score = np.full(m, -np.inf)          # max |corr| with the kept set
    remaining = np.ones(m, dtype=bool)
    order: list[int] = []
    for _ in range(min(kmax, m)):
        # first variable: the one whose max correlation is the lowest
        best = int(np.argmin(np.where(remaining, np.maximum(score, 0.0), np.inf)))
        selected.append(best)
        remaining[best] = False
        order.append(best)
        score = np.maximum(score, A[best])
        score[best] = np.inf if not remaining[best] else score[best]
    elapsed = time.perf_counter() - t0

    def subset_for(k: int) -> list[int]:
        return order[:k]

    return subset_for, elapsed


def threshold_filter(R: np.ndarray, t: float, kmax: int) -> list[int]:
    """Threshold filter: keep a variable if |corr| < t with all already kept
    variables (scan in natural order)."""
    m = R.shape[0]
    A = np.abs(R)
    kept: list[int] = []
    for j in range(m):
        if len(kept) >= kmax:
            break
        if not kept or np.all(A[j, kept] < t):
            kept.append(j)
    return kept


def threshold_path(R: np.ndarray, kmax: int, grid: int = 60):
    """Curve (K, lambda_min) of the threshold filter by sweeping t, plus the time."""
    t0 = time.perf_counter()
    lo, hi = 0.0, float(np.abs(R - np.eye(R.shape[0])).max())
    best_for_k: dict[int, list[int]] = {}
    thresholds = np.linspace(lo, hi, grid)
    for t in thresholds:
        kept = threshold_filter(R, float(t), kmax)
        k = len(kept)
        if k and k not in best_for_k:
            best_for_k[k] = kept
    elapsed = time.perf_counter() - t0
    return best_for_k, elapsed


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="blocks")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--m", type=int, default=400)
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--rho-out", type=float, default=0.05)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", default="10,25,50,100,200")
    ap.add_argument("--kmin", type=int, default=None,
                    help="lower bound of the sifters path (def. min(k))")
    args = ap.parse_args()

    ks = [int(v) for v in args.k.split(",")]
    kmin = args.kmin if args.kmin is not None else min(ks)

    X = make_data(args.kind, args.n, args.m, args.rho, args.rho_out,
                  args.blocks, args.rank, args.seed)
    R = correlation(X)
    print(f"# {args.kind}: N={args.n}, M={args.m}, rho={args.rho}, "
          f"rho_out={args.rho_out}, blocks={args.blocks}, rank={args.rank}, seed={args.seed}")

    # ---- sifters (Rust): one call per K, in both directions ----
    mv_back: dict[int, tuple[list[int], float, float]] = {}
    mv_fwd: dict[int, tuple[list[int], float, float]] = {}
    for k in ks:
        t0 = time.perf_counter()
        r = sifters.select(X, k=k, direction="backward")
        mv_back[k] = (r.subset, r.lambda_min, time.perf_counter() - t0)
        t0 = time.perf_counter()
        r = sifters.select(X, k=k, direction="forward")
        mv_fwd[k] = (r.subset, r.lambda_min, time.perf_counter() - t0)
    print("# sifters: one call per K (backward = elimination from M ; forward = greedy addition)")

    # ---- naive heuristics ----
    kmax = max(ks)
    subset_minmax, t_minmax = minmax_greedy_path(R, kmax)
    thr_subsets, t_thr = threshold_path(R, kmax)

    # ---- table ----
    print()
    def thr_for(k: int) -> tuple[list[int], int]:
        if k in thr_subsets:
            return thr_subsets[k], k
        if not thr_subsets:
            return [], 0
        near = min(thr_subsets, key=lambda kk: abs(kk - k))
        return thr_subsets[near], near

    hdr = (f"{'K':>5} | {'sifters back':>10} {'back t':>7} | {'sifters fwd':>10} {'fwd t':>7} | "
           f"{'minmax':>10} {'mm t':>6} | {'threshold':>10} {'K_got':>5}")
    print()
    print(hdr)
    print("-" * len(hdr))
    for k in ks:
        sb, lb, tb = mv_back[k]
        sf, lf, tf = mv_fwd[k]
        s = subset_minmax(k)
        lm = lambda_min(R, s) if len(s) == k else float("nan")
        st, kt = thr_for(k)
        lt = lambda_min(R, st) if st else float("nan")
        print(f"{k:>5} | {lb:>10.6f} {tb:>7.2f} | {lf:>10.6f} {tf:>7.2f} | "
              f"{lm:>10.6f} {t_minmax:>6.3f} | {lt:>10.6f} {kt:>5}")

    print()
    print(f"total time: minmax greedy {t_minmax:.3f} s | threshold sweep {t_thr:.3f} s")
    print("(sifters times include the strict validation of the kept subset)")


if __name__ == "__main__":
    main()
