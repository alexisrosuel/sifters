"""Benchmark the exact mode (`select(..., exact=True)`) across sizes.

For each (structure, M, K) the script measures:

* the multi-start greedy walk (the incumbent), and the gain the exact search
  brings over it;
* the wall-clock time of the exact search, its number of `lambda_min`
  evaluations, whether optimality was **proven** within the budget, and the
  certified gap when it was not.

The results are written to ``bench/out/exact.json``; pass ``--figure`` to also
render ``docs/img/fig_exact.png`` (requires matplotlib).

Usage
-----
    python bench/exact.py
    python bench/exact.py --figure --time-budget 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sifters

#: Dense AR(1) correlation with a strictly positive spectrum.
def _ar1(n: int, m: int, rho: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    i = np.arange(m)
    C = rho ** np.abs(i[:, None] - i[None, :])
    w, V = np.linalg.eigh(C)
    C = V @ np.diag(np.clip(w, 1e-9, None)) @ V.T
    return rng.normal(size=(n, m)) @ np.linalg.cholesky(C).T


def make_data(kind: str, n: int, m: int, seed: int) -> np.ndarray:
    """'iid' (flat spectrum, hard for the bound) or 'ar1' (rho = 0.9)."""
    if kind == "iid":
        return np.random.default_rng(seed).normal(size=(n, m))
    if kind == "ar1":
        return _ar1(n, m, 0.9, seed)
    raise ValueError(f"unknown structure: {kind}")


def sweep(
    kinds: tuple[str, ...],
    ms: tuple[int, ...],
    ks: tuple[int, ...],
    n: int,
    seeds: int,
    time_budget: float,
) -> list[dict]:
    rows: list[dict] = []
    for kind in kinds:
        for m in ms:
            for k in ks:
                if k >= m:
                    continue
                greedy_s = exact_s = float("nan")
                proved = False
                gap = evals = 0
                gain = 0.0
                for seed in range(seeds):
                    X = make_data(kind, n, m, seed)
                    n_starts = min(m, 8)
                    t0 = time.perf_counter()
                    g = sifters.select(X, k=k, direction="forward", forward_seeds=n_starts)
                    greedy_s = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    e = sifters.select(
                        X,
                        k=k,
                        direction="forward",
                        forward_seeds=n_starts,
                        exact=True,
                        exact_time_s=time_budget,
                    )
                    exact_s = time.perf_counter() - t0
                    proved = bool(e.proved_optimal)
                    gap = float(e.gap_certified)
                    evals = int(e.exact_evals)
                    gain += 100.0 * (e.lambda_min - g.lambda_min) / g.lambda_min
                rows.append(
                    {
                        "structure": kind,
                        "n": n,
                        "m": m,
                        "k": k,
                        "seeds": seeds,
                        "greedy_s": greedy_s,
                        "exact_s": exact_s,
                        "evals": evals,
                        "proved": proved,
                        "gap": gap,
                        "gain_pct": gain / seeds,
                    }
                )
                print(
                    f"{kind:4s} M={m:4d} K={k:3d} | greedy {greedy_s:7.3f}s "
                    f"exact {exact_s:8.3f}s {evals:9d} evals | "
                    f"proved={str(proved):5s} gap={gap:7.4f} gain={gain / seeds:5.2f}%",
                    flush=True,
                )
    return rows


def figure(rows: list[dict], path: Path) -> None:
    """Time of the exact search against M, one curve per K, log scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = sorted({r["structure"] for r in rows})
    ks = sorted({r["k"] for r in rows})
    fig, axes = plt.subplots(1, len(kinds), figsize=(6.0 * len(kinds), 4.6), sharey=True)
    if len(kinds) == 1:
        axes = [axes]
    for ax, kind in zip(axes, kinds):
        for k in ks:
            pts = sorted(
                (r["m"], r["exact_s"], r["proved"])
                for r in rows
                if r["structure"] == kind and r["k"] == k
            )
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"K = {k}")
            bad = [(p[0], p[1]) for p in pts if not p[2]]
            if bad:
                ax.plot(
                    [p[0] for p in bad],
                    [p[1] for p in bad],
                    "X",
                    ms=11,
                    color="crimson",
                    label="budget exhausted",
                )
        ax.set_yscale("log")
        ax.set_xlabel("M (candidate variables)")
        ax.set_title(kind)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("exact search time (s)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Certified exact mode: time to prove optimality at K")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    print(f"figure -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structures", nargs="+", default=["iid", "ar1"])
    ap.add_argument("--ms", nargs="+", type=int, default=[20, 30, 40, 60, 80])
    ap.add_argument("--ks", nargs="+", type=int, default=[5, 8, 12])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--time-budget", type=float, default=120.0)
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()

    rows = sweep(
        tuple(args.structures),
        tuple(args.ms),
        tuple(args.ks),
        args.n,
        args.seeds,
        args.time_budget,
    )
    out = Path(__file__).resolve().parent / "out" / "exact.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"json -> {out}")
    if args.figure:
        figure(rows, Path(__file__).resolve().parent.parent / "docs" / "img" / "fig_exact.png")


if __name__ == "__main__":
    main()
