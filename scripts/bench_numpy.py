"""`sifters` (Rust) versus a naive *full Python / numpy* implementation of the same
problem: the exact E-optimal greedy.

The competitor solves the **same** objective as `sifters.path(X, direction="forward")` --
build a nested family by adding, at each step, the variable that maximizes
`lambda_min` of the correlation matrix of the subset -- but without any
algorithmic trick: at each step it re-evaluates *all* remaining candidates by
recomputing a complete `numpy.linalg.eigvalsh` decomposition (dimension K+1).
That is `O(K * M)` decompositions of size `~K`.

Three series are plotted:

* `sifters` **with prefilter** (`prefilter=True`, top-16): the forward greedy
  only evaluates the 16 best candidates of the secular score;
* `sifters` **without prefilter** (`prefilter=False`, the default): the same set of
  candidates as the naive one at each step -- we check that `lambda_min` is then
  identical to the naive one bit for bit, which makes the time comparison strictly
  fair;
* **naive numpy**: the `eigvalsh` reference.

The time gap between series 1 and 2 therefore measures exactly what the bounds
and the prefilter bring (and what they cost in optimality).

Usage:
    python3 bench_numpy.py
    python3 bench_numpy.py --m 400,800,1600

Outputs:
    docs/img/fig_temps_vs_numpy.png   README figure (time + quality)
    docs/img/temps_vs_numpy.json      raw values (reproducible)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sifters
from compare_baseline import correlation, make_data, lambda_min

DOCS = Path(__file__).resolve().parent.parent / "docs"
IMG = DOCS / "img"
IMG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

C_PREFILTRE = "#1b6ca8"  # sifters with prefilter (top-16)
C_EXACT = "#7fb3d5"      # sifters without prefilter (same as the naive one)
C_NUMPY = "#d9534f"      # naive numpy


# --------------------------------------------------------------------------- #
# The competitor: naive E-optimal greedy, 100 % numpy
# --------------------------------------------------------------------------- #
def naive_forward_greedy(R: np.ndarray, k: int, first: int) -> tuple[list[int], int]:
    """Exact E-optimal greedy, naive version.

    At each step, each remaining candidate is evaluated by recomputing `lambda_min`
    with `numpy.linalg.eigvalsh` on the enlarged submatrix.  `first` fixes the
    starting variable (the one chosen by `sifters`, to compare the same family).
    Returns (subset, number of spectral decompositions).
    """
    m = R.shape[0]
    sel = [int(first)]
    ndec = 0
    for _ in range(1, k):
        best_j, best_v = -1, -np.inf
        for j in range(m):
            if j in sel:
                continue
            idx = sel + [j]
            v = float(np.linalg.eigvalsh(R[np.ix_(idx, idx)])[0])
            ndec += 1
            if v > best_v:
                best_v, best_j = v, j
        sel.append(best_j)
    return sel, ndec


def time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def run_sifters(X, k, prefilter: bool):
    t, r = time_call(
        lambda: sifters.select(X, k=k, direction="forward", prefilter=prefilter)
    )
    return t, r


def run_naive_from_R(R, k, first):
    t, (sel, ndec) = time_call(lambda: naive_forward_greedy(R, k, first))
    return t, sel, ndec, lambda_min(R, sel)


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
def measure_one(X, k):
    """Return the times/qualities of the three series for a given (X, k).

    On the naive side we *also* time the construction of the correlation matrix,
    as `sifters` does it itself from `X`: the budgets are comparable.
    """
    t_prefiltre, r_prefiltre = run_sifters(X, k, prefilter=True)
    t_exact, r_exact = run_sifters(X, k, prefilter=False)
    first = r_exact.initial_subset[0] if r_exact.initial_subset else 0
    t0 = time.perf_counter()
    R = correlation(X)
    build = time.perf_counter() - t0
    t_np, _, ndec, lam_np = run_naive_from_R(R, k, first)
    return dict(
        k=k,
        t_rust=t_prefiltre,
        t_rust_exact=t_exact,
        t_numpy=t_np + build,
        ndec=ndec,
        lam_rust=r_prefiltre.lambda_min,
        lam_rust_exact=r_exact.lambda_min,
        lam_numpy=lam_np,
        first=first,
    )


def sweep_m(ms, n, k, kind, seed):
    rows = []
    for m in ms:
        X = make_data(kind, n, m, 0.6, 0.05, 8, 4, seed)
        row = measure_one(X, k)
        row["m"] = m
        rows.append(row)
        print(f"[M] M={m:>6} K={k:>3} | prefilter {row['t_rust']:7.3f}s | exact {row['t_rust_exact']:7.3f}s "
              f"| numpy {row['t_numpy']:8.3f}s | x{row['t_numpy'] / row['t_rust']:6.1f} vs prefilter "
              f"| {row['ndec']:>8} eigvalsh")
    return rows


def sweep_k(ks, n, m, kind, seed):
    X = make_data(kind, n, m, 0.6, 0.05, 8, 4, seed)
    rows = []
    for k in ks:
        row = measure_one(X, k)
        row["m"] = m
        rows.append(row)
        print(f"[K] M={m:>6} K={k:>3} | prefilter {row['t_rust']:7.3f}s | exact {row['t_rust_exact']:7.3f}s "
              f"| numpy {row['t_numpy']:8.3f}s | x{row['t_numpy'] / row['t_rust']:6.1f} vs prefilter "
              f"| {row['ndec']:>8} eigvalsh")
    return rows


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def figure(by_m, by_k, n_m: int, n_k: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2))

    # (a) time vs M
    ms = np.array([r["m"] for r in by_m])
    ax = axes[0]
    ax.loglog(ms, [r["t_rust"] for r in by_m], "o-", color=C_PREFILTRE, lw=2.2, ms=5,
              label="sifters with prefilter (top-16)")
    ax.loglog(ms, [r["t_rust_exact"] for r in by_m], "^--", color=C_EXACT, lw=1.8, ms=5,
              label="sifters without prefilter (exact, same candidates as the naive one)")
    ax.loglog(ms, [r["t_numpy"] for r in by_m], "s-", color=C_NUMPY, lw=2.0, ms=5,
              label="naive numpy (eigvalsh)")
    for r in [r for r in by_m if r["t_numpy"] / r["t_rust"] >= 1.5][-2:]:
        ax.annotate(f"x{r['t_numpy'] / r['t_rust']:.0f}",
                    xy=(r["m"], r["t_numpy"]), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8.5, color="#4a934a", fontweight="bold")
    ax.set_xlabel(f"M (number of variables), N={n_m}, K={by_m[0]['k']}")
    ax.set_ylabel("selection time (s)")
    ax.set_title("(a) Time vs number of variables")

    # (b) time vs K
    ks = np.array([r["k"] for r in by_k])
    ax = axes[1]
    ax.loglog(ks, [r["t_rust"] for r in by_k], "o-", color=C_PREFILTRE, lw=2.2, ms=5)
    ax.loglog(ks, [r["t_rust_exact"] for r in by_k], "^--", color=C_EXACT, lw=1.8, ms=5)
    ax.loglog(ks, [r["t_numpy"] for r in by_k], "s-", color=C_NUMPY, lw=2.0, ms=5)
    ax.set_xlabel(f"K (variables kept), N={n_k}, M={by_k[0]['m']}")
    ax.set_ylabel("selection time (s)")
    ax.set_title("(b) Time vs subset size")
    for r in [r for r in by_k if r["t_numpy"] / r["t_rust"] >= 1.5][-1:]:
        ax.annotate(f"x{r['t_numpy'] / r['t_rust']:.0f}",
                    xy=(r["k"], r["t_numpy"]), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8.5, color="#4a934a", fontweight="bold")

    # (c) quality: lambda_min (the naive one and sifters without prefilter coincide)
    ax = axes[2]
    ax.semilogx(ms, [100 * r["lam_rust"] for r in by_m], "o-", color=C_PREFILTRE, lw=2.2, ms=5)
    ax.semilogx(ms, [100 * r["lam_numpy"] for r in by_m], "s-", color=C_NUMPY, lw=2.0, ms=5)
    ax.set_xlabel(f"M (number of variables), N={n_m}, K={by_m[0]['k']}")
    ax.set_ylabel(r"$100\,\lambda_{\min}$ of the selected subset")
    ax.set_title("(c) Quality: what the prefilter costs")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Same objective (E-optimal greedy), same hardware: Rust vs naive numpy",
                 fontsize=12, y=1.02)  # noqa: E501
    fig.tight_layout()
    fig.savefig(IMG / "fig_temps_vs_numpy.png", bbox_inches="tight")
    plt.close(fig)
    print("docs/img/fig_temps_vs_numpy.png written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="200,400,800,1600,3200,6400",
                    help="list of M for sweep (a) and (c)")
    ap.add_argument("--k", default="10,20,40,80,120",
                    help="list of K for sweep (b)")
    ap.add_argument("--n-m", type=int, default=500, help="N of sweep (a)/(c)")
    ap.add_argument("--n-k", type=int, default=500, help="N of sweep (b)")
    ap.add_argument("--m-fixed", type=int, default=800, help="fixed M of sweep (b)")
    ap.add_argument("--kind", default="blocks")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--replot", action="store_true",
                    help="regenerate the figure from docs/img/temps_vs_numpy.json")
    args = ap.parse_args()

    if args.replot:
        data = json.loads((IMG / "temps_vs_numpy.json").read_text())
        figure(data["by_m"], data["by_k"], data["n_m"], data["n_k"])
        return

    ms = [int(v) for v in args.m.split(",")]
    ks = [int(v) for v in args.k.split(",")]
    print(f"# {args.kind}: sweep (a)/(c) N={args.n_m}, K=50 ; sweep (b) N={args.n_k}, "
          f"M={args.m_fixed} ; seed={args.seed}")
    by_m = sweep_m(ms, args.n_m, 50, args.kind, args.seed)
    by_k = sweep_k(ks, args.n_k, args.m_fixed, args.kind, args.seed)

    payload = dict(by_m=by_m, by_k=by_k, n_m=args.n_m, n_k=args.n_k,
                   kind=args.kind, seed=args.seed)
    (IMG / "temps_vs_numpy.json").write_text(json.dumps(payload, indent=2))
    print("docs/img/temps_vs_numpy.json written")

    # equivalence check: sifters exact == naive numpy
    worst = max(abs(r["lam_rust_exact"] - r["lam_numpy"]) for r in by_m + by_k)
    print(f"# equivalence sifters exact / naive numpy: max gap |lambda_min| = {worst:.2e}")

    figure(by_m, by_k, args.n_m, args.n_k)


if __name__ == "__main__":
    main()
