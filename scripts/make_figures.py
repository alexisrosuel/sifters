"""Generate the README figures (docs/img/fig_qualite.png and docs/img/fig_performance.png).

Usage:
    python3 make_figures.py [qualite] [performance]

The `sifters` curves come from the Rust backend (Python binding); the naive
heuristics and all `lambda_min` values are computed in pure numpy.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sifters
from compare_baseline import correlation, lambda_min, make_data, minmax_greedy_path, threshold_filter

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

C_SIFTERS_FWD = "#1b6ca8"
C_SIFTERS_BWD = "#7fb3d5"
C_MINMAX = "#d9534f"
C_SEUIL = "#e8a33d"


# --------------------------------------------------------------------------- #
# Figure 1: quality (lambda_min at fixed K)
# --------------------------------------------------------------------------- #
def figure_qualite(kmax: int = 80) -> None:
    jeux: list[tuple[str, dict[str, Any]]] = [
        ("blocks (6 blocks, rho=0.6)", dict(kind="blocks", n=500, m=250, rho=0.6, rho_out=0.05, blocks=6, rank=4, seed=1)),
        ("AR(1) rho=0.9", dict(kind="ar", n=400, m=250, rho=0.9, rho_out=0.0, blocks=1, rank=4, seed=1)),
        ("factors (rank 20)", dict(kind="factor", n=500, m=250, rho=0.6, rho_out=0.0, blocks=1, rank=20, seed=1)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=False)
    ks = np.arange(2, kmax + 1)

    for ax, (titre, kw) in zip(axes, jeux):
        X = make_data(**kw)
        R = correlation(X)

        # --- sifters: one call for the forward family, one for the backward ---
        fwd = sifters.path(X, direction="forward", kmax=kmax)
        bwd = sifters.path(X, direction="backward", kmin=1)
        v_fwd = [lambda_min(R, fwd.subset_at(int(k))) for k in ks]
        v_bwd = [lambda_min(R, bwd.subset_at(int(k))) for k in ks]

        # --- baseline 1: min-max correlation greedy ---
        sub_mm, t_mm = minmax_greedy_path(R, kmax)
        v_mm = [lambda_min(R, sub_mm(int(k))) for k in ks]

        # --- baseline 2: threshold filter (sweep) ---
        best = {}
        for t in np.linspace(0.0, 1.0, 400):
            kept = threshold_filter(R, float(t), kmax)
            if 2 <= len(kept) <= kmax and len(kept) not in best:
                best[len(kept)] = kept
        ks_t = sorted(best)
        v_t = [lambda_min(R, best[k]) for k in ks_t]

        ax.plot(ks, v_fwd, color=C_SIFTERS_FWD, lw=2.2, label="sifters (forward, secular)")
        ax.plot(ks, v_bwd, color=C_SIFTERS_BWD, lw=1.6, ls="--", label="sifters (backward, certified)")
        ax.plot(ks, v_mm, color=C_MINMAX, lw=1.6, label="min-max corr. greedy")
        ax.plot(ks_t, v_t, color=C_SEUIL, lw=1.6, ls=":", label="threshold filter")
        ax.set_title(titre)
        ax.set_xlabel("K (number of variables)")
        ax.set_yscale("log")
        gain = np.mean([(f - b) / b for f, b in zip(v_fwd[8:40], v_mm[8:40])]) * 100
        ax.text(
            0.03, 0.06, f"mean gain vs min-max: +{gain:.0f} %",
            transform=ax.transAxes, fontsize=9, color=C_SIFTERS_FWD,
        )
    axes[0].set_ylabel(r"$\lambda_{\min}$ exact (numpy)")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle(
        "Quality: same number of variables, $\\lambda_{\\min}$ larger",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(IMG / "fig_qualite.png", bbox_inches="tight")
    plt.close(fig)
    print("docs/img/fig_qualite.png written")


# --------------------------------------------------------------------------- #
# Figure 2: performance
# --------------------------------------------------------------------------- #
def figure_performance() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    # (a) scaling of the forward mode: the prefilter is an explicit choice
    ms = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
    ms_exact = [m for m in ms if m <= 3200]
    temps = []
    for m in ms:
        X = make_data("blocks", 100, m, 0.6, 0.05, 8, 4, 1)
        t0 = time.perf_counter()
        sifters.path(X, direction="forward", kmax=50, prefilter=True)
        temps.append(time.perf_counter() - t0)
    temps_exact = []
    for m in ms_exact:
        X = make_data("blocks", 100, m, 0.6, 0.05, 8, 4, 1)
        t0 = time.perf_counter()
        sifters.path(X, direction="forward", kmax=50, prefilter=False)
        temps_exact.append(time.perf_counter() - t0)
    axes[0].loglog(ms, temps, "o-", color=C_SIFTERS_FWD, lw=2.0,
                   label="with prefilter (top-16), K=50")
    axes[0].loglog(ms_exact, temps_exact, "^--", color=C_SIFTERS_BWD, lw=1.8,
                   label="without prefilter (exact)")
    axes[0].annotate(
        "explicit prefilter: ~16 candidates/step\n-> linear cost in M",
        xy=(1600, min(temps)), xytext=(0.06, 0.72), textcoords="axes fraction",
        fontsize=8.5, color="dimgrey",
        arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
    )
    axes[0].set_xlabel("M (number of variables), N=100")
    axes[0].set_ylabel("path time (s)")
    axes[0].set_title("(a) Scaling of the forward mode")
    axes[0].legend(fontsize=8.5, loc="lower right")

    # (b) direct vs inverse evaluation (backward path)
    ms_b = [100, 150, 200, 250]
    t_direct: list[float] = []
    t_inv: list[float] = []
    n_direct: list[float] = []
    n_inv: list[float] = []
    for m in ms_b:
        X = make_data("blocks", 600, m, 0.6, 0.05, 6, 4, 1)
        for mode, tt, nn in (("direct", t_direct, n_direct), ("inverse", t_inv, n_inv)):
            t0 = time.perf_counter()
            r = sifters.path(X, direction="backward", kmin=1, eval=mode, verify=False)
            tt.append(time.perf_counter() - t0)
            nn.append(r.total_lanczos_iters / max(1, r.total_exact_evals))
    x = np.arange(len(ms_b))
    w = 0.38
    axes[1].bar(x - w / 2, t_direct, w, color=C_MINMAX, label="direct matvec on R")
    axes[1].bar(x + w / 2, t_inv, w, color=C_SIFTERS_FWD, label="evaluation via the inverse")
    axes[1].set_xticks(x, [f"M={m}" for m in ms_b])
    axes[1].set_ylabel("full backward path time (s)")
    axes[1].set_title("(b) Iterations per candidate / 3 (inverse)")
    for xi, (td, ti, nd, ni) in enumerate(zip(t_direct, t_inv, n_direct, n_inv)):
        axes[1].text(xi - w / 2, td, f"{td:.1f}s\n{nd:.0f} it.", ha="center", va="bottom", fontsize=8)
        axes[1].text(xi + w / 2, ti, f"{ti:.1f}s\n{ni:.0f} it.", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, max(t_direct) * 1.45)
    axes[1].legend(fontsize=8.5, loc="upper left")
    axes[1].grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(IMG / "fig_performance.png", bbox_inches="tight")
    plt.close(fig)
    print("docs/img/fig_performance.png written")


if __name__ == "__main__":
    import sys

    choix = sys.argv[1:] or ["qualite", "performance"]
    if "qualite" in choix:
        figure_qualite()
    if "performance" in choix:
        figure_performance()
