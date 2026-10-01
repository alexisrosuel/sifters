"""Genere les figures du README (docs/fig_qualite.png et docs/fig_performance.png).

Usage :
    python3 make_figures.py [qualite] [performance]

Les courbes `minvp` proviennent du backend Rust (binding Python) ; les heuristiques
naives et toutes les valeurs de `lambda_min` sont calculees en numpy pur.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import minvp
from compare_baseline import correlation, lambda_min, make_data, minmax_greedy_path, threshold_filter

DOCS = Path(__file__).resolve().parent.parent / "docs"
DOCS.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

C_MINVP_FWD = "#1b6ca8"
C_MINVP_BWD = "#7fb3d5"
C_MINMAX = "#d9534f"
C_SEUIL = "#e8a33d"


# --------------------------------------------------------------------------- #
# Figure 1 : qualite (lambda_min a K fixe)
# --------------------------------------------------------------------------- #
def figure_qualite(kmax: int = 80) -> None:
    jeux = [
        ("blocs (6 blocs, rho=0.6)", dict(kind="blocks", n=500, m=250, rho=0.6, rho_out=0.05, blocks=6, rank=4, seed=1)),
        ("AR(1) rho=0.9", dict(kind="ar", n=400, m=250, rho=0.9, rho_out=0.0, blocks=1, rank=4, seed=1)),
        ("facteurs (rang 20)", dict(kind="factor", n=500, m=250, rho=0.6, rho_out=0.0, blocks=1, rank=20, seed=1)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=False)
    ks = np.arange(2, kmax + 1)

    for ax, (titre, kw) in zip(axes, jeux):
        X = make_data(**kw)
        R = correlation(X)

        # --- minvp : un appel pour la famille avant, un pour l'arriere ---
        fwd = minvp.path(X, direction="forward", kmax=kmax)
        bwd = minvp.path(X, direction="backward", kmin=1)
        v_fwd = [lambda_min(R, fwd.subset_at(int(k))) for k in ks]
        v_bwd = [lambda_min(R, bwd.subset_at(int(k))) for k in ks]

        # --- baseline 1 : glouton min-max correlation ---
        sub_mm, t_mm = minmax_greedy_path(R, kmax)
        v_mm = [lambda_min(R, sub_mm(int(k))) for k in ks]

        # --- baseline 2 : filtre par seuil (balayage) ---
        best = {}
        for t in np.linspace(0.0, 1.0, 400):
            kept = threshold_filter(R, float(t), kmax)
            if 2 <= len(kept) <= kmax and len(kept) not in best:
                best[len(kept)] = kept
        ks_t = sorted(best)
        v_t = [lambda_min(R, best[k]) for k in ks_t]

        ax.plot(ks, v_fwd, color=C_MINVP_FWD, lw=2.2, label="minvp (avant, séculaire)")
        ax.plot(ks, v_bwd, color=C_MINVP_BWD, lw=1.6, ls="--", label="minvp (arrière, certifié)")
        ax.plot(ks, v_mm, color=C_MINMAX, lw=1.6, label="glouton min-max corr.")
        ax.plot(ks_t, v_t, color=C_SEUIL, lw=1.6, ls=":", label="filtre par seuil")
        ax.set_title(titre)
        ax.set_xlabel("K (nombre de variables)")
        ax.set_yscale("log")
        gain = np.mean([(f - b) / b for f, b in zip(v_fwd[8:40], v_mm[8:40])]) * 100
        ax.text(
            0.03, 0.06, f"gain moyen vs min-max : +{gain:.0f} %",
            transform=ax.transAxes, fontsize=9, color=C_MINVP_FWD,
        )
    axes[0].set_ylabel(r"$\lambda_{\min}$ exact (numpy)")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle(
        "Qualité : même nombre de variables, $\\lambda_{\\min}$ plus grand",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(DOCS / "fig_qualite.png", bbox_inches="tight")
    plt.close(fig)
    print("docs/fig_qualite.png ecrit")


# --------------------------------------------------------------------------- #
# Figure 2 : performance
# --------------------------------------------------------------------------- #
def figure_performance() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    # (a) passage a l'echelle du mode avant : le pre-filtre est un choix explicite
    ms = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
    ms_exact = [m for m in ms if m <= 3200]
    temps = []
    for m in ms:
        X = make_data("blocks", 100, m, 0.6, 0.05, 8, 4, 1)
        t0 = time.perf_counter()
        minvp.path(X, direction="forward", kmax=50, prefilter=True)
        temps.append(time.perf_counter() - t0)
    temps_exact = []
    for m in ms_exact:
        X = make_data("blocks", 100, m, 0.6, 0.05, 8, 4, 1)
        t0 = time.perf_counter()
        minvp.path(X, direction="forward", kmax=50, prefilter=False)
        temps_exact.append(time.perf_counter() - t0)
    axes[0].loglog(ms, temps, "o-", color=C_MINVP_FWD, lw=2.0,
                   label="avec pré-filtre (top-16), K=50")
    axes[0].loglog(ms_exact, temps_exact, "^--", color=C_MINVP_BWD, lw=1.8,
                   label="sans pré-filtre (exact)")
    axes[0].annotate(
        "pré-filtre explicite : ~16 candidats/étape\n→ coût linéaire en M",
        xy=(1600, min(temps)), xytext=(0.06, 0.72), textcoords="axes fraction",
        fontsize=8.5, color="dimgrey",
        arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
    )
    axes[0].set_xlabel("M (nombre de variables), N=100")
    axes[0].set_ylabel("temps du parcours (s)")
    axes[0].set_title("(a) Passage à l'échelle du mode avant")
    axes[0].legend(fontsize=8.5, loc="lower right")

    # (b) evaluation directe vs inverse (parcours arriere)
    ms_b = [100, 150, 200, 250]
    t_direct, t_inv, n_direct, n_inv = [], [], [], []
    for m in ms_b:
        X = make_data("blocks", 600, m, 0.6, 0.05, 6, 4, 1)
        for mode, tt, nn in (("direct", t_direct, n_direct), ("inverse", t_inv, n_inv)):
            t0 = time.perf_counter()
            r = minvp.path(X, direction="backward", kmin=1, eval=mode, verify=False)
            tt.append(time.perf_counter() - t0)
            nn.append(r.total_lanczos_iters / max(1, r.total_exact_evals))
    x = np.arange(len(ms_b))
    w = 0.38
    axes[1].bar(x - w / 2, t_direct, w, color=C_MINMAX, label="matvec direct sur R")
    axes[1].bar(x + w / 2, t_inv, w, color=C_MINVP_FWD, label="évaluation par l'inverse")
    axes[1].set_xticks(x, [f"M={m}" for m in ms_b])
    axes[1].set_ylabel("temps du parcours arrière complet (s)")
    axes[1].set_title("(b) Itérations par candidat ÷ 3 (inverse)")
    for xi, (td, ti, nd, ni) in enumerate(zip(t_direct, t_inv, n_direct, n_inv)):
        axes[1].text(xi - w / 2, td, f"{td:.1f}s\n{nd:.0f} it.", ha="center", va="bottom", fontsize=8)
        axes[1].text(xi + w / 2, ti, f"{ti:.1f}s\n{ni:.0f} it.", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, max(t_direct) * 1.45)
    axes[1].legend(fontsize=8.5, loc="upper left")
    axes[1].grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(DOCS / "fig_performance.png", bbox_inches="tight")
    plt.close(fig)
    print("docs/fig_performance.png ecrit")


if __name__ == "__main__":
    import sys

    choix = sys.argv[1:] or ["qualite", "performance"]
    if "qualite" in choix:
        figure_qualite()
    if "performance" in choix:
        figure_performance()
