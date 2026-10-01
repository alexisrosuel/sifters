"""`minvp` (Rust) contre une implementation naive *full Python / numpy* du meme
probleme : le glouton E-optimal exact.

Le concurrent resout le **meme** objectif que `minvp --dir forward` -- construire
une famille imbriquee en ajoutant, a chaque etape, la variable qui maximise
`lambda_min` de la matrice de correlation du sous-ensemble -- mais sans aucune
astuce algorithmique : a chaque etape il reevalue *tous* les candidats restants en
recalculant une decomposition `numpy.linalg.eigvalsh` complete (dimension K+1).
Soit `O(K * M)` decompositions de taille `~K`.

Trois series sont tracees :

* `minvp` **avec pre-filtre** (`prefilter=True`, top-16) : le glouton avant
  n'evalue que les 16 meilleurs candidats du score seculaire ;
* `minvp` **sans pre-filtre** (`prefilter=False`, le defaut) : meme ensemble de
  candidats que le naif a chaque etape -- on verifie que `lambda_min` est alors
  identique au naif au bit pres, ce qui rend la comparaison de temps strictement
  equitable ;
* **naif numpy** : la reference `eigvalsh`.

L'ecart de temps entre les series 1 et 2 mesure donc exactement ce que les bornes
et le pre-filtre apportent (et ce qu'ils coutent en optimalite).

Usage :
    python3 bench_numpy.py
    python3 bench_numpy.py --m 400,800,1600

Sorties :
    docs/fig_temps_vs_numpy.png   figure du README (temps + qualite)
    docs/temps_vs_numpy.json      valeurs brutes (reproductibles)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import minvp
from compare_baseline import correlation, make_data, lambda_min

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

C_PREFILTRE = "#1b6ca8"  # minvp avec pre-filtre (top-16)
C_EXACT = "#7fb3d5"      # minvp sans pre-filtre (meme chose que le naif)
C_NUMPY = "#d9534f"      # naif numpy


# --------------------------------------------------------------------------- #
# Le concurrent : glouton E-optimal naif, 100 % numpy
# --------------------------------------------------------------------------- #
def naive_forward_greedy(R: np.ndarray, k: int, first: int) -> tuple[list[int], int]:
    """Glouton E-optimal exact, version naive.

    A chaque etape, chaque candidat restant est evalue en recalculant `lambda_min`
    par `numpy.linalg.eigvalsh` sur la sous-matrice agrandie.  `first` fixe la
    variable de depart (celle choisie par `minvp`, pour comparer la meme famille).
    Retourne (sous-ensemble, nombre de decompositions spectrales).
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


def run_minvp(X, k, prefilter: bool):
    t, r = time_call(
        lambda: minvp.select(X, k=k, direction="forward", prefilter=prefilter)
    )
    return t, r


def run_naive_from_R(R, k, first):
    t, (sel, ndec) = time_call(lambda: naive_forward_greedy(R, k, first))
    return t, sel, ndec, lambda_min(R, sel)


# --------------------------------------------------------------------------- #
# Balayages
# --------------------------------------------------------------------------- #
def measure_one(X, k):
    """Renvoie les temps/qualites des trois series pour un (X, k).

    Cote naif on chronometre *aussi* la construction de la matrice de correlation,
    comme `minvp` la fait lui-meme a partir de `X` : les budgets sont comparables.
    """
    t_prefiltre, r_prefiltre = run_minvp(X, k, prefilter=True)
    t_exact, r_exact = run_minvp(X, k, prefilter=False)
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
        print(f"[M] M={m:>6} K={k:>3} | pre-filtre {row['t_rust']:7.3f}s | exact {row['t_rust_exact']:7.3f}s "
              f"| numpy {row['t_numpy']:8.3f}s | x{row['t_numpy'] / row['t_rust']:6.1f} vs pre-filtre "
              f"| {row['ndec']:>8} eigvalsh")
    return rows


def sweep_k(ks, n, m, kind, seed):
    X = make_data(kind, n, m, 0.6, 0.05, 8, 4, seed)
    rows = []
    for k in ks:
        row = measure_one(X, k)
        row["m"] = m
        rows.append(row)
        print(f"[K] M={m:>6} K={k:>3} | pre-filtre {row['t_rust']:7.3f}s | exact {row['t_rust_exact']:7.3f}s "
              f"| numpy {row['t_numpy']:8.3f}s | x{row['t_numpy'] / row['t_rust']:6.1f} vs pre-filtre "
              f"| {row['ndec']:>8} eigvalsh")
    return rows


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def figure(by_m, by_k, n_m: int, n_k: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2))

    # (a) temps vs M
    ms = np.array([r["m"] for r in by_m])
    ax = axes[0]
    ax.loglog(ms, [r["t_rust"] for r in by_m], "o-", color=C_PREFILTRE, lw=2.2, ms=5,
              label="minvp avec pré-filtre (top-16)")
    ax.loglog(ms, [r["t_rust_exact"] for r in by_m], "^--", color=C_EXACT, lw=1.8, ms=5,
              label="minvp sans pré-filtre (exact, mêmes candidats que le naïf)")
    ax.loglog(ms, [r["t_numpy"] for r in by_m], "s-", color=C_NUMPY, lw=2.0, ms=5,
              label="naïf numpy (eigvalsh)")
    for r in [r for r in by_m if r["t_numpy"] / r["t_rust"] >= 1.5][-2:]:
        ax.annotate(f"x{r['t_numpy'] / r['t_rust']:.0f}",
                    xy=(r["m"], r["t_numpy"]), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8.5, color="#4a934a", fontweight="bold")
    ax.set_xlabel(f"M (nombre de variables), N={n_m}, K={by_m[0]['k']}")
    ax.set_ylabel("temps de sélection (s)")
    ax.set_title("(a) Temps vs nombre de variables")

    # (b) temps vs K
    ks = np.array([r["k"] for r in by_k])
    ax = axes[1]
    ax.loglog(ks, [r["t_rust"] for r in by_k], "o-", color=C_PREFILTRE, lw=2.2, ms=5)
    ax.loglog(ks, [r["t_rust_exact"] for r in by_k], "^--", color=C_EXACT, lw=1.8, ms=5)
    ax.loglog(ks, [r["t_numpy"] for r in by_k], "s-", color=C_NUMPY, lw=2.0, ms=5)
    ax.set_xlabel(f"K (variables retenues), N={n_k}, M={by_k[0]['m']}")
    ax.set_ylabel("temps de sélection (s)")
    ax.set_title("(b) Temps vs taille du sous-ensemble")
    for r in [r for r in by_k if r["t_numpy"] / r["t_rust"] >= 1.5][-1:]:
        ax.annotate(f"x{r['t_numpy'] / r['t_rust']:.0f}",
                    xy=(r["k"], r["t_numpy"]), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8.5, color="#4a934a", fontweight="bold")

    # (c) qualite : lambda_min (le naif et minvp sans pre-filtre coincident)
    ax = axes[2]
    ax.semilogx(ms, [100 * r["lam_rust"] for r in by_m], "o-", color=C_PREFILTRE, lw=2.2, ms=5)
    ax.semilogx(ms, [100 * r["lam_numpy"] for r in by_m], "s-", color=C_NUMPY, lw=2.0, ms=5)
    ax.set_xlabel(f"M (nombre de variables), N={n_m}, K={by_m[0]['k']}")
    ax.set_ylabel(r"$100\,\lambda_{\min}$ du sous-ensemble retenu")
    ax.set_title("(c) Qualité : ce que coûte le pré-filtre")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Même objectif (glouton E-optimal), même matériel : Rust vs numpy naïf",
                 fontsize=12, y=1.02)  # noqa: E501
    fig.tight_layout()
    fig.savefig(DOCS / "fig_temps_vs_numpy.png", bbox_inches="tight")
    plt.close(fig)
    print("docs/fig_temps_vs_numpy.png ecrit")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="200,400,800,1600,3200,6400",
                    help="liste de M pour le balayage (a) et (c)")
    ap.add_argument("--k", default="10,20,40,80,120",
                    help="liste de K pour le balayage (b)")
    ap.add_argument("--n-m", type=int, default=500, help="N du balayage (a)/(c)")
    ap.add_argument("--n-k", type=int, default=500, help="N du balayage (b)")
    ap.add_argument("--m-fixed", type=int, default=800, help="M fixe du balayage (b)")
    ap.add_argument("--kind", default="blocks")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--replot", action="store_true",
                    help="regenerer la figure depuis docs/temps_vs_numpy.json")
    args = ap.parse_args()

    if args.replot:
        data = json.loads((DOCS / "temps_vs_numpy.json").read_text())
        figure(data["by_m"], data["by_k"], data["n_m"], data["n_k"])
        return

    ms = [int(v) for v in args.m.split(",")]
    ks = [int(v) for v in args.k.split(",")]
    print(f"# {args.kind}: balayage (a)/(c) N={args.n_m}, K=50 ; balayage (b) N={args.n_k}, "
          f"M={args.m_fixed} ; seed={args.seed}")
    by_m = sweep_m(ms, args.n_m, 50, args.kind, args.seed)
    by_k = sweep_k(ks, args.n_k, args.m_fixed, args.kind, args.seed)

    payload = dict(by_m=by_m, by_k=by_k, n_m=args.n_m, n_k=args.n_k,
                   kind=args.kind, seed=args.seed)
    (DOCS / "temps_vs_numpy.json").write_text(json.dumps(payload, indent=2))
    print("docs/temps_vs_numpy.json ecrit")

    # verification d'equivalence : minvp exact == naif numpy
    worst = max(abs(r["lam_rust_exact"] - r["lam_numpy"]) for r in by_m + by_k)
    print(f"# equivalence minvp exact / naif numpy : ecart max |lambda_min| = {worst:.2e}")

    figure(by_m, by_k, args.n_m, args.n_k)


if __name__ == "__main__":
    main()
