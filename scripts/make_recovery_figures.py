"""Generate the ground-truth recovery figures of the README.

Usage:
    python3 make_recovery_figures.py

Produces:
    docs/img/fig_recovery_setup.png   the generative model and the two dependence
                                      matrices (distance correlation vs correlation)
    docs/img/fig_recovery_result.png  what each method selects, group by group

The numbers come from the real package (`sifters.select_dependence`); the
generator lives in `scripts/independence_recovery.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
IMG = DOCS / "img"
IMG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

import sifters  # noqa: E402
from independence_recovery import make_ground_truth  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 9,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

METHODS = ("linear", "rank", "dcor", "hsic", "nmi")
OK_COLOR = "#1b6ca8"
BAD_COLOR = "#d9534f"
NOISE_COLOR = "#8a8a8a"
SOURCE_COLORS = ("#1b6ca8", "#e8a33d", "#3f9d63", "#8e6bbf")
LINKS = ("s", "s$^2$", "tanh(2s)", "sin(2s)")


def _selection(X, groups, method):
    k = int(groups.max()) + 1
    result = sifters.select_dependence(X, k, method=method, direction="backward", exact=True)
    return np.asarray(result.subset, dtype=int), float(result.lambda_min)


def figure_setup(X, groups, D_dcor, D_lin) -> None:
    fig = plt.figure(figsize=(13.5, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.30)

    # -- (a) generative model ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ys = (0.88, 0.66, 0.44, 0.22)
    for g, y in enumerate(ys):
        ax.add_patch(Circle((0.10, y), 0.055, color=SOURCE_COLORS[g], zorder=3))
        ax.text(0.10, y, f"s{g}", ha="center", va="center", color="white",
                fontsize=10, fontweight="bold", zorder=4)
        ax.add_patch(FancyBboxPatch((0.28, y - 0.075), 0.68, 0.15,
                                    boxstyle="round,pad=0.012",
                                    facecolor=SOURCE_COLORS[g], alpha=0.14,
                                    edgecolor=SOURCE_COLORS[g], lw=1.2))
        ax.text(0.62, y, "   ".join(LINKS) + "  + noise", ha="center", va="center",
                fontsize=8.5, color="#22303c")
        ax.add_patch(FancyArrowPatch((0.16, y), (0.27, y), arrowstyle="-|>",
                                     mutation_scale=11, color=SOURCE_COLORS[g], lw=1.3))
    ax.add_patch(FancyBboxPatch((0.28, 0.02), 0.68, 0.13,
                                boxstyle="round,pad=0.012",
                                facecolor=NOISE_COLOR, alpha=0.12,
                                edgecolor=NOISE_COLOR, lw=1.2, linestyle="--"))
    ax.text(0.62, 0.085, "3 independent features  u1 u2 u3", ha="center", va="center",
            fontsize=8.5, color="#22303c")
    ax.text(0.0, 1.02, "(a) generative model   M = 19, true groups = 7",
            fontsize=9.5, fontweight="bold", color="#22303c")

    # -- (b) distance correlation -------------------------------------------
    _heatmap(fig.add_subplot(gs[0, 1]), D_dcor, groups,
             "(b) distance correlation  D", vmax=1.0,
             note="within groups ~0.9, across ~0.05")

    # -- (c) |linear correlation| -------------------------------------------
    _heatmap(fig.add_subplot(gs[0, 2]), np.abs(D_lin), groups,
             "(c) |linear correlation|", vmax=1.0,
             note="group 0 block almost 0: s, s$^2$, sin(2s) look independent",
             highlight_group=0)

    fig.savefig(IMG / "fig_recovery_setup.png", bbox_inches="tight")
    plt.close(fig)


def _heatmap(ax, matrix, groups, title, vmax, note, highlight_group=None):
    m = matrix.shape[0]
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")
    bounds = np.cumsum(np.bincount(groups))[:-1] - 0.5
    for b in bounds:
        ax.axhline(b, color="white", lw=1.2)
        ax.axvline(b, color="white", lw=1.2)
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(m))
    ax.set_xticklabels(np.arange(m), fontsize=6)
    ax.set_yticklabels(np.arange(m), fontsize=6)
    ax.set_title(title, fontsize=9.5, pad=8)
    ax.text(0.5, -0.16, note, transform=ax.transAxes, ha="center", fontsize=8,
            color="#5a6b78")
    if highlight_group is not None:
        idx = np.nonzero(groups == highlight_group)[0]
        lo, hi = idx.min() - 0.5, idx.max() + 0.5
        ax.add_patch(Rectangle((lo, lo), hi - lo, hi - lo, fill=False,
                               edgecolor=BAD_COLOR, lw=2.0))
    fig = ax.figure
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=7)


def figure_result(X, groups, selections) -> None:
    k = int(groups.max()) + 1
    n_groups = int(groups.max()) + 1
    fig = plt.figure(figsize=(13.5, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.22)

    # -- (a) picks per source ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    counts = np.zeros((len(METHODS), n_groups), dtype=int)
    for r, method in enumerate(METHODS):
        picked = groups[selections[method][0]]
        for g in picked:
            counts[r, g] += 1
    im = ax.imshow(counts, cmap="Blues", vmin=0, vmax=2, interpolation="nearest",
                   aspect="auto")
    for r in range(len(METHODS)):
        for g in range(n_groups):
            bad = counts[r, g] != 1
            ax.text(g, r, str(counts[r, g]), ha="center", va="center",
                    fontsize=10, fontweight="bold" if bad else "normal",
                    color=BAD_COLOR if bad else "#22303c")
            if bad:
                ax.add_patch(Rectangle((g - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor=BAD_COLOR, lw=2.0))
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([f"src {g}" if g < 4 else f"u{g - 4 + 1}" for g in range(n_groups)],
                       fontsize=8)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS, fontsize=9)
    ax.set_title("(a) features picked per source  (1 = correct)", fontsize=9.5, pad=8)
    ax.set_xlabel("ground-truth group", fontsize=8.5)

    # -- (b) selection map ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    m = X.shape[1]
    bounds = np.cumsum(np.bincount(groups))[:-1]
    for b in bounds:
        ax.axvline(b - 0.5, color="#c9d2d9", lw=1.0, zorder=0)
    for r, method in enumerate(METHODS):
        subset, _lmin = selections[method]
        picked = groups[subset]
        values, reps = np.unique(picked, return_counts=True)
        duplicates = set(values[reps > 1].tolist())
        for j in subset:
            g = int(groups[j])
            color = SOURCE_COLORS[g] if g < 4 else NOISE_COLOR
            ax.scatter(j, r, s=90, color=color, zorder=3,
                       edgecolor=BAD_COLOR if g in duplicates else "white",
                       linewidth=2.0 if g in duplicates else 0.8)
        for g in duplicates:
            idx = subset[groups[subset] == g]
            ax.plot(idx, [r] * len(idx), color=BAD_COLOR, lw=2.0, zorder=2)
    ax.set_xlim(-1, m)
    ax.set_ylim(-0.7, len(METHODS) - 0.3)
    ax.invert_yaxis()
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS, fontsize=9)
    ax.set_xticks(range(m))
    ax.set_xticklabels(range(m), fontsize=6)
    ax.set_xlabel("feature index (ordered by source)", fontsize=8.5)
    ax.set_title("(b) which features each method keeps", fontsize=9.5, pad=22)
    ax.text(0.0, 1.045, "red circle = a source kept twice (missed independent direction)",
            transform=ax.transAxes, fontsize=8, color=BAD_COLOR)

    handles = [plt.Line2D([], [], marker="o", ls="", color=SOURCE_COLORS[g],
                          label=f"source {g}") for g in range(4)]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=NOISE_COLOR,
                              label="independent noise"))
    ax.legend(handles=handles, fontsize=7.5, ncol=5, loc="lower center",
              bbox_to_anchor=(0.5, -0.34))

    fig.savefig(IMG / "fig_recovery_result.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    X, groups = make_ground_truth()
    D_dcor = sifters.dependence_matrix(X, method="dcor")
    D_lin = sifters.dependence_matrix(X, method="linear")
    selections = {method: _selection(X, groups, method) for method in METHODS}

    figure_setup(X, groups, D_dcor, D_lin)
    figure_result(X, groups, selections)

    print("wrote docs/img/fig_recovery_setup.png and docs/img/fig_recovery_result.png")
    for method in METHODS:
        subset, lmin = selections[method]
        picks = groups[subset]
        counts = np.bincount(picks, minlength=int(groups.max()) + 1)
        print(f"{method:7s} one_per_group={bool((counts == 1).all())}  "
              f"lambda_min={lmin:.3f}  picks={counts.tolist()}")


if __name__ == "__main__":
    main()
