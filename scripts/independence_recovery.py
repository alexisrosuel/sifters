"""Ground-truth recovery of the independent features.

The package selects ``k`` features whose joint distribution is as close as
possible to the product of their marginals.  This example checks that claim on a
problem whose answer is **known by construction**.

The generator builds ``M`` features from a few independent latent sources:

* each source ``s_g`` produces a group of ``per_group`` features linked to it by
  a *possibly nonlinear, possibly non-monotone* function (``linear``, ``square``,
  ``tanh``, ``sin``), plus a little noise.  Features of the same group are
  therefore strongly dependent on each other, through a link that a correlation
  does not see (``s`` and ``s**2`` are uncorrelated);
* ``n_singletons`` extra features are pure independent noise, each being its own
  group.

The true number of independent directions is the number of groups
(``n_groups + n_singletons``).  Asking the selector for exactly that ``k``
should return **one feature per group** - never two features built from the same
source.  This module makes the check explicit and the test file
``tests/test_independence_recovery.py`` asserts it.

Run it directly::

    python scripts/independence_recovery.py
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

import sifters

__all__ = [
    "LINKS",
    "make_ground_truth",
    "dependence_summary",
    "recovery_report",
    "main",
]

#: The nonlinear links used to build the groups (cycled over ``per_group``).
LINKS = ("linear", "square", "tanh", "sin")


def _apply_link(name: str, s: np.ndarray) -> np.ndarray:
    if name == "linear":
        return s
    if name == "square":
        return s ** 2
    if name == "tanh":
        return np.tanh(2.0 * s)
    if name == "sin":
        return np.sin(2.0 * s)
    raise ValueError(f"unknown link {name!r}")


def make_ground_truth(
    n: int = 2000,
    n_groups: int = 4,
    per_group: int = 4,
    n_singletons: int = 3,
    seed: int = 0,
    noise: float = 0.1,
    links: Optional[tuple] = None,
) -> tuple:
    """Build ``(X, groups)`` with a known dependence structure.

    ``X`` is ``n x M`` (standardized columns) and ``groups[j]`` is the index of
    the latent source that generated column ``j``.  Columns sharing a group are
    dependent; columns of different groups are independent.  The number of
    independent directions is ``n_groups + n_singletons``.

    ``links`` restricts the functions used inside a group (default
    :data:`LINKS`); pass ``("linear", "tanh")`` to keep the links monotone.
    """
    rng = np.random.default_rng(seed)
    links = LINKS if links is None else tuple(links)
    if not links:
        raise ValueError("links must not be empty")
    sources = rng.normal(size=(n, n_groups))
    columns, groups = [], []
    for g in range(n_groups):
        s = sources[:, g]
        for r in range(per_group):
            link = links[r % len(links)]
            columns.append(_apply_link(link, s) + noise * rng.normal(size=n))
            groups.append(g)
    for j in range(n_singletons):
        columns.append(rng.normal(size=n))
        groups.append(n_groups + j)

    X = np.column_stack(columns)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, np.array(groups, dtype=int)


def dependence_summary(X: Any, groups: np.ndarray, method: str) -> tuple:
    """``(max within-group dependence, max across-group dependence)``."""
    D = sifters.dependence_matrix(X, method=method)
    m = D.shape[0]
    within = across = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            if groups[i] == groups[j]:
                within = max(within, float(D[i, j]))
            else:
                across = max(across, float(D[i, j]))
    return within, across


def recovery_report(
    X: Any,
    groups: np.ndarray,
    *,
    method: str = "dcor",
    k: Optional[int] = None,
    **kwargs: Any,
) -> dict:
    """Select ``k`` features and report whether the true groups were recovered.

    ``k`` defaults to the true number of independent directions.  The selection
    is *correct* when it takes exactly one feature per group, which for
    ``k == number of groups`` is the only way to cover every group once.
    """
    groups = np.asarray(groups, dtype=int)
    n_groups = int(groups.max()) + 1
    k = n_groups if k is None else int(k)
    result = sifters.select_dependence(X, k, method=method, **kwargs)
    picked = groups[result.subset]
    values, counts = np.unique(picked, return_counts=True)
    histogram = {int(v): int(c) for v, c in zip(values, counts)}
    within, across = dependence_summary(X, groups, method)
    return {
        "method": method,
        "k": k,
        "n_groups": n_groups,
        "subset": [int(i) for i in result.subset],
        "groups_selected": [int(v) for v in values],
        "picks_per_group": histogram,
        "one_per_group": len(histogram) == k and all(c == 1 for c in histogram.values()),
        "lambda_min": float(result.lambda_min),
        "within_dependence": within,
        "across_dependence": across,
    }


def main() -> None:
    X, groups = make_ground_truth()
    k = int(groups.max()) + 1
    print(f"X: {X.shape[0]} observations, {X.shape[1]} features, {k} ground-truth groups")
    print("links: " + ", ".join(LINKS) + " (same latent source) + independent noise\n")

    header = f"{'method':7s} {'within':>8s} {'across':>8s} {'lambda':>8s}  {'recovered':>9s}  groups picked"
    print(header)
    print("-" * len(header))
    for method in ("linear", "rank", "dcor", "hsic", "nmi"):
        rep = recovery_report(X, groups, method=method, direction="backward", exact=True)
        picks = ",".join(f"{g}x{c}" for g, c in sorted(rep["picks_per_group"].items()))
        print(
            f"{method:7s} {rep['within_dependence']:8.3f} {rep['across_dependence']:8.3f} "
            f"{rep['lambda_min']:8.3f}  {str(rep['one_per_group']):>9s}  {picks}"
        )

    print(
        "\nA correct method selects exactly one feature per group (`gx1`), i.e. it\n"
        "recovers the independent directions. `linear` and `rank` cannot see the\n"
        "non-monotone links (square, sin) and pick two features of the same source."
    )


if __name__ == "__main__":
    main()
