"""Catalogue of benchmark scenarios.

Reproduces exactly the scenarios of the old ``bench/*.sh`` (which drove the
``sifters`` binary):

* ``blocks`` generator, ``rho=0.3``, ``rho_out=0.0``, ``blocks=8``, ``seed=1``
  (old CLI defaults);
* exact path: ``prefilter=False``, ``forward_top=0``, ``max_exact=0``
  (certified greedy, no prefilter);
* ``verify=False``: the run is measured, not the strict cold revalidation.

Each scenario reuses the engine through the Python binding (``sifters.path``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Scenario:
    """A dataset + a run to measure."""

    name: str
    m: int
    direction: str = "backward"
    n: int = 500
    kind: str = "blocks"
    rho: float = 0.3
    rho_out: float = 0.0
    blocks: int = 8
    rank: int = 4
    seed: int = 1
    kmin: int = 1
    kmax: int | None = None
    #: K sizes whose subset is to be checked/reported
    subset: tuple[int, ...] = ()
    #: overrides passed to :func:`sifters.path`
    options: dict[str, Any] = field(default_factory=dict)

    def sifters_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Options of :func:`sifters.path` for this scenario."""
        kw: dict[str, Any] = {
            "direction": self.direction,
            "kmin": self.kmin,
            "center": True,
            "verify": False,
            "prefilter": False,
            "forward_top": 0,
            "max_exact": 0,
        }
        if self.direction == "forward":
            kw["kmax"] = self.kmax
        kw.update(self.options)
        kw.update(overrides)
        return kw

    def data_kwargs(self, seed: int | None = None) -> dict[str, Any]:
        """Generator parameters (for ``make_data``)."""
        return {
            "kind": self.kind,
            "n": self.n,
            "m": self.m,
            "rho": self.rho,
            "rho_out": self.rho_out,
            "blocks": self.blocks,
            "rank": self.rank,
            "seed": self.seed if seed is None else seed,
        }


# --------------------------------------------------------------------------- #
# Historical scenarios (forward: bench/final.sh; backward: bench/ab.sh)
# --------------------------------------------------------------------------- #
SCENARIOS: tuple[Scenario, ...] = (
    # --- forward selection (certified greedy) ---
    Scenario("fwd400", m=400, direction="forward", kmax=50, subset=(5, 10, 20, 50)),
    Scenario("fwd800", m=800, direction="forward", kmax=50, subset=(5, 10, 20, 50)),
    Scenario("fwd3200", m=3200, direction="forward", kmax=50, subset=(5, 10, 20, 50)),
    Scenario("fwd1600k150", m=1600, direction="forward", kmax=150),
    # --- backward elimination ---
    Scenario("bwd200", n=600, m=200, direction="backward", kmin=1,
             subset=(5, 10, 20, 50)),
    Scenario("bwd500d", n=600, m=500, direction="backward", kmin=200,
             subset=(250, 300, 400), options={"eval": "direct"}),
    Scenario("bwd500i", n=600, m=500, direction="backward", kmin=200,
             subset=(250, 300, 400), options={"eval": "inverse"}),
    Scenario("bwd500c", n=600, m=500, direction="backward", kmin=200,
             subset=(250, 300, 400),
             options={"eval": "direct", "iters_warm": 300}),
    Scenario("bwd500def", n=600, m=500, direction="backward", kmin=200,
             subset=(250, 300, 400)),
)

BY_NAME: dict[str, Scenario] = {s.name: s for s in SCENARIOS}


def select(names: list[str] | None) -> list[Scenario]:
    """Filters the catalogue; ``None`` or ``["all"]`` returns everything."""
    if not names or names == ["all"]:
        return list(SCENARIOS)
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        raise SystemExit(
            f"unknown scenario(s): {', '.join(unknown)}\n"
            f"available: {', '.join(BY_NAME)}"
        )
    return [BY_NAME[n] for n in names]


__all__ = ["Scenario", "SCENARIOS", "BY_NAME", "select"]
