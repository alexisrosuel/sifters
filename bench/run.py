#!/usr/bin/env python3
"""Benchmark of ``sifters`` runs - replaces the old ``bench/*.sh``.

The shell scripts drove the ``sifters`` binary (CLI interface removed); this
script does the same thing through the Python binding (PyO3), without a subprocess.

For each scenario in :mod:`scenarios`:

* the data are generated once (``make_data``, ``blocks`` generator);
* the run is replayed ``--repeat`` times and the **best** time is kept
  (engine ``seconds``, like the old JSON field) as well as ``path_seconds``;
* one JSON per scenario, a summary CSV and, optionally, a curve CSV
  (``K, lambda_min, subset``) are written into ``bench/out/<label>/``.

Usage:
    python3 bench/run.py --label run --all
    python3 bench/run.py --scenario fwd400 bwd200 --repeat 3
    python3 bench/run.py --all --threads 1 --curve-csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import _paths  # noqa: F401  (inserts scripts/ into sys.path)
import sifters
import scenarios as scen

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "bench" / "out"


def run_once(sc: scen.Scenario, x, threads: int | None) -> tuple[Any, float]:
    """One run; returns ``(Selection, wall duration measured on the Python side)``."""
    kwargs = sc.sifters_kwargs(threads=threads)
    t0 = time.perf_counter()
    result = sifters.path(x, **kwargs)
    return result, time.perf_counter() - t0


def best_of(sc: scen.Scenario, x, repeat: int, threads: int | None):
    """Replays the scenario and keeps the fastest execution (``seconds``)."""
    best = None
    for _ in range(max(1, repeat)):
        result, wall = run_once(sc, x, threads)
        if best is None or result.seconds < best[0].seconds:
            best = (result, wall)
    return best


def curve_rows(result) -> list[dict[str, Any]]:
    rows = []
    for k in sorted(result.curve):
        subset = result.subset_at(k)
        rows.append({
            "k": k,
            "lambda_min": result.curve[k],
            "lambda_raw": result.lambda_raw_at(k),
            "subset": " ".join(str(i) for i in (subset or [])),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", nargs="*", metavar="NAME",
                    help="scenarios to run (default: all)")
    ap.add_argument("--all", action="store_true", help="all scenarios (default)")
    ap.add_argument("--label", default="run", help="output folder name")
    ap.add_argument("--outdir", type=Path, default=None,
                    help=f"output root (default: {DEFAULT_OUT})")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repetitions, the best time is kept (default: 1)")
    ap.add_argument("--threads", type=int, default=None,
                    help="rayon pool size (default: all cores)")
    ap.add_argument("--curve-csv", action="store_true",
                    help="also writes a K/lambda_min/subset CSV per scenario")
    ap.add_argument("--quiet", action="store_true", help="no table on stdout")
    args = ap.parse_args()

    names = args.scenario or ["all"]
    selected = scen.select(names)

    outdir = (args.outdir or DEFAULT_OUT) / args.label
    outdir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    if not args.quiet:
        header = (f"{'scenario':<12} {'N':>5} {'M':>5} {'dir':>8} {'K':>5} "
                  f"{'seconds':>9} {'path_s':>8} {'exacts':>8} {'iters':>10} {'cert':>5}")
        print(header)
        print("-" * len(header))

    for sc in selected:
        data_kw = sc.data_kwargs()
        x = _paths.make_data(**data_kw)
        result, wall = best_of(sc, x, args.repeat, args.threads)

        record = {
            "scenario": sc.name,
            "generator": {k: v for k, v in data_kw.items()},
            "sifters": sc.sifters_kwargs(threads=args.threads),
            "repeat": args.repeat,
            "wall_seconds": wall,
            "result": result.to_dict(),
            "subsets": {
                str(k): result.subset_at(k) for k in sc.subset
                if result.subset_at(k) is not None
            },
        }
        (outdir / f"{sc.name}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if args.curve_csv:
            rows = curve_rows(result)
            with (outdir / f"{sc.name}.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["k", "lambda_min", "lambda_raw", "subset"])
                writer.writeheader()
                writer.writerows(rows)

        ky = result.k
        summaries.append({
            "scenario": sc.name,
            "n": sc.n,
            "m": sc.m,
            "direction": sc.direction,
            "k": ky,
            "seconds": result.seconds,
            "path_seconds": result.path_seconds,
            "load_seconds": result.load_seconds,
            "total_exact_evals": result.total_exact_evals,
            "total_lanczos_iters": result.total_lanczos_iters,
            "certified_steps": result.certified_steps,
            "representation": result.representation,
            "eval": result.eval,
        })

        if not args.quiet:
            print(f"{sc.name:<12} {sc.n:>5} {sc.m:>5} {sc.direction:>8} {ky:>5} "
                  f"{result.seconds:>9.3f} {result.path_seconds:>8.3f} "
                  f"{result.total_exact_evals:>8} {result.total_lanczos_iters:>10} "
                  f"{result.certified_steps:>5}")

    fields = ["scenario", "n", "m", "direction", "k", "seconds", "path_seconds",
              "load_seconds", "total_exact_evals", "total_lanczos_iters",
              "certified_steps", "representation", "eval"]
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    if not args.quiet:
        print(f"\nresults written to {outdir}")


if __name__ == "__main__":
    main()
