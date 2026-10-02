#!/usr/bin/env python3
"""Compares two runs of ``bench/run.py`` - replaces ``bench/ab.sh`` and
``bench/final.sh``.

The old A/B compared two binaries (reference revision vs current
revision) via ``--out-json``. Without a binary, two result folders
produced by :mod:`run` are compared: run ``run.py --label ref`` on the
reference revision, ``run.py --label now`` on the current one, then:

    python3 bench/compare.py bench/out/ref bench/out/now
    python3 bench/compare.py bench/out/ref bench/out/now --markdown

As in the old ``ab.sh``, the order of the variables is compared: ``OK`` if the
two revisions keep the same subsets, ``DIVERGENT`` otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(directory: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name in {"summary.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "result" in payload:
            runs[path.stem] = payload
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", type=Path, help="reference results folder")
    ap.add_argument("now", type=Path, help="current results folder")
    ap.add_argument("--scenario", nargs="*", metavar="NAME",
                    help="restrict to certain scenarios")
    ap.add_argument("--markdown", action="store_true", help="Markdown table")
    args = ap.parse_args()

    ref, now = load(args.ref), load(args.now)
    names = [n for n in ref if n in now]
    if args.scenario:
        names = [n for n in names if n in set(args.scenario)]
    missing = sorted(set(ref) - set(now))
    if missing:
        print(f"# missing from {args.now}: {', '.join(missing)}")

    if args.markdown:
        print("| scenario | run ref | run now | gain | exacts ref->now "
              "| iters ref->now | order |")
        print("|---|---|---|---|---|---|---|")
    else:
        print(f"{'scenario':<12} {'ref s':>8} {'now s':>8} {'gain':>7}  "
              f"{'exacts ref->now':>16} {'iters ref->now':>20}  order")
        print("-" * 90)

    for name in names:
        r = ref[name]["result"]
        n = now[name]["result"]
        rs, ns = float(r["seconds"]), float(n["seconds"])
        gain = rs / ns if ns else 0.0
        same = "OK" if r.get("order") == n.get("order") else "DIVERGENT"
        exacts = f"{r['total_exact_evals']} -> {n['total_exact_evals']}"
        iters = f"{r['total_lanczos_iters']} -> {n['total_lanczos_iters']}"
        if args.markdown:
            print(f"| {name} | {rs:.2f} s | {ns:.2f} s | **{gain:.2f}x** | "
                  f"{exacts} | {iters} | {same} |")
        else:
            print(f"{name:<12} {rs:>8.2f} {ns:>8.2f} {gain:>6.2f}x  "
                  f"{exacts:>16} {iters:>20}  {same}")


if __name__ == "__main__":
    main()
