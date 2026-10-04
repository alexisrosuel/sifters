"""Reproduce (or fail to reproduce) the speed rows of the README, and say which.

Why this script exists
----------------------
Performance claims in a numerical package usually rot silently: the machine
changes, the build profile changes, the configuration in the command that was
timed is forgotten.  This script pins all three down.

* It re-measures the exact rows of README section 4 "Speed" **using the library
  defaults**, writing down the configuration it used, and compares against the
  value the README claims.
* It reports `total_exact_evals` next to each timing.  That count is
  deterministic and is the real check: if the algorithm traverses what the README
  says it traverses, a timing difference is an environment difference and nothing
  more.
* It takes a **warmup** call and the **best of N** repetitions.  Skipping either
  is how a release-build timing gets inflated: the same call in this working tree
  measured anywhere between 0.05 s and 1.45 s depending on warmup and load, while
  doing bit-identical work.
* It refuses to silently time a debug build (`sifters.build_profile()`), which is
  about 14x slower.

It also records the claims that are **not** reproducible, so the failure is
documented instead of hidden: the `x24`-`x30` gain table of the README compares
against a pre-optimization revision that is absent from the git history, so no
one can re-run it.

Usage
-----
    python3 scripts/reproduce_speed.py                 # best of 3, ~1 min
    python3 scripts/reproduce_speed.py --reps 1 --only forward
    python3 scripts/reproduce_speed.py --out docs/measurements/speed_ledger.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sifters  # noqa: E402
from compare_baseline import make_data  # noqa: E402

#: (label, kind, N, M, kwargs, README claim in seconds, README exact_evals)
#: `kwargs` are what the README row uses, on top of the library defaults.
ROWS: tuple = (
    ("forward, M=400, K=50", "ar", 500, 400,
     dict(k=50, direction="forward"), 0.2, 392),
    ("forward, M=3200, K=50", "ar", 500, 3200,
     dict(k=50, direction="forward"), 0.8, 392),
    ("forward, M=1600, K=150", "ar", 400, 1600,
     dict(k=150, direction="forward"), 8.4, 1192),
    ("forward, M=10000, N=50, K=100, prefilter", "iid", 50, 10000,
     dict(k=100, direction="forward", prefilter=True), 0.74, 1584),
    ("full backward, M=200", "ar", 500, 200,
     dict(direction="backward", kmin=1), 0.85, 6235),
    ("backward, M=500, kmin=200, eval=inverse", "ar", 600, 500,
     dict(direction="backward", kmin=200, eval="inverse"), 14.6, 38392),
)

#: README rows that cannot be reproduced, with the reason.  Kept in the output on
#: purpose: a claim nobody can re-run should be visible, not quietly deleted.
UNREPRODUCIBLE: tuple = (
    ("forward, M=400, K=50 : x24 gain vs the original version",
     "the pre-optimization revision is absent from the git history "
     "(3 commits, all from the same day, the oldest already being the "
     "optimization commit), so bench/compare.py has no ref revision to check out"),
    ("forward, M=800, K=50 : x28 ; forward, M=3200, K=50 : x30", "same reason"),
    ("full backward, M=200 : x2.8 ; backward M=500 kmin=200 : x2.5", "same reason"),
)


def _best_of(fn: Callable[[], object], reps: int) -> tuple:
    """Run ``fn`` ``reps`` times and return ``(result, best_seconds, all_seconds)``."""
    result: Optional[object] = None
    seconds: list[float] = []
    for _ in range(max(1, reps)):
        t0 = time.perf_counter()
        result = fn()
        seconds.append(time.perf_counter() - t0)
    return result, min(seconds), seconds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=3, help="best of N (default 3)")
    ap.add_argument("--only", default="", choices=["", "forward", "backward"],
                    help="restrict to one direction")
    ap.add_argument("--out", default="docs/measurements/speed_ledger.json")
    args = ap.parse_args()

    profile = sifters.build_profile()
    print("# reproducing the README section 4 speed rows")
    print(f"# {platform.platform()}  |  cores: {os.cpu_count()}  |  "
          f"build profile: {profile}  |  best of {args.reps}, one warmup call")
    if profile != "release":
        print("# ABORT: a debug build is about 14x slower; "
              "rebuild with `maturin develop --release`.")
        raise SystemExit(2)
    print()

    # warmup: first-call effects (rayon pool, page faults, caches) are large enough
    # to dominate the smaller rows if they are not paid for first
    print("# warmup...")
    warm = make_data("ar", 500, 400, 0.9, 0.05, 8, 6, 1)
    sifters.select(warm, k=50, direction="forward", verify=False)
    del warm
    print()

    header = (f"{'row':<42} {'README':>8} {'measured':>9} {'ratio':>7} "
              f"{'evals':>7} {'README ev':>9} {'ev match':>8}")
    print(header)
    print("-" * len(header))

    out: dict = {
        "platform": platform.platform(),
        "build_profile": profile,
        "reps": args.reps,
        "warmup": True,
        "rows": [],
        "unreproducible": [{"claim": c, "reason": r} for c, r in UNREPRODUCIBLE],
    }

    for label, kind, n, m, kwargs, claim_s, claim_evals in ROWS:
        direction = kwargs.get("direction", "forward")
        if args.only and direction != args.only:
            continue
        X = make_data(kind, n, m, 0.9, 0.05, 8, 6, 1)
        # `kmin`/`kmax` belong to `path` (a nested family); `k` belongs to `select`
        call = sifters.path if ("kmin" in kwargs or "kmax" in kwargs) else sifters.select
        result, best, all_s = _best_of(lambda: call(X, **kwargs), args.reps)
        evals = int(getattr(result, "total_exact_evals", -1))
        match = "yes" if evals == claim_evals else "NO"
        print(f"{label:<42} {claim_s:>7.2f}s {best:>8.3f}s "
              f"{best / claim_s:>6.2f}x {evals:>7d} {claim_evals:>9d} {match:>8}")
        out["rows"].append({
            "label": label, "kind": kind, "N": n, "M": m,
            "kwargs": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                       for k, v in kwargs.items()},
            "readme_seconds": claim_s, "measured_seconds": best,
            "all_seconds": all_s, "ratio": best / claim_s,
            "exact_evals": evals, "readme_exact_evals": claim_evals,
            "exact_evals_match": evals == claim_evals,
        })

    print()
    print("# NOT reproducible from this repository (recorded, not hidden):")
    for claim, reason in UNREPRODUCIBLE:
        print(f"#   - {claim}")
        print(f"#     {reason}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
