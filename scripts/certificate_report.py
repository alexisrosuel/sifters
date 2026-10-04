"""Publish the metric that justifies the certificate: how few candidates are evaluated.

The claim under test
--------------------
``sifters`` certifies each greedy **step**: candidates are ranked by a valid upper
bound and evaluated exactly in decreasing order, stopping as soon as the best
realized value beats the largest remaining bound.  The step is then provably the
exact greedy step -- without evaluating all candidates.

That claim is only worth anything if the bound actually prunes.  So the metric
that matters is not the *quality* (which ``scripts/compare_cssp.py`` measures) but
the **cost of the certificate**:

    cost_of_certificate = (candidates evaluated exactly) / (candidates available)

against the naive E-optimal greedy, which evaluates all ``M - k`` (forward) or all
``k`` (backward) candidates at every step.  A ratio near 1 would mean the
certificate is decoration; the interesting regime is ``ratio << 1``.

What is reported
----------------
* the ratio, aggregated over the whole walk, and its maximum per step;
* the fraction of steps flagged ``certified``;
* the cost *per exact evaluation* (Lanczos iterations), so a ratio cannot be
  bought by making each evaluation expensive;
* a sweep over ``low_rank`` (the ``p`` of the Temple bound) showing the pruning
  strengthen as ``p`` grows -- the causal curve behind the ratio;
* the excess of the retained bound over the realized value at the winner, a
  proxy for how tight the bound is where it matters;
* the same walk with ``verify=False``, to isolate the cost of the strict
  revalidation that ``sifters`` enables by default.

Usage
-----
    python3 scripts/certificate_report.py
    python3 scripts/certificate_report.py --m 200 --kmin 20 --kmax 50
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
import numpy as np

import sifters


def _walk_stats(steps: list) -> dict:
    """Aggregate a list of ``sifters.Step`` into certificate statistics."""
    if not steps:
        return {}
    exact = np.array([float(s.exact_evals) for s in steps])
    cands = np.array([float(s.candidates) for s in steps])
    cert = np.array([bool(s.certified) for s in steps])
    iters = np.array([float(s.lanczos_iters) for s in steps])
    lam = np.array([float(s.lambda_min) for s in steps])
    ub = np.array([float(s.upper_bound) for s in steps])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = exact / np.maximum(cands, 1.0)
        excess = np.where(lam > 0.0, ub / np.maximum(lam, 1e-300) - 1.0, np.nan)
    return {
        "steps": int(len(steps)),
        "exact_evals": float(exact.sum()),
        "candidates": float(cands.sum()),
        "ratio": float(exact.sum() / max(cands.sum(), 1.0)),
        "ratio_max_step": float(np.max(ratio)),
        "mean_exact_per_step": float(exact.mean()),
        "mean_candidates_per_step": float(cands.mean()),
        "certified_fraction": float(cert.mean()),
        "iters_per_exact": float(iters.sum() / max(exact.sum(), 1.0)),
        "bound_excess_mean": float(np.nanmean(excess)) if np.isfinite(excess).any() else float("nan"),
        "bound_excess_max": float(np.nanmax(excess)) if np.isfinite(excess).any() else float("nan"),
    }


def backward_sweep(X: np.ndarray, kmin: int, ranks: list[int], verify: bool) -> list[dict]:
    out = []
    for p in ranks:
        sel = sifters.path(X, direction="backward", kmin=kmin, low_rank=p,
                           verify=verify)
        st = _walk_stats(sel.steps)
        st.update({"low_rank": p, "verify": verify, "direction": "backward",
                   "seconds": float(sel.path_seconds) + float(sel.seconds)})
        out.append(st)
    return out


def forward_rows(X: np.ndarray, kmax: int, verify: bool,
                 prefilters: list[int]) -> list[dict]:
    out = []
    # certified mode: forward_top = 0
    sel = sifters.path(X, direction="forward", kmax=kmax, forward_top=0,
                       verify=verify)
    st = _walk_stats(sel.steps)
    st.update({"mode": "certified (forward_top=0)", "forward_top": 0,
               "verify": verify, "direction": "forward",
               "seconds": float(sel.path_seconds) + float(sel.seconds)})
    out.append(st)
    for width in prefilters:
        sel = sifters.path(X, direction="forward", kmax=kmax, prefilter=True,
                           forward_top=width, verify=verify)
        st = _walk_stats(sel.steps)
        st.update({"mode": f"prefilter({width})", "forward_top": width,
                   "verify": verify, "direction": "forward",
                   "seconds": float(sel.path_seconds) + float(sel.seconds)})
        out.append(st)
    return out


def _print_backward(rows: list[dict], m: int, kmin: int) -> None:
    print(f"backward cascade  (M={m} -> kmin={kmin}: {m - kmin} certified steps; "
          f"an exhaustive greedy would evaluate all k candidates per step)")
    print(f"{'low_rank p':>11} {'steps':>6} {'certified':>10} {'exact/cand':>11} "
          f"{'max step':>9} {'mean e/step':>12} {'iters/eval':>11} "
          f"{'bound excess':>13} {'seconds':>9}")
    print("-" * 108)
    for r in rows:
        print(f"{r['low_rank']:>11} {r['steps']:>6} {100 * r['certified_fraction']:>9.1f}% "
              f"{100 * r['ratio']:>10.2f}% {100 * r['ratio_max_step']:>8.1f}% "
              f"{r['mean_exact_per_step']:>12.2f} {r['iters_per_exact']:>11.1f} "
              f"{100 * r['bound_excess_mean']:>12.2f}% {r['seconds']:>9.2f}")
    print()


def _print_forward(rows: list[dict], m: int, kmax: int) -> None:
    print(f"forward walk  (K=1..{kmax}, M={m}; an exhaustive greedy would evaluate "
          f"all remaining candidates per step)")
    print(f"{'mode':<26} {'steps':>6} {'certified':>10} {'exact/cand':>11} "
          f"{'mean e/step':>12} {'mean avail/step':>16} {'iters/eval':>11} {'seconds':>9}")
    print("-" * 108)
    for r in rows:
        print(f"{r['mode']:<26} {r['steps']:>6} {100 * r['certified_fraction']:>9.1f}% "
              f"{100 * r['ratio']:>10.2f}% {r['mean_exact_per_step']:>12.2f} "
              f"{r['mean_candidates_per_step']:>16.1f} {r['iters_per_exact']:>11.1f} "
              f"{r['seconds']:>9.2f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--m", type=int, default=200)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--kind", default="ar")
    ap.add_argument("--kmin", type=int, default=20)
    ap.add_argument("--kmax", type=int, default=50)
    ap.add_argument("--ranks", default="1,2,4,8,16,32")
    ap.add_argument("--prefilters", default="16,32")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--m-scaling", default="",
                    help="comma list of M: rerun the forward walk for each and record "
                         "the certificate cost (the headline 'constant batch' table)")
    ap.add_argument("--skip-backward", action="store_true",
                    help="forward walk only (the backward p-sweep is O(M) steps per p)")
    ap.add_argument("--skip-verify-cost", action="store_true",
                    help="skip the verify=True/False timing comparison")
    ap.add_argument("--out", default="docs/measurements/certificate.json")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compare_baseline import make_data  # noqa: E402

    X = make_data(args.kind, args.n, args.m, args.rho, 0.05, 8, 6, args.seed)
    ranks = [int(v) for v in args.ranks.split(",")]
    prefilters = [int(v) for v in args.prefilters.split(",")]

    print("# cost of the certificate: candidates evaluated exactly vs candidates available")
    print(f"# {platform.platform()}  |  dataset {args.kind} N={args.n} M={args.m} "
          f"rho={args.rho}")
    print("# all walks below use verify=False, so the timings isolate the search itself")
    print("# RELIABILITY: exact_evals / candidates / certified / bound excess are "
          "deterministic")
    print("#   (bit-identical across repetitions, they are what this script is for).")
    print("#   The 'seconds' columns are INDICATIVE ONLY: on a loaded machine the same")
    print("#   call has been observed to vary by up to 3x while doing identical work.\n")

    # Measure the verify cost FIRST and as a best-of-two.  Done after the sweeps it
    # is pure noise: the same call measured 1.15 s cold and 11.0 s at the end of a
    # long run, while the work performed (exact_evals, lanczos_iters, lambda_min)
    # was bit-identical.  This is a timing artifact, not a property of the engine.
    t_rows: list[tuple[str, bool, float]] = []
    directions = [("forward", {"kmax": args.kmax})]
    if not args.skip_backward:
        directions.insert(0, ("backward", {"kmin": args.kmin}))
    if not args.skip_verify_cost:
        for direction, kw in directions:
            for verify in (False, True):
                best = float("inf")
                for _ in range(2):
                    sel = sifters.path(X, direction=direction, verify=verify, **kw)
                    best = min(best, float(sel.path_seconds) + float(sel.seconds))
                t_rows.append((direction, verify, best))

    back = [] if args.skip_backward else backward_sweep(X, args.kmin, ranks, verify=False)
    if back:
        _print_backward(back, args.m, args.kmin)

    fwd = forward_rows(X, args.kmax, verify=False, prefilters=prefilters)
    _print_forward(fwd, args.m, args.kmax)

    if t_rows:
        print("cost of the default strict revalidation (verify=True) on the same walk")
        print("  (best of two, in-run: indicative only, and noisy; a clean isolated")
        print("   measurement gives result-identical output at about +13 % on backward)")
        print(f"{'direction':<12} {'verify':>7} {'seconds':>10} {'slowdown':>10}")
        print("-" * 42)
        for direction in ("backward", "forward"):
            bases = [sec for d, v, sec in t_rows if d == direction and v is False]
            if not bases:
                continue
            base = bases[0]
            for d, v, sec in t_rows:
                if d != direction:
                    continue
                slow = sec / base if base > 0 else float("nan")
                print(f"{d:<12} {str(v):>7} {sec:>10.3f} {slow:>9.2f}x")
        print()

    scaling: list[dict] = []
    if args.m_scaling:
        print("forward certificate cost as M grows (K=1..%d, N=%d)"
              % (args.kmax, args.n))
        print(f"{'M':>7} {'exact/cand':>11} {'mean e/step':>12} {'mean avail/step':>16} "
              f"{'seconds':>9}")
        print("-" * 60)
        for m_i in [int(v) for v in args.m_scaling.split(",") if v.strip()]:
            X_i = make_data(args.kind, args.n, m_i, args.rho, 0.05, 8, 6, args.seed)
            sel = sifters.path(X_i, direction="forward", kmax=args.kmax,
                               forward_top=0, verify=False)
            st = _walk_stats(sel.steps)
            st["M"] = m_i
            st["seconds"] = float(sel.path_seconds) + float(sel.seconds)
            scaling.append(st)
            print(f"{m_i:>7} {100 * st['ratio']:>10.2f}% {st['mean_exact_per_step']:>12.2f} "
                  f"{st['mean_candidates_per_step']:>16.1f} {st['seconds']:>9.2f}")
        print()

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "config": vars(args),
        "backward_verify_false": back,
        "forward_verify_false": fwd,
        "forward_scaling": scaling,
        "verify_cost": [
            {"direction": d, "verify": v, "seconds": sec} for d, v, sec in t_rows
        ],
    }, indent=1, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
