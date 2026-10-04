"""Quality / cost comparison of `sifters` against the classical competitors.

The two heuristics of ``scripts/compare_baseline.py`` (min-max correlation greedy,
threshold filter) are the weak ones.  This script pits `sifters` against the
baselines that a numerical-linear-algebra or applied-statistics reviewer reaches
for first, all implemented in ``scripts/cssp_baselines.py``:

* **QR with column pivoting** / **pivoted Cholesky** -- and these two are the same
  pivot rule, which *is* the D-optimal greedy (``sifters.dopt.greedy``);
* **strong RRQR** (Gu & Eisenstat swaps) with its certified floor;
* **VIF top-k** -- the textbook multicollinearity diagnostic;
* **Fedorov exchange** -- the classical local search on ``lambda_min``;
* **exhaustive E-optimal greedy** with an exact ``eigvalsh`` oracle (forward and
  backward) -- the quality reference;
* **uniform sampling** -- which, on unit-norm columns, is exactly the randomized
  CSSP sampler.

Quality is reported as the scale-free **efficiency**

    ``eta(S) = lambda_min(R_S) / sigma_k(Z)^2``

because Cauchy interlacing makes ``sigma_k(Z)^2`` the largest value any
``k``-subset can reach, so ``eta`` is comparable across datasets.

Usage
-----
    python3 scripts/compare_cssp.py                      # defaults, ~1 min
    python3 scripts/compare_cssp.py --m 400 --k 10,30
    python3 scripts/compare_cssp.py --out docs/measurements/compare_cssp.json

The script prints the same tables it writes to JSON.  numpy only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sifters  # noqa: E402
from compare_baseline import make_data, minmax_greedy_path, threshold_path  # noqa: E402
from cssp_baselines import (  # noqa: E402
    ceiling,
    efficiency,
    eig_greedy_forward,
    fedorov_exchange,
    lam_min,
    pivoted_cholesky,
    qr_pivot,
    rrqr_strong,
    standardize,
    uniform_sample,
    vif_topk,
)

#: Method label -> is it a sifters method (kept first in the tables)?
SIFTERS_LABELS = ("sifters fwd", "sifters fwd x8", "sifters back", "sifters prefilter")


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def run_dataset(X: np.ndarray, ks: list[int], seeds: int, fedorov_passes: int,
                fedorov_restarts: int, do_exhaustive: bool) -> dict:
    """Evaluate every method on one dataset, for every requested K."""
    Z = standardize(X)
    R = Z.T @ Z
    m = R.shape[0]
    kmax = max(ks)
    rng = np.random.default_rng(12345)

    rows: dict[str, dict[int, dict]] = {}

    def put(label: str, k: int, value: float, seconds: float, **extra) -> None:
        ceil = ceiling(R, k)
        rows.setdefault(label, {})[k] = {
            "lambda_min": float(value),
            "efficiency": efficiency(value, ceil),
            "seconds": float(seconds),
            **extra,
        }

    # ---- sifters: one nested path for all K, then read every size back ------
    t0 = time.perf_counter()
    fwd = sifters.path(X, direction="forward", kmax=kmax, forward_seeds=1)
    t_fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    fwd8 = sifters.path(X, direction="forward", kmax=kmax, forward_seeds=8)
    t_fwd8 = time.perf_counter() - t0
    t0 = time.perf_counter()
    back = sifters.path(X, direction="backward", kmin=min(ks))
    t_back = time.perf_counter() - t0
    t0 = time.perf_counter()
    back_nv = sifters.path(X, direction="backward", kmin=min(ks), verify=False)
    t_back_nv = time.perf_counter() - t0
    t0 = time.perf_counter()
    pref = sifters.path(X, direction="forward", kmax=kmax, prefilter=True)
    t_pref = time.perf_counter() - t0
    t0 = time.perf_counter()
    # local 1-for-1 swaps live on the backward walk and only apply at sizes
    # <= swap_from, so swap_from must be at least the target K
    swapped = sifters.select(X, k=kmax, direction="backward", swap_passes=5,
                             swap_top=16, swap_from=kmax)
    t_swap = time.perf_counter() - t0
    for label, sel, total in (("sifters fwd", fwd, t_fwd),
                              ("sifters fwd x8", fwd8, t_fwd8),
                              ("sifters back", back, t_back),
                              ("sifters back (verify=False)", back_nv, t_back_nv),
                              ("sifters prefilter", pref, t_pref)):
        # The whole path cost is reported on every row: it is a single run that
        # answers every K, which is the property under test.
        for k in ks:
            sub = sel.subset_at(k)
            value = sel.lambda_at(k)
            if sub is not None and value is not None and len(sub) == k:
                put(label, k, value, total, path_total=True,
                    exact_evals=int(sel.total_exact_evals),
                    candidates=int(sum(s.candidates for s in sel.steps)),
                    certified_steps=int(sel.certified_steps),
                    steps=len(sel.steps))
    # local 1-for-1 swaps only exist on the backward walk; they correct greedily
    # at a single size, so only the largest K is meaningful
    put("sifters back + swaps", kmax, swapped.lambda_min, t_swap, path_total=False,
        exact_evals=int(swapped.total_exact_evals))

    # ---- deterministic cheap heuristics ------------------------------------
    mm_subset, t_mm = minmax_greedy_path(R, kmax)
    thr_subsets, t_thr = threshold_path(R, kmax)

    for k in ks:
        ceil_k = ceiling(R, k)
        # QRCP / pivoted Cholesky / D-optimal greedy: one and the same pivot rule
        S_qr, t = _timed(qr_pivot, Z, k)
        S_chol = pivoted_cholesky(R, k)
        put("QRCP = pivoted Cholesky = D-opt", k, lam_min(R, S_qr), t)
        rows.setdefault("_check_qrcp_is_pivoted_cholesky", {})[k] = {
            "identical": sorted(S_qr) == sorted(S_chol)
        }
        # strong RRQR + its certified floor
        (S, swaps, bound, extra), t = _timed(rrqr_strong, Z, k, f=1.0)
        value = lam_min(R, S)
        put("strong RRQR (f=1)", k, value, t, swaps=swaps,
            bound=float(bound), bound_efficiency=efficiency(bound, ceil_k),
            achieved_over_bound=float(value / max(bound, 1e-300)),
            growth=extra["growth"], f_eff=extra["f_eff"])
        # VIF
        S, t = _timed(vif_topk, R, k, ridge=1e-10)
        put("VIF top-k", k, lam_min(R, S), t)
        # the weak baselines, kept for continuity with the README tables
        s = mm_subset(k)
        if len(s) == k:
            put("min-max corr greedy", k, lam_min(R, s), t_mm)
        if thr_subsets:
            near = min(thr_subsets, key=lambda kk: abs(kk - k))
            # A subset of the wrong size is not comparable: a larger subset has a
            # mechanically smaller lambda_min.  Flag it instead of hiding it.
            if near == k:
                put("threshold filter", k, lam_min(R, thr_subsets[near]), t_thr,
                    k_got=int(near), comparable=True)
            else:
                put("threshold filter", k, lam_min(R, thr_subsets[near]), t_thr,
                    k_got=int(near), comparable=False)
        # uniform sampling, averaged over `seeds` draws
        vals, times = [], []
        for s_i in range(seeds):
            r = np.random.default_rng(1000 + s_i)
            S, t = _timed(uniform_sample, m, k, r)
            vals.append(lam_min(R, S))
            times.append(t)
        put("uniform sample (mean)", k, float(np.mean(vals)),
            float(np.mean(times)), std=float(np.std(vals)))
        # Fedorov exchange
        (S, stats), t = _timed(fedorov_exchange, R, k, rng,
                               passes=fedorov_passes, restarts=fedorov_restarts)
        put("Fedorov exchange", k, lam_min(R, S), t,
            evaluations=int(stats["evaluations"]), improvements=int(stats["improvements"]))
        # exhaustive eigen-oracle forward greedy: the quality reference, and the
        # check that the *certified* sifters forward walk returns the same values
        if do_exhaustive:
            (Sf, infof), t = _timed(eig_greedy_forward, R, kmax)
            oracle_value = infof["curve"].get(k)
            fwd_value = fwd.lambda_at(k)
            if oracle_value is not None:
                put("eigvalsh greedy fwd (oracle)", k, oracle_value, t,
                    evaluations=int(infof["evaluations"]))
                if fwd_value is not None:
                    rows.setdefault("_check_fwd_matches_oracle", {})[k] = {
                        "max_abs_diff": abs(float(oracle_value) - float(fwd_value))
                    }

    return {"M": int(m), "N": int(X.shape[0]), "ceiling": {k: ceiling(R, k) for k in ks},
            "ranges": {"forward": [1, kmax], "backward": [min(ks), int(m)]},
            "rows": rows}


def print_tables(kind: str, data: dict, ks: list[int]) -> None:
    rows = data["rows"]
    order = [lbl for lbl in SIFTERS_LABELS if lbl in rows]
    order += [lbl for lbl in rows if lbl not in SIFTERS_LABELS and not lbl.startswith("_")]
    print(f"\n### {kind}  (N={data['N']}, M={data['M']})")
    ceil_txt = "  ".join(f"K={k}: sigma_K^2={data['ceiling'][k]:.4f}" for k in ks)
    print(f"ceiling per K (max reachable): {ceil_txt}\n")
    for metric, fmt, header in (("efficiency", "{:>9.3f}", "efficiency eta = lambda_min / sigma_K^2"),
                                ("lambda_min", "{:>9.5f}", "lambda_min (absolute)")):
        print(f"{header}")
        print(f"{'method':<34}" + "".join(f"{('K=' + str(k)):>10}" for k in ks))
        print("-" * (34 + 10 * len(ks)))
        for lbl in order:
            cells = []
            for k in ks:
                cell = rows.get(lbl, {}).get(k)
                if cell is None:
                    cells.append(f"{'-':>9}")
                elif cell.get("comparable") is False:
                    cells.append(f"{'n/a*':>9}")
                else:
                    cells.append(fmt.format(cell[metric]))
            print(f"{lbl:<34}" + "".join(cells))
        print()
    print("seconds (path total for sifters: one run answers every K in the visited range)")
    rng_txt = data.get("ranges", {})
    if rng_txt:
        print(f"  visited ranges: forward K=1..{rng_txt['forward'][1]}, "
              f"backward K={rng_txt['backward'][0]}..{rng_txt['backward'][1]} "
              f"(the backward paths therefore do far more work; "
              f"'sifters back + swaps' covers K={max(ks)} only)")
    print(f"{'method':<34}" + "".join(f"{('K=' + str(k)):>10}" for k in ks))
    print("-" * (34 + 10 * len(ks)))
    for lbl in order:
        cells = []
        for k in ks:
            cell = rows.get(lbl, {}).get(k)
            sec = cell["seconds"] if cell else None
            if sec is None or not np.isfinite(sec):
                cells.append(f"{'-':>9}")
            elif sec >= 10.0:
                cells.append(f"{sec:>9.2f}")
            elif sec >= 1.0:
                cells.append(f"{sec:>9.3f}")
            else:
                cells.append(f"{sec:>9.4f}")
        print(f"{lbl:<34}" + "".join(cells))
    print()

    rr = rows.get("strong RRQR (f=1)", {})
    if rr:
        print("strong RRQR: is its global guarantee binding?")
        for k in ks:
            c = rr.get(k)
            if not c:
                continue
            print(f"  K={k:3d}  bound={c['bound']:.3e}  achieved={c['lambda_min']:.6f}  "
                  f"achieved/bound={c['achieved_over_bound']:.0f}x  "
                  f"eta_guaranteed={c['bound_efficiency']:.4f}  eta_achieved={c['efficiency']:.3f}")
        print()

    if any(rows.get(lbl, {}).get(k, {}).get("comparable") is False
           for lbl in rows for k in ks):
        print("n/a*: the threshold filter could not produce a subset of exactly that "
              "size, so the row is not a like-for-like comparison.\n")

    check = rows.get("_check_fwd_matches_oracle", {})
    if check:
        worst = max(v["max_abs_diff"] for v in check.values())
        print(f"certification self-check: |lambda_min(sifters fwd) - oracle fwd| <= {worst:.3e}")
        print("  (the certified forward walk must reproduce the exhaustive greedy)\n")

    ident = rows.get("_check_qrcp_is_pivoted_cholesky", {})
    if ident:
        ok = all(v["identical"] for v in ident.values())
        print(f"identity self-check: QRCP == pivoted Cholesky (D-optimal greedy): {ok}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", default="iid,ar,blocks,factor")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--m", type=int, default=200)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--rho-out", type=float, default=0.05)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", default="5,15,30")
    ap.add_argument("--sample-seeds", type=int, default=10,
                    help="number of draws averaged for the uniform sampler")
    ap.add_argument("--fedorov-passes", type=int, default=10)
    ap.add_argument("--fedorov-restarts", type=int, default=2)
    ap.add_argument("--exhaustive", default="auto",
                    help="include the eigvalsh-oracle greedy (auto = on when M*K is small)")
    ap.add_argument("--out", default="docs/measurements/compare_cssp.json")
    args = ap.parse_args()

    kinds = [s.strip() for s in args.kind.split(",") if s.strip()]
    ks = [int(v) for v in args.k.split(",")]
    do_exhaustive = (args.exhaustive == "on"
                     or (args.exhaustive == "auto" and args.m * max(ks) <= 12000))

    print("# sifters vs classical column-subset-selection baselines")
    print(f"# {platform.platform()}  |  machine cores: {os.cpu_count()}")
    print(f"# N={args.n} M={args.m} rho={args.rho} k={ks} "
          f"exhaustive_oracle={do_exhaustive}")
    print("# quality metric: eta = lambda_min(R_S) / sigma_K(Z)^2  (<= 1, higher is better)")

    out: dict = {"config": vars(args), "datasets": {}}
    for kind in kinds:
        X = make_data(kind, args.n, args.m, args.rho, args.rho_out,
                      args.blocks, args.rank, args.seed)
        data = run_dataset(X, ks, args.sample_seeds, args.fedorov_passes,
                           args.fedorov_restarts, do_exhaustive)
        out["datasets"][kind] = data
        print_tables(kind, data, ks)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
