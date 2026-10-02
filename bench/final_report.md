# Optimization of the "exact path" of `sifters` - measurements

> **Historical document.** These measurements date from the time when `sifters`
> exposed a command-line binary (since removed). The current benchmark
> harnesses, in Python via the PyO3 binding, are in [`bench/README.md`](README.md)
> (`run.py`, `compare.py`, `scenarios.py`). The options cited below
> (`--max-exact 0`, `--forward-top 0`, `--threads 1`, `--no-verify`)
> map directly to arguments of `sifters.path` / `sifters.select`.

Comparison of the **reference** binary (original sources, `/tmp/sifters_base`) and the
optimized binary, on the **exact path**: `--max-exact 0` and `--forward-top 0`
(certified greedy, no prefilter), `--no-verify`.

* machine: Apple M1 Max, 10 cores, under heavy contention (load average 9-16 during
  the measurements) - hence two metrics;
* `run` = run duration (`seconds` from the JSON), multi-thread, `REP=1`;
* `1 thread` = same duration with `--threads 1`, **the metric least sensitive to
  contention**;
* `exacts` / `iters` = exactly evaluated candidates / cumulative Lanczos iterations
  (deterministic metrics, independent of load).

## Forward selection (`forward --forward-top 0`)

| scenario | run ref | run now | gain | 1 thread ref | 1 thread now | gain | exacts ref->now | iters ref->now | order |
|---|---|---|---|---|---|---|---|---|---|
| fwd400 (M=400, K=50) | 0.92 s | 0.22 s | **4.10x** | 5.11 s | 0.21 s | **23.8x** | 18375 -> 392 | 315356 -> 392 | ok |
| fwd800 (M=800, K=50) | 1.62 s | 0.29 s | **5.52x** | 9.98 s | 0.36 s | **28.1x** | 37975 -> 392 | 629548 -> 392 | ok |
| fwd3200 (M=3200, K=50) | 5.30 s | 0.75 s | **7.04x** | 36.84 s | 1.22 s | **30.1x** | 155575 -> 392 | 2437409 -> 392 | ok |
| fwd1600k150 (M=1600, K=150) | 37.91 s | 8.40 s | **4.51x** | 265.67 s | 10.34 s | **25.7x** | 227225 -> 1192 | 6547307 -> 32701 | ok |

The subsets are **identical** to the reference in all four cases. Better: the
`lambda_min` reported there are **exact**, because the KKT seed built on the full
spectrum converges in a handful of iterations. The reference overestimates (its
warm evaluations saturate `--iters-warm 60`): on `fwd1600k150` at K=112,
0.334289554 (reference) versus 0.334279286 (optimized) - the strict cold revalidation
gives 0.334279286397, residual 1e-13. Likewise at K=150: 0.254933341 (optimized) versus
0.254933379 (reference) for an exact value of 0.254933340508.

## Backward elimination (`backward --max-exact 0`)

| scenario | run ref | run now | gain | 1 thread ref | 1 thread now | gain | exacts ref->now |
|---|---|---|---|---|---|---|---|
| bwd200 (M=200, full family) | 2.23 s | 0.85 s | **2.62x** | 8.75 s | 3.16 s | **2.77x** | 6235 -> 6235 |
| bwd500d (M=500, kmin=200, `direct`) | 17.69 s | 11.04 s | **1.60x** | 89.45 s | 56.49 s | **1.58x** | 13896 -> 16606 |
| bwd500i (same, `inverse`) | 32.90 s | 14.58 s | **2.26x** | 193.10 s | 76.66 s | **2.52x** | 39797 -> 38392 |
| bwd500c (same, `direct` **converged**, `--iters-warm 300`) | 358.96 s | 102.31 s | **3.51x** | 1835.35 s | 624.30 s | **2.94x** | 29993 -> 33976 |
| bwd500def (same, `--eval auto`, the default) | 32.55 s | 14.43 s | **2.25x** | 185.86 s | 77.72 s | **2.39x** | 39797 -> 38392 |

Scenarios: `--gen blocks --blocks 8`, `--gen-n 500` (forward) or `600` (backward).

## Reading

1. **Forward selection: x23 to x30 in actual work.** Three effects add up -
   certification (no more exhaustive evaluation), bordered operator `O(k^2)` on the
   materialized correlation, and above all the **exact** secular bound (`p = k`) that
   reduces evaluations to the `batch` floor (8 per step): the Lanczos iterations
   drop from 2.4 M to 392 on `fwd3200` (~6000x less).
2. **Backward elimination: x1.6 to x3.5.** The gain comes from the secular seed finally
   used (~30 % fewer iterations per candidate), from the Ritz vector by inverse
   iteration, from the second pass of conditional reorthogonalization and from the
   branchless inverse matvec.
3. **Multi-thread < single-thread.** Certification reduces the number of candidates to
   evaluate: little parallelism remains, and the time is lost to synchronization. The
   "work" gain (single-core) is the honest measurement.

## `divergent*`: tie indeterminacy, not a regression

For some backward scenarios, the sequence of subsets differs from the
reference. Verified:

* on i.i.d. **and** correlated data, both binaries reach the **brute-force maximum
  at each step** (deficit <= 1e-15, 0 suboptimal step out of 25);
  this is also true of the forward path (test `forward_matches_brute_force_greedy`,
  and `test_forward_exact_is_stepwise_optimal` on the Python side);
* the optimized binary is even more accurate (mean deficit ~1e-16 versus ~1e-14);
* the divergence appears at a step where two candidates have `lambda_min` 4e-10 apart,
  i.e. **below the certification margin** (`tol*50*(1+|lambda|)` ~= 5e-9): the two
  choices are optimal up to the margin. Over a family of 300 steps, this
  indeterminacy amplifies (greedy is chaotic).

Over 6 independent `blocks` datasets, the mean deviation of the final `lambda_min`
is -0.7 % (direct) / -0.5 % (inverse), with a spread of +/-4 % in both directions.

## Pre-existing bug fixed along the way: CLI revalidation

`sifters --subset K` revalidates the selected subset with a strict cold Lanczos. The
revalidation built a fresh `Dataset` - whose `active` restarts at `0..m` - from
`ds.z`, **already permuted** by the run: the position -> original index
mapping was therefore broken and the revalidation applied to another subset
(0.6036 instead of 0.8634 on a test arbitrated by numpy). Fixed by
`head_subset_dataset`, with the regression test
`head_subset_dataset_validates_the_right_subset`.

## What remains (identified, not implemented)

The forward path is essentially optimal: the exact bound brings evaluations back
to the batch floor (8/step) and the dominant cost is the full spectrum of `R_S`
(`k` warm Lanczos per step).

The **backward** path keeps a factor of ~7: its Temple bound leaves 40.6 % of
the candidate range at `p = 4` (24.1 % at `p = 8`, 12.3 % at `p = 16` - cf. the test
`deletion_bound_quality`), hence ~57 exact evaluations per step instead of 8. Making
it exact requires `p = k`, hence the full spectrum, which would have to be **maintained
incrementally** from one step to the next (Cauchy interlacing + secular
equation) so that the cost per step becomes `O(k^3)` instead of `O(e*t*k^2)`. This is
an algorithmic change (incremental eigensolver), not a tuning; it was not
attempted here.

A more modest avenue: increasing `--low-rank` is a measured bad trade-off - at
`p = 16` evaluations drop by ~20 %, but the spectrum costs 4x more, for a total
time within the noise (+/-10 %).

## Reproducibility

```bash
cargo build --release
REP=1 bench/final.sh final <path/to/reference/binary> [scenario...]
bench/bench_exact.sh <label> all
```
