# Measurements: what is reproducible, and the honest reading

Every number quoted in the README's quality and certificate tables lives here,
with the command that regenerates it. The point of this file is that a claim
about a numerical package is only worth the reproducibility of its measurement,
so the measurements that turned out to be *unreliable* are reported as such
rather than dropped.

## 0. How to regenerate, and what can go wrong

```bash
maturin develop --release          # --release is mandatory, see below
python3 scripts/compare_cssp.py    # quality vs the classical competitors
python3 scripts/certificate_report.py --m-scaling 200,400,800,1600,3200
pytest tests -q
```

Raw output of the runs quoted below is in `docs/measurements/`:

| file | content |
|---|---|
| `compare_cssp.json` | efficiency, absolute `lambda_min`, timings, RRQR slack |
| `certificate.json` | certificate cost, backward p-sweep, forward M-scaling |

Reference machine: 10-core Apple silicon, macOS, `float64`, extension built with
`maturin develop --release` (no `-C target-cpu=native`).

### Trap 1: a debug build silently invalidates every timing

The working tree shipped an **unoptimized** `_sifters` extension (2452256 bytes,
identical to `target/debug/lib_sifters.dylib`). Measured on the same machine and
the same problem (`forward, M=400, K=50`): **2.78 s in debug against 0.19 s in
release, a factor of about 14**, with bit-identical results and an identical
`exact_evals` count of 392.

`maturin develop` without `--release` is the most natural command to type, so the
package now reports what it was built with: `sifters.build_profile()` returns
`"release"` or `"debug"`, and importing `sifters` emits a `RuntimeWarning` on a
debug build.

### Trap 2: timings measured under load are not timings

Repeated calls with identical arguments do **identical work** - verified
bit-identical `exact_evals`, `lanczos_iters` and `lambda_min` over four
repetitions - but the wall clock on a loaded machine varied by up to **3x** for
the same call (11.0 s against 1.15 s for one backward walk). Consequently:

* `exact_evals`, `candidates`, `certified` and the bound excess are **facts**;
* the `seconds` columns are **indicative**, and `scripts/certificate_report.py`
  says so in its own header.

## 1. The cost of the certificate

This is the metric the package's design actually rests on. Candidates are ranked
by a valid upper bound and evaluated exactly in decreasing order, stopping as
soon as the best realized value beats the largest remaining bound. That is only
worth something if the bound prunes, so the quantity to publish is

```
cost of the certificate = (candidates evaluated exactly) / (candidates available)
```

against a naive E-optimal greedy, which evaluates **every** candidate at every
step (`M - k` in the forward direction, `k` in the backward one).

### 1.1 Forward: a constant number of evaluations per step

`N=400`, `K=1..50`, AR(1) `rho=0.9`, certified mode (`forward_top=0`):

| M | exact / candidates | mean exact evals per step | mean candidates per step | seconds |
|---|---|---|---|---|
| 200 | 4.57 % | 8.00 | 175 | 0.06 |
| 400 | 2.13 % | 8.00 | 375 | 0.09 |
| 800 | 1.03 % | 8.00 | 775 | 0.13 |
| 1600 | 0.51 % | 8.00 | 1575 | 0.24 |
| 3200 | 0.25 % | 8.00 | 3175 | 0.50 |

The mean number of exact evaluations per step is **8.00 - the batch size -
independently of M**, while the available candidates grow linearly. The relative
cost of the certificate therefore decays as `1/M`. On every one of these walks
`certified` is `true` for 100 % of the steps, and the walk reproduces the
exhaustive `eigvalsh` greedy to `<= 1e-15` (the self-check printed by
`scripts/compare_cssp.py`).

This is the strongest empirical statement the package can make, and it is the
reason the forward certified mode is the default recommendation.

### 1.2 Backward: the bound is weak, and that is documented

`M=200 -> kmin=20`, 180 certified steps, AR(1), sweep over `low_rank` (the `p` of
the Temple bound):

| `p` | certified | exact / candidates | worst step | mean evals per step | bound excess at winner | seconds |
|---|---|---|---|---|---|---|
| 1 | 100 % | 82.92 % | 100.0 % | 91.62 | 155.28 % | 2.80 |
| 2 | 100 % | 53.74 % | 90.0 % | 59.38 | 87.29 % | 3.41 |
| 4 | 100 % | 40.82 % | 87.0 % | 45.11 | 23.33 % | 9.25 |
| 8 | 100 % | 33.58 % | 83.9 % | 37.11 | 11.94 % | 3.07 |
| 16 | 100 % | 28.84 % | 71.2 % | 31.87 | 8.90 % | 5.44 |
| 32 | 100 % | 26.34 % | 69.3 % | 29.11 | 7.43 % | 9.37 |

Every step is still certified, but the pruning is weak: at the default `p = 4`
the cascade still evaluates **40.8 %** of the candidates, and even `p = 32` only
brings it to 26.3 %. The bound excess (how far the retained upper bound sits
above the realized value, at the winner) decays roughly as `1/p`, from 155 % at
`p = 1` to 7.4 % at `p = 32` - consistent with the `1/p` decay claimed in
`docs/algorithm.md` section 12, which measures it against the range of the
candidate values instead (40.6 % at `p = 4`, i.e. the same `p = 4` behaviour).

Consequence, stated plainly: **the backward cascade saves a factor of about
2.5-3.8 in exact evaluations, not orders of magnitude.** Its value is the
certified nested family and the cheap warm starts, not the pruning. The forward
direction is where the certificate pays.

The `seconds` column here is the unreliable one, and visibly so: `p = 4` at 9.25 s
against `p = 8` at 3.07 s is not a real ordering (the higher `p` does strictly
more work). Read the counters, not the clock.

### 1.3 The prefilter is not a cheapened certificate

`M=200`, `K=1..50`:

| mode | certified | exact / candidates | mean evals per step | mean candidates per step |
|---|---|---|---|---|
| certified (`forward_top=0`) | **100 %** | **4.57 %** | **8.00** | 175.0 |
| `prefilter=True`, width 16 | 0 % | 9.14 % | 16.00 | 175.0 |
| `prefilter=True`, width 32 | 0 % | 18.29 % | 32.00 | 175.0 |

The prefilter evaluates *more* candidates than the certified mode (16 or 32
against 8) and gives up the guarantee, so at this size it is strictly dominated.
Its advantage appears only at large `M`, where the exact bound (`p = k`, hence a
full spectrum of `R_S` per step) becomes the dominant cost and a fixed-width
scan becomes cheaper. That is a real trade-off, but it is not "the certified mode
with less work" - it is a different algorithm, and the README's 4-12 % loss of
`lambda_min` is the price.

## 2. Quality against the classical competitors

### 2.1 The scale-free metric

Cauchy interlacing gives an algorithm-independent ceiling: for `|S| = k`,
`lambda_min(R_S) <= lambda_{M-k+1}(R) = sigma_k(Z)^2`. So quality is reported as

```
eta(S) = lambda_min(R_S) / sigma_k(Z)^2      in (0, 1], comparable across datasets
```

`N=400`, `M=200`, `K = 5 / 15 / 30`, AR(1) `rho=0.9` for the correlated cases.
`n/a*` marks a cell where the threshold filter could not produce a subset of
exactly that size, so the comparison would not be like-for-like.

### 2.2 iid (no correlation structure)

| method | K=5 | K=15 | K=30 |
|---|---|---|---|
| `sifters` forward | 0.375 | 0.375 | **0.385** |
| `sifters` forward, 8 seeds | 0.378 | 0.380 | **0.393** |
| `sifters` backward | 0.347 | 0.338 | 0.348 |
| `sifters` prefilter | 0.373 | 0.378 | 0.376 |
| QRCP = pivoted Cholesky = D-opt | 0.373 | 0.373 | 0.363 |
| strong RRQR (`f=1`) | 0.373 | 0.373 | 0.363 |
| VIF top-k | 0.345 | 0.320 | 0.345 |
| min-max correlation greedy | **0.379** | 0.366 | 0.368 |
| threshold filter | **0.380** | 0.358 | 0.331 |
| uniform sample (mean of 10) | 0.331 | 0.315 | 0.299 |
| Fedorov exchange | **0.381** | 0.357 | 0.356 |
| exhaustive `eigvalsh` oracle | 0.375 | 0.375 | 0.385 |

### 2.3 AR(1) `rho=0.9` (very regular structure)

| method | K=5 | K=15 | K=30 |
|---|---|---|---|
| `sifters` forward | 0.074 | 0.146 | 0.252 |
| `sifters` backward | 0.070 | 0.102 | 0.171 |
| `sifters` prefilter | 0.071 | 0.138 | 0.246 |
| QRCP = pivoted Cholesky = D-opt | 0.072 | **0.161** | **0.258** |
| VIF top-k | 0.065 | 0.029 | 0.060 |
| min-max correlation greedy | 0.073 | 0.125 | **0.258** |
| threshold filter | n/a* | 0.153 | 0.248 |
| uniform sample | 0.037 | 0.029 | 0.066 |
| Fedorov exchange | **0.075** | **0.166** | 0.155 |

### 2.4 Blocks

| method | K=5 | K=15 | K=30 |
|---|---|---|---|
| `sifters` forward | 0.044 | **0.431** | **0.407** |
| `sifters` forward, 8 seeds | 0.043 | 0.428 | **0.414** |
| `sifters` backward | 0.041 | 0.357 | 0.350 |
| QRCP = pivoted Cholesky = D-opt | 0.042 | 0.392 | 0.337 |
| VIF top-k | 0.005 | 0.363 | 0.356 |
| min-max correlation greedy | 0.043 | 0.388 | 0.316 |
| threshold filter | 0.041 | n/a* | 0.354 |
| uniform sample | 0.011 | 0.312 | 0.297 |
| Fedorov exchange | **0.044** | 0.398 | 0.355 |

### 2.5 Factors (rank 6 latent structure)

| method | K=5 | K=15 | K=30 |
|---|---|---|---|
| `sifters` forward | 0.036 | **0.557** | 0.540 |
| `sifters` prefilter | 0.033 | 0.405 | 0.405 |
| `sifters` backward | 0.029 | 0.399 | 0.345 |
| QRCP = pivoted Cholesky = D-opt | 0.032 | 0.455 | 0.463 |
| VIF top-k | 0.027 | 0.518 | **0.560** |
| min-max correlation greedy | 0.027 | 0.328 | 0.305 |
| threshold filter | 0.030 | n/a* | 0.173 |
| uniform sample | 0.011 | 0.166 | 0.157 |
| Fedorov exchange | **0.037** | 0.318 | 0.236 |

### 2.6 The honest reading

**No method dominates, and the README says so rather than hiding it.**

* **`sifters` forward wins on the structures it is meant for.** Factors:
  `0.557` against `0.455` for QRCP (+22 %) and `0.318` for Fedorov (+75 %) at
  `K=15`; blocks: `0.431` against `0.392` (+10 %) and `0.398` (+8 %) at `K=15`,
  and `0.407` against `0.337` (+21 %) and `0.355` (+15 %) at `K=30`. On iid it is
  the best or within 2 % of the best at every K.
* **It loses on very regular structures.** On AR(1) the cheap methods win:
  QRCP gets `0.161`/`0.258` against `0.146`/`0.252`, Fedorov `0.166` at `K=15`,
  and the threshold filter `0.153`. For an AR(1) chain the well-conditioned subset
  is essentially an arithmetic progression of indices, which an index-based
  threshold finds directly - the structure is doing the work, not the algorithm.
* **`K=5` is decided by the tie, not by the criterion.** All five singletons have
  `lambda_min = 1`, so the first pick is not guided by the objective (this is why
  `forward_seeds` exists). Fedorov and the min-max greedy win at `K=5`
  everywhere, by 3-8 %.
* **VIF top-k is dominated.** The textbook multicollinearity diagnostic is beaten
  by column-pivoted QR on three of four structures, and catastrophically so on
  blocks at `K=5` (`0.005` against `0.042`). Its single win is factors at `K=30`
  (`0.560` against `0.540`). See section 3 for *why* VIF cannot guarantee
  anything about a subset.
* **Uniform sampling is far behind** (`0.011` to `0.037` at `K=5`). This is the
  randomized CSSP sampler in this geometry: because `sifters` normalizes every
  column to unit norm before looking at the Gram, the norm-based sampling
  probabilities are all equal.
* **The certified walk is exactly the exhaustive greedy.** On all four datasets
  and all three sizes, `|lambda_min(sifters forward) - oracle| <= 8.9e-16`. The
  certificate costs nothing in quality, which is the whole point.

## 3. The strong-RRQR guarantee is valid but non-binding

Gu & Eisenstat's strong RRQR returns `k` columns with
`sigma_i(Z_S) >= sigma_i(Z) / sqrt(1 + f^2 k (M-k))`, hence
`lambda_min(R_S) >= sigma_k(Z)^2 / (1 + f^2 k (M-k))` - a global, deterministic,
multiplicative guarantee on the same objective. Measured at `f = 1`, `M = 200`:

| structure | K | bound | achieved | achieved / bound | `eta` guaranteed | `eta` achieved |
|---|---|---|---|---|---|---|
| iid | 5 | 2.66e-03 | 0.968 | 364x | 0.0010 | 0.373 |
| iid | 15 | 8.16e-04 | 0.845 | 1035x | 0.0004 | 0.373 |
| iid | 30 | 3.68e-04 | 0.682 | 1850x | 0.0002 | 0.363 |
| AR(1) | 5 | 1.33e-02 | 0.930 | 70x | 0.0010 | 0.072 |
| AR(1) | 15 | 1.25e-03 | 0.555 | 446x | 0.0004 | 0.161 |
| AR(1) | 30 | 2.03e-04 | 0.267 | 1315x | 0.0002 | 0.258 |
| blocks | 5 | 2.21e-02 | 0.905 | 41x | 0.0010 | 0.042 |
| blocks | 30 | 3.65e-05 | 0.063 | 1719x | 0.0002 | 0.337 |
| factors | 5 | 2.35e-02 | 0.743 | 32x | 0.0010 | 0.032 |
| factors | 30 | 7.62e-05 | 0.180 | 2359x | 0.0002 | 0.463 |

The `k(M-k)` factor inside the square root is what kills it at these sizes: the
guarantee is `eta >= 1/(1 + k(M-k))`, i.e. `<= 0.001`, where the algorithm
actually reaches `0.03` to `0.46`.

Two things this does *not* mean. First, the guarantee is **valid** - asserted by
`tests/test_cssp_baselines.py::test_rrqr_bound_never_exceeds_what_rrqr_achieves`
across 240 structure/size/`f` combinations, after making the bound
*unconditionally* valid by evaluating it at the growth factor actually achieved
(`f_eff = max(f, max|R11^-1 R12|)`) rather than at the requested `f`, which an
unreachable `f` would invalidate. Second, a vacuous worst-case bound does not
make QRCP a bad algorithm: it is 10x faster than `sifters` and within 10-25 % of
it on efficiency in most cells. The two guarantees are simply not substitutes -
one is a worst-case statement that does not bite, the other a statement about the
cost of exactness at each step.

### VIF, precisely

For PSD `A`, `lambda_max(A) >= max_i A_ii`. With `A = R^-1`:

```
lambda_min(R) = 1 / lambda_max(R^-1) <= 1 / max_j VIF_j
```

This is an **upper** bound: a large VIF *certifies* that `R` is badly conditioned,
which is what a collinearity diagnostic is for. A small VIF certifies nothing, and
the inequality is strict on all four test structures (0.289 against 0.676 on one
AR(1) draw). It is also a statement about the whole `R`: `(R_S)^-1` is not the
submatrix of `R^-1`, so no VIF bounds `lambda_min(R_S)`. Measured and asserted in
`tests/test_cssp_baselines.py::test_vif_only_certifies_ill_conditioning`.

## 4. Timing (indicative, best of 3, isolated)

`N=400`, `M=200`, AR(1), release build. `sifters` rows are a **single nested path**
answering every K in the visited range, not a single request.

| method | K=5 | K=15 | K=30 | scope |
|---|---|---|---|---|
| QRCP = pivoted Cholesky = D-opt | 0.0008 | 0.0025 | 0.0052 | one subset |
| strong RRQR (`f=1`) | 0.0028 | 0.0046 | 0.0077 | one subset |
| VIF top-k | 0.0005 | 0.0004 | 0.0004 | one subset |
| `sifters` forward (certified) | 0.019 | 0.019 | 0.019 | path K=1..30 |
| `sifters` prefilter | 0.017 | 0.017 | 0.017 | path K=1..30 |
| `sifters` backward | 0.94 | 0.94 | 0.94 | path K=5..200 |
| Fedorov exchange | 0.12 | 0.81 | 6.2 | one subset |
| exhaustive `eigvalsh` oracle | 0.088 | 0.095 | 0.087 | path K=1..30 |

Read this carefully before quoting a speedup:

* **QRCP is 10x faster than `sifters` forward** for a single subset at this size,
  and the `sifters` row buys the whole nested family plus a certificate. At
  `M=200` the pragmatic choice is column-pivoted QR unless the certificate, the
  family or a non-linear dependence matrix is needed.
* **`sifters` forward is 46x cheaper than the exhaustive greedy** it reproduces
  (0.019 s against 0.088 s), and 330x cheaper than Fedorov at `K=30` (0.019 s
  against 6.2 s) while being the only one of the three that is *proven* identical
  to the exhaustive greedy.
* **The backward path is expensive** because it visits `K = 5..200`, i.e. 195
  steps, not 30. It is not comparable to the forward row; the table says so at the
  point of use.

## 5. Reproducibility ledger

Reproduced with `python3 scripts/reproduce_speed.py` (best of 3, one warmup call,
library defaults, release build, generic CPU - no `-C target-cpu=native`), which
writes `docs/measurements/speed_ledger.json`. The `exact_evals` column is the
honest part of the comparison: it is deterministic, so if it does not match, the
discrepancy is not environmental.

| README row | README | measured | ratio | `exact_evals` | README evals |
|---|---|---|---|---|---|
| forward, `M=400`, `K=50` | 0.2 s | **0.053 s** | 0.27x | 392 | 392 (match) |
| forward, `M=3200`, `K=50` | 0.8 s | **0.499 s** | 0.62x | 392 | 392 (match) |
| forward, `M=1600`, `K=150` | 8.4 s | **1.139 s** | 0.14x | 1192 | 1192 (match) |
| forward, `M=10000`, `N=50`, `K=100`, prefilter | 0.74 s | 0.766 s | 1.04x | 1584 | 1584 (match) |
| full backward, `M=200` | 0.85 s | 1.319 s | 1.55x | **8027** | 6235 (**mismatch**) |
| backward, `M=500`, `kmin=200`, `eval="inverse"` | 14.6 s | 28.048 s | 1.92x | **56640** | 38392 (**mismatch**) |

### What reproduces

**The whole forward half, and it is conservative.** The four forward rows
reproduce, three of them substantially *faster* than the README claims (0.14x to
0.62x), and their `exact_evals` match the table exactly. A generic-CPU build
being faster than a `target-cpu=native` one is not a contradiction: the `native`
timings were single-shot without a warmup, and the first call in a process pays
for the rayon pool, page faults and cold caches. With a warmup and best-of-3 the
forward claims hold with margin.

### What does not reproduce

* **The two backward rows, and not because of the machine.** `exact_evals` is a
  deterministic count of work performed, and the package performs **29 % more**
  (8027 against 6235) and **48 % more** (56640 against 38392) evaluations than the
  README table states. A count cannot differ because of load or cache warmth, so
  those two rows were measured with a different configuration or on a different
  revision. The times follow (1.55x and 1.92x). Until the configuration is
  identified, **the backward half of the speed table should not be quoted**; the
  forward half is the trustworthy one.
* **The `x24` to `x30` speedups and the "before" column of the gain table.** The
  git history contains three commits, all from the same day, and the oldest one is
  *already* the optimization commit. The pre-optimization revision is absent from
  the history, so `bench/compare.py` has no reference revision to check out and
  those relative gains cannot be reproduced by anyone, including the author. They
  are historical measurements and should be read as such or dropped.
  `scripts/reproduce_speed.py` records them in its output as
  `unreproducible`, with the reason, rather than quietly omitting them.
* **Every timing taken before the release rebuild in this working tree.** The
  installed extension was a debug build; see trap 1. This alone accounts for a
  factor of ~14 and is why `sifters.build_profile()` exists.

### Known-good invariants (asserted, not just measured)

* QRCP, pivoted Cholesky and the exact D-optimal greedy share one pivot rule -
  asserted pivot by pivot against brute force.
* `lambda_min` of the certified forward walk equals the exhaustive `eigvalsh`
  oracle to `<= 8.9e-16` on all four structures and all tested K.
* The strong-RRQR floor never exceeds what strong RRQR achieves, over 240
  structure / size / `f` combinations.
* The Fedorov output is a local optimum: no improving 1-for-1 swap.
* `verify=True` changes **no counter and no value**. Isolated best-of-3, all
  bit-identical (`exact_evals` 8120 vs 8120, `lanczos_iters` 242130 vs 242130,
  `lambda_min` equal to 12 digits), at a cost of **+9 %** on the backward walk and
  **+8 %** on the forward one. It is a correctness check, not a search
  accelerator.
