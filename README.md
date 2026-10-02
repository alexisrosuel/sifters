# sifters - spectral feature selection, in Rust

[![CI](https://github.com/alexisrosuel/sifters/actions/workflows/ci.yml/badge.svg)](https://github.com/alexisrosuel/sifters/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/alexisrosuel/sifters#license)

**`sifters` picks the best-conditioned subset of variables** - for each size K, the
K features whose correlation matrix has the largest smallest eigenvalue
`lambda_min(R_S)`.

In practice it **kills multicollinearity**: when `X` has near-collinear columns,
regression and downstream models become unstable and their coefficients hard to
interpret; `sifters` returns the subset that stays as far as possible from
singularity, for every K, with no labels required.

This is the min-eigenvalue member of the D-optimal / DPP /
column-subset-selection family - the same greedy structure as maximum-volume
selection, but the objective is the worst-case direction. Optimal design theory
calls this criterion **E-optimal** (maximize the minimum eigenvalue of the
information matrix); on a correlation matrix it reads `sigma_min(Z_S)^2`.

NP-hard problem -> **nested** families built by greedy, but a greedy
that affords itself the luxury of being **exact and certified**:

| | |
|---|---|
| 🎯 **Quality** | secular criterion (and not a mere correlation), up to **+82 %** of `lambda_min` at fixed K vs the naive correlation filter |
| ✅ **Certificate** | **both directions** of the traversal are certified: backward elimination *and* forward selection stop on a valid upper bound per candidate |
| 🏁 **Exact mode** | `select(..., exact=True)` **proves** the optimum at the requested K by branch and bound, or returns a certified optimality gap |
| ⚡ **Speed** | **10 000 variables -> 100 in 0.7 s** (prefilter); the exact certified greedy is **x24 to x30 faster** in forward selection, **x2.5 to x3.5** in backward elimination |
| 🦀 **Safe Rust** | `#![forbid(unsafe_code)]`, multicore (`rayon`), zero heavy dependencies |
| 🐍 **Python** | `import sifters; sifters.select(X, k=50)` |

<p align="center">
  <img src="docs/img/fig_qualite.png" alt="Quality: lambda_min at fixed K, sifters vs naive heuristics" width="100%">
</p>

For an equal number of variables, `sifters` keeps a better conditioned subset than the
"min-max correlation" greedy or the threshold filter - the gap widens when the
latent structure is rich (factors: **+82 %**, blocks: +15 %, AR(1): +9 %).

<p align="center">
  <img src="docs/img/fig_performance.png" alt="Performance: scaling and evaluation by the inverse" width="100%">
</p>

---

## 1. Installation

The core is a Rust library; usage is through the Python extension (PyO3
binding), which is the only public interface.

```bash
pip install .                  # builds the extension via maturin -> import sifters
# or, in development:
maturin develop --release
```

`numpy` is not required at runtime: when present it is only used to convert
lists and DataFrames into a contiguous `float64` buffer.

The wheels are built for a generic CPU. For a local speedup, compile for the host
CPU explicitly (do not redistribute such a wheel):

```bash
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

> The timings in section 4 were measured with `-C target-cpu=native` on an Apple
> M1 Max (10 cores).

### Development environment

The project is managed with **pixi** (conda-forge) for the Python environment,
**Cargo** for the Rust crate and **maturin** (PyO3) for the extension. The pixi
tasks mirror the CI gates:

```bash
pixi install            # Python 3.14 + numpy, matplotlib, maturin, pytest, ruff, mypy
pixi run build          # maturin develop --release
pixi run test           # cargo test
pixi run test-py        # pytest tests (run `pixi run build` first)
pixi run lint           # cargo clippy --workspace --all-targets -- -D warnings
pixi run lint-py        # ruff check .
pixi run type-py        # mypy scripts tests
```

The library unit tests (bounds, brute force, inverse, implicit replica, ...)
run standalone with a Rust toolchain:

```bash
cargo test --release
```

## 2. Quick start

```python
import numpy as np, sifters

X = np.random.default_rng(0).normal(size=(500, 300))   # (observations, variables)

r = sifters.select(X, k=30)                 # direction="auto"
print(r.subset, r.lambda_min, r.seconds)

curve = sifters.curve(X, kmin=5)            # {K: lambda_min} for the whole family
family = sifters.path(X, direction="forward", kmax=100)
family.subset_at(42)                      # the subset of size 42

# prefilter (heuristic): only evaluates the top-16 per step, ~4 to 12 % of lambda_min
fast = sifters.select(X, k=30, prefilter=True)

# multi-start: tries 4 starting variables and keeps the best family
# (the gap to the global optimum drops from ~11 % to ~2 % on an AR(1) structure, section 4)
robust = sifters.select(X, k=30, forward_seeds=4)

# exact mode: proves the optimum at k (branch and bound over all k-subsets)
proof = sifters.select(X, k=30, forward_seeds=8, exact=True)
proof.proved_optimal, proof.gap_certified, proof.exact_evals
```

The engine is a native extension: numpy arrays are read through the buffer
protocol, the GIL is released during the computation, and `numpy` remains optional.

The Python benchmarks (scenarios, A/B between revisions) are described in
[`bench/README.md`](bench/README.md).

## 3. What `sifters` returns

A **nested family** `S_M > S_{M-1} > ... > S_1` (mode `backward`) or
`S_1 < S_2 < ... < S_kmax` (mode `forward`), hence a response for *all* sizes K
in a single computation, plus:

* `curve`: `lambda_min` for each K, revalidated by a strict cold Lanczos;
* `order`: the order in which variables are removed/added (allows reconstructing any
  subset);
* per step: upper bound of the retained candidate, number of candidates evaluated, flag
  `certified`, residual, Lanczos iterations.

On the Python side, `select` and `path` return a `sifters.Selection` object:

```python
r.k, r.subset, r.lambda_min       # retained result (original indices, increasing)
r.curve                           # {K: lambda_min} - the whole family
r.subset_at(42), r.lambda_at(42)  # any visited size, without recomputation
r.steps                           # list[Step]: bounds, candidates, certification, residuals
r.certified_steps, r.total_exact_evals, r.total_lanczos_iters
r.seconds, r.path_seconds, r.load_seconds, r.representation, r.eval
r.to_dict()                       # same content, JSON-serializable
```

`progress=callback` is called after each step - returning `False` cleanly interrupts
the traversal (partial result) - `threads=N` sets the size of the Rayon pool,
and `Ctrl-C` is intercepted during the computation.

## 4. Measured results

10-core machine, `-C target-cpu=native`, `f64`. Reproducible with
`scripts/make_figures.py`, `scripts/compare_baseline.py` and `scripts/bench_numpy.py`.

### Quality (exact lambda_min, computed in numpy)

| dataset | K | **sifters forward** | min-max corr. greedy | threshold filter |
|---|---|---|---|---|
| blocks (N=500, M=250) | 10 | **0.416** | 0.404 | 0.407 |
| | 30 | **0.317** | 0.266 | 0.277 |
| | 60 | **0.241** | 0.197 | 0.193 |
| AR(1) rho=0.9 | 10 | 0.794 | **0.811** | 0.798 |
| | 30 | 0.306 | 0.279 | **0.324** |
| | 60 | **0.123** | 0.103 | 0.123 |
| factors (rank 20) | 10 | **0.655** | 0.552 | 0.614 |
| | 30 | **0.057** | 0.033 | 0.039 |
| | 60 | **0.036** | 0.021 | 0.021 |

**Honest reading**: the threshold filter is excellent and almost free (1-40 ms); it
remains competitive on a very regular structure (AR(1) at small K). `sifters` wins in the
majority of cases, very widely when the latent structure is rich, and additionally brings
certification and the complete family.

### Speed

The table gives the **default exact mode** (no prefilter), that is, the
**certified** greedy traversal; `N=500` (forward) / `600` (backward), 10-core machine.

| configuration | time | candidates evaluated exactly |
|---|---|---|
| forward, M=400, K=50 | **0.2 s** | 392 |
| forward, M=3 200, K=50 | **0.8 s** | 392 |
| forward, M=1 600, K=150 | **8.4 s** | 1 192 |
| forward, M=10 000 (N=50), K=100, `prefilter=True` | 0.74 s | 1 584 |
| full backward, M=200 | **0.85 s** | 6 235 |
| backward, M=500, kmin=200, `eval="inverse"` | **14.6 s** | 38 392 |
| same, `eval="direct"` **converged** (300 iter.) | **102 s** | 33 976 |

**Measured gain vs the original version** (same inputs, `verify=False`, `max_exact=0`,
`forward_top=0`):

| scenario | before | after | gain (1 core) |
|---|---|---|---|
| forward, M=400, K=50 | 5.11 s | 0.21 s | **x24** |
| forward, M=800, K=50 | 9.98 s | 0.36 s | **x28** |
| forward, M=3 200, K=50 | 36.8 s | 1.22 s | **x30** |
| forward, M=1 600, K=150 | 266 s | 10.3 s | **x26** |
| full backward, M=200 | 8.75 s | 3.16 s | **x2.8** |
| backward, M=500, kmin=200, `inverse` | 193 s | 76.7 s | **x2.5** |
| backward, M=500, kmin=200, `direct` converged | 1835 s | 624 s | **x2.9** |

What explains it:

* **forward selection is certified**: it no longer evaluates the `M-K` candidates, but
  stops as soon as a valid upper bound allows it. Better still, in exact mode it uses the
  **full spectrum** of `R_S`, which makes the secular bound *exact*: the
  evaluations drop to the floor of the batch (8 per step) and we go from 2.4 M to 392
  Lanczos iterations on `fwd3200`;
* **the bordered operator** exploits the materialized correlation (`O(k^2)` instead of
  `O(Nk)` per matvec);
* **the Lanczos solver** is lightened: Ritz vector by inverse iteration (instead of
  an `O(m^3)` diagonalization at each call), second conditional reorthogonalization
  pass, stopping on the Ritz **residual**, Sturm sequence with early exit;
* **the deletion secular seed is finally used** (`deletion_seed` no longer
  restricted it to `R^{k-1}`, the solver silently rejected it): ~30 %
  fewer iterations per candidate;
* the maintained inverse matvec no longer has a per-element branch.

Details and measurement protocol (history, CLI binary) in
[`bench/final_report.md`](bench/final_report.md); the current benchmarks, in
Python, are in [`bench/README.md`](bench/README.md).

### Speed vs a naive 100 % Python / numpy implementation

`scripts/bench_numpy.py` times `sifters` against a **naive E-optimal greedy that
attacks the same problem**, entirely in numpy: at each step, *all* the candidates
remaining are re-evaluated by recomputing `lambda_min` with a complete
`numpy.linalg.eigvalsh` decomposition - that is `K x M` decompositions of size `~ K`. Three
series are compared:

* **sifters without prefilter** (`prefilter=False`, **the default**) - the same
  E-optimal greedy as the naive one, certified but **without evaluating all candidates** at each
  step (sorted by decreasing secular bound); the `lambda_min` obtained remains identical to the
  naive one to within `5x10^-14` where both evaluate the same candidates;
* **sifters with prefilter** (`prefilter=True`, top-16) - only evaluates the 16 best
  candidates of the secular score: this is a heuristic, enabled **explicitly** by
  the caller (never automatically according to size);
* **naive numpy** - the `eigvalsh` reference.

The time gap between series 1 and 2 therefore measures exactly what the prefilter
brings (and what it costs in optimality).

<p align="center">
  <img src="docs/img/fig_temps_vs_numpy.png" alt="Time: sifters (Rust) vs naive E-optimal greedy in numpy" width="100%">
</p>

> ⚠️ **Tables below measured before the optimization of the exact path**: the column
> "sifters exact" there is overestimated by a factor of ~2.5 (see section 4 "Speed" and
> [`bench/final_report.md`](bench/final_report.md)). The prefilter and naive
> numpy columns remain valid; `scripts/bench_numpy.py` can refresh the whole set.

Sweep over the number of variables (`N=500`, `K=50`):

| M | sifters prefilter | sifters exact (default) | naive numpy | naive / prefilter |
|---|---|---|---|---|
| 200 | **0.11 s** | 0.46 s | 0.28 s | x2.5 |
| 400 | **0.12 s** | 0.87 s | 0.62 s | x5.0 |
| 800 | **0.13 s** | 1.54 s | 1.27 s | x9.6 |
| 1 600 | **0.19 s** | 3.05 s | 2.72 s | x14 |
| 3 200 | **0.41 s** | 6.06 s | 5.44 s | x13 |
| 6 400 | **1.18 s** | 12.0 s | 10.9 s | x9.3 |

Sweep over subset size (`N=500`, `M=800`):

| K | sifters prefilter | sifters exact (default) | naive numpy | naive / prefilter |
|---|---|---|---|---|
| 10 | **0.03 s** | 0.06 s | 0.06 s | x2.0 |
| 20 | **0.04 s** | 0.24 s | 0.17 s | x4.3 |
| 40 | **0.10 s** | 1.01 s | 0.72 s | x7.0 |
| 80 | **0.36 s** | 4.57 s | 4.49 s | x13 |
| 120 | **0.94 s** | 11.3 s | 14.0 s | x15 |

What the prefilter costs in quality (panel (c): `lambda_min` prefiltered vs exact):

| `M` (K=50) | 200 | 400 | 800 | 1 600 | 3 200 | 6 400 |
|---|---|---|---|---|---|---|
| loss of `lambda_min` | -4.1 % | -3.8 % | -6.0 % | -8.1 % | -10.0 % | -11.6 % |

| `K` (M=800) | 10 | 20 | 40 | 80 | 120 |
|---|---|---|---|---|---|
| loss of `lambda_min` | -6.1 % | -4.9 % | -8.5 % | -6.6 % | -5.6 % |

**Honest reading** - the naive one is not a scarecrow:

1. **With a strictly equal algorithm, numpy holds its own - before the optimization of the
   exact path.** When `sifters` evaluated *all* the candidates like the naive one, it was
   on equal footing: slower at small `M`/`K`, barely ahead at `K=120` (11.3 s versus 14.0 s).
   `numpy.linalg.eigvalsh` (LAPACK) is very optimized; since then, the exact mode of
   `sifters` **no longer evaluates all the candidates** (certification by secular bound) and
   beats it clearly, but the speedup still does not come from a better spectral
   solver: it comes from the fact that it evaluates *far fewer candidates* (bounds,
   certification in both directions, maintained inverse, and prefilter if enabled).
2. **The prefilter speeds up everywhere, but is paid for in quality.** It divides the time of the
   exact greedy by 2 to 16 depending on the configuration, at the cost of **4 to 12 %** of `lambda_min`
   - worst case measured: -11.6 % at `M = 6 400`. It is **never enabled automatically**:
   it is an explicit choice (`prefilter=True`), including
   at small `M` since the old guard "exact below 512" has disappeared.
   As the switch depends on the problem, it is up to the user to decide.
3. **Bottom line**: the naive `eigvalsh` is an excellent choice for small problems; the
   default mode of `sifters` (exact greedy) is the right trade-off as soon as `M` grows or
   when one wants the whole certified nested family in a single computation. The prefilter is
   there when one prefers speed to a few percent of `lambda_min`.

### Gap to the global optimum (measured by brute force)

The greedy is myopic: "certified" qualifies each **step**, not the subset. To
quantify the gap, `sifters` was compared to the global optimum obtained by enumerating **all**
the `C(M,K)` combinations (`M <= 30`, `N = 400`, average of 3 seeds):

| structure | K | `forward_seeds=1` | `=2` | `=4` | `=8` | all (`=M`) | backward |
|---|---|---|---|---|---|---|---|
| iid | 3 | 0.83 % | 0.55 % | 0.30 % | 0.17 % | **0.01 %** | 2.10 % |
| iid | 8 | 1.64 % | 1.17 % | 0.97 % | 0.72 % | **0.36 %** | 4.11 % |
| AR(1) rho=0.9 | 3 | 10.93 % | 4.23 % | 2.11 % | 1.01 % | **0.02 %** | 16.68 % |
| AR(1) rho=0.9 | 5 | 11.91 % | 7.35 % | 5.84 % | 4.12 % | **1.37 %** | 26.94 % |
| AR(1) rho=0.9 | 8 | 12.86 % | 9.90 % | 7.89 % | 6.73 % | **4.29 %** | 23.84 % |

<p align="center">
  <img src="docs/img/fig_ecart_seeds.png" alt="Gap to the global optimum according to the number of starts in forward selection" width="100%">
</p>

Three lessons:

* **the gap does not diverge with M**: it rises from `M ~ 10` to `M ~ 20`, then plateaus. The
  determining parameter is `M - K` (the number of variables "in excess"), not `M`;
* **most of the forward gap comes from the initial seed**, not from the greedy rule:
  all single variables have `lambda_min = 1`, so the first step is not guided by
  the objective. Hence `forward_seeds`, which brings 10.9 % -> 0.02 % at `K = 3` on AR(1);
* **backward elimination remains far behind** in quality (10 to 30 %): its myopia is
  intrinsic, unlike that of the forward start.

### Exact mode: proving the optimum at K

`select(X, k=..., exact=True)` closes the gap: a branch and bound explores the
`k`-subsets and either **proves** the optimum or returns a **certified optimality
gap**.

The bound costs nothing. `lambda_min` is monotonically non-increasing as the set
grows (Cauchy interlacing: `R_S` is a principal submatrix of `R_T` for
`S subset T`, hence `lambda_min(R_S) >= lambda_min(R_T)`). The value of a partial
set therefore upper-bounds every completion of it, and the search only ever
evaluates `f(S + {j})` - exactly the quantity it needs to order the children. The
greedy family then supplies the incumbent, which is what makes the proof cheap.

Time to prove optimality (N=400, incumbent `forward_seeds=8`, 120 s budget,
`bench/exact.py`):

| structure | K | M=20 | M=30 | M=40 | M=60 | M=80 |
|---|---|---|---|---|---|---|
| iid | 5 | 0.03 s | 0.03 s | 0.04 s | 0.11 s | 0.20 s |
| iid | 8 | 0.21 s | 0.79 s | 1.3 s | 4.7 s | 13.4 s |
| iid | 12 | 4.9 s | 55.6 s | *gap 13.7 %* | *12.4 %* | *12.1 %* |
| AR(1) rho=0.9 | 5 | 0.04 s | 0.09 s | 0.25 s | 1.2 s | 1.1 s |
| AR(1) rho=0.9 | 8 | 0.28 s | 1.6 s | 8.0 s | 43.5 s | *gap 0.77 %* |
| AR(1) rho=0.9 | 12 | 2.4 s | 50.6 s | *gap 4.5 %* | *3.0 %* | *2.0 %* |

*Italic* cells hit the 120 s budget: the search returns its best subset together
with the relative gap that still separates it from the optimum, so the answer
stays usable and honest.

<p align="center">
  <img src="docs/img/fig_exact.png" alt="Exact mode: time to prove optimality at K, for K = 5, 8 and 12" width="100%">
</p>

Two practical points:

* the exact search does not only certify, it **improves**: on AR(1) it gains
  between +0 % and +20 % of `lambda_min` over the multi-start greedy (for
  instance +19.7 % at `M = 80, K = 8`, where the proof times out but the
  incumbent is already much better);
* `K` is the cost driver, far more than `M`: on AR(1), `K = 12` at `M = 80` is
  as expensive as `K = 12` at `M = 40`. For `K <= 8` and `M <= 80` the proof
  lands in seconds.

## 5. How it works

> Detailed version, formulas and complete invariants: **[`docs/algorithm.md`](docs/algorithm.md)**.

### 5.1 The right criterion: the secular equation

The eigenvalues of the submatrix with index `i` removed are **exactly** the
roots of (KKT conditions of `min x^T R x` under `x_i = 0`):

```
S_i(mu) = sum_l u_l(i)^2 / (lambda_l - mu) = 0 ,      smallest root in (lambda_1, lambda_2)
exact eigenvector:  x = (R - mu I)^-1 e_i
```

For an **addition**, the bordered matrix `[[R_S, c],[c^T, 1]]` gives
`1 - mu = sum_l (u_l x c)^2/(lambda_l - mu)` and `x = [-(R_S - mu I)^-1 c ; 1]`. These two identities
provide both the **ranking criterion**, the **bounds** and the **warm starts** of
Lanczos - this is what distinguishes `sifters` from a simple correlation filter:
the first-order loss equals `sum_l g_l^2/(lambda_l - lambda_1)`, weighted by the proximity of the
eigen-directions, and not `|g_1|`.

### 5.2 The certified cascade (backward elimination)

```
1. y = R_S u ; rq = u^T y                              (1 matvec)
2. bound_i = min(Rayleigh_i, Temple_i) for each candidate        O(p x k)
   - Rayleigh: rho_i = (rq - 2u_i y_i + u_i^2)/(1-u_i^2)   - valid for any u, O(1)
   - Temple p-dimensional: tail lower-bounded by Cauchy-Schwarz, ~25x tighter
3. sort by DECREASING bound, evaluate exactly until
   best_value >= bound of the next candidate   ->  step CERTIFIED
```

### 5.3 Evaluation by the maintained inverse (`eval="inverse"`)

`lambda_min(R_{-i})` is the **largest** eigenvalue of `R_{-i}^-1`, and the inversion
amplifies the relative gaps at the bottom of the spectrum (cluster `{0.0050, 0.0057,...}` -> `{200, 175,...}`).
We maintain `W = R^-1` **without factorization per candidate**:

```
A^-1 = W_{-i,-i} - (1/W_ii) w w^T ,  w = W_{.,i}     (rank 1, O(k^2))
```

that is ~20 Lanczos iterations instead of 100-150, at **better** accuracy.

### 5.4 Numerical engineering

* reorthogonalized Lanczos, **explicit** orthogonality constraints for deflation
  (the blow-up by projected operator `(I-P)A` is unstable: amplification `alpha/beta`);
* invariant subspace detection (no spurious eigenvalues from amplification
  of noise);
* `representation="implicit"` (`R x = Z_S^T(Z_S x)`, `O(Nk)`) when `N << k`;
* *packed* lower triangle, blocked parallel GEMM construction, vectorizable matvec,
  variable removal in `O(k)` by row/column swap;
* `representation` / `eval` chosen automatically according to `M`, `N`, memory and definiteness.

## 6. Beyond the linear link: selecting on any dependence

Everything above maximizes `lambda_min(R_S)` where `R` is the **linear**
correlation matrix. Two variables linked by `y = x**2` are invisible to it (their
correlation is ~0 whatever the link strength). The engine itself only needs a
symmetric PSD matrix with unit diagonal, i.e. the Gram matrix of normalized
columns, so the whole machinery extends unchanged to any **dependence matrix**
`D` with `D_ij = 0` meaning "columns `i` and `j` are independent".

| method | link it sees | cost |
|---|---|---|
| `linear` | linear (historical) | `O(M^2 N)` |
| `rank` | any **monotone** link (Gaussian copula / normal scores) | `O(M^2 N)` |
| `dcor` | any link, `0` iff independent | `O(M^2 N^2)`, `O(M N^2)` memory |
| `hsic` | any link (RBF kernel), PSD by construction | `O(M^2 N^2)` |
| `nmi` | any link, histogram estimate | `O(M^2 N)` |

Maximizing `lambda_min(D_S)` then means: make the sub-joint as close as possible
to the product of its marginals. Because `trace(D_S) = K` for a unit diagonal,
`lambda_min(D_S) >= 1 - eps` squeezes every eigenvalue into
`[1 - eps, 1 + (K-1) eps]`, hence `det(D_S)` close to 1 and a Gaussian total
correlation `TC(S) = -0.5 log det(D_S) <= -K/2 * log(1 - eps)`.

```python
import sifters
D = sifters.dependence_matrix(X, method="dcor")           # (M, M), unit diagonal
r = sifters.select_from_dependence(D, k=50, exact=True)   # certified E-optimal
# end to end:
r = sifters.select_dependence(X, k=50, method="dcor")
```

`dcor` and `nmi` are not guaranteed PSD: `select_from_dependence` clips the
negative eigenvalues of `D` and rescales the diagonal to 1 before factoring
`D = B^T B` and handing `B` to the existing engine (`center=False`). Since the
engine only ever sees a Gram matrix - the same contract as `Z^T Z` - the
certification, the exact branch and bound and every option of `select` keep
working verbatim.

### Invariance to the scaling of each feature

With the default settings, all five measures are invariant to a per-feature
**positive affine rescaling** `x_j -> a_j x_j + b_j` (`a_j > 0`): the dependence
matrix `D` is unchanged (up to floating point) and so is the selected subset.
The result therefore does not depend on the units or the offset of the features
- `linear` is the usual Pearson correlation, `rank` compares ranks, `dcor` is
invariant to translation and scaling by construction, `hsic` adapts its
bandwidth with the median heuristic, and `nmi` bins uniformly over the observed
range (a scaling moves the bin edges with the data). Two caveats:

* `hsic` with an explicit `sigma` (instead of the default median heuristic)
  **loses** the invariance: the bandwidth must follow the amplitude of the
  feature.
* the two *signed* measures (`linear`, `rank`) change when a feature is
  **reflected** (`x_j -> -x_j`), because the sign of a correlation is
  meaningful. `dcor`, `hsic` and `nmi` measure the magnitude of the dependence
  and are invariant to a reflection.

Note also that the plain engine call `sifters.select(X, center=True)` (the
default) is scale-invariant because it uses the correlation matrix, whereas
`center=False` uses the cosine matrix, which is invariant to a scaling of each
column but **not** to a shift.

### A ground-truth experiment: recovering the independent features

`scripts/independence_recovery.py` checks the claim on a problem whose answer is
known by construction. `M = 19` features are generated from **4 independent
latent sources**: each source produces 4 features through a different link -
`s`, `s**2`, `tanh(2s)`, `sin(2s)` - plus noise, and 3 further features are pure
independent noise. The number of truly independent directions is therefore
**7**, and the ideal answer for `k = 7` is **exactly one feature per source**.

<p align="center">
  <img src="docs/img/fig_recovery_setup.png" alt="Generative model, distance correlation and linear correlation" width="100%">
</p>

Panel (a) is the generative model. Panel (b) is the distance-correlation matrix
of the data: it is block diagonal, ~0.9 inside a source and ~0.05 between
sources, so the dependence structure is exactly the one the selector is asked to
find. Panel (c) is the absolute linear correlation: the block of source 0 is
almost entirely dark, because `s` and `s**2` are uncorrelated and so is
`sin(2s)`. A correlation-based criterion sees those features as independent and
has no reason to keep only one of them - this is the trap.

<p align="center">
  <img src="docs/img/fig_recovery_result.png" alt="Features selected by each method" width="100%">
</p>

Asking each method for `k = 7` and scoring the returned subset on the **true**
(distance-correlation) matrix:

| method | `lambda_min(D_dcor)` of the subset | one feature per source |
|---|---|---|
| `linear` | 0.475 | no - source 0 kept twice, one independent feature missed |
| `rank` | 0.733 | no - sources 1 and 3 kept twice |
| `dcor` | **0.953** | **yes** |
| `hsic` | 0.952 | **yes** |
| `nmi` | 0.947 | **yes** |

Panel (a) counts the features kept per source (1 is correct; a red box flags a
violation) and panel (b) shows which features were kept. The nonlinear measures
return one feature per source - they recover the independent directions - while
`linear` and `rank` spend two of their seven slots on a single source. The
subsets are still "optimal" for their own criterion, which is precisely why the
wrong dependence measure cannot be detected by looking at its own `lambda_min`:
scored against the truth, the linear subset loses half of its conditioning
(0.475 vs 0.953).

`tests/test_independence_recovery.py` asserts the experiment: one feature per
group for `dcor`, `hsic`, `nmi` and for the D-optimal criterion, stable over
seeds, and - conversely - that `rank` does recover as soon as the links are
monotone. The two figures are regenerated with

```bash
python3 scripts/make_recovery_figures.py     # requires matplotlib
```


### The D-optimal criterion: minimize the total correlation directly

E-optimality *implies* a small total correlation but does not optimize it (it
controls the worst direction). When the target is literally "the sub-joint
closest to the product of the marginals", the Gaussian multi-information
`TC(S) = -0.5 log det(D_S)` is the right objective:

```python
d = sifters.dopt.select(D, k=50, exact=True)   # max log det(D_S), branch and bound
d.subset, d.det, d.proved_optimal, d.total_correlation
```

`sifters.dopt` is a pure-Python reference solver: a forward greedy with a rank-1
inverse update, and an exact branch and bound that is valid because `det(D_S)`
is non-increasing as the set grows (the Schur complement is `<= 1`), so any
partial set upper-bounds all its completions. The certified Rust engine remains
the E-optimal one.

### Pairwise is not joint

All the measures above are **pairwise**. `D = I` means "all pairs are
independent", which is equivalent to the joint being the product of the
marginals for a pairwise Markov random field or a Gaussian copula, but not for a
general distribution with genuine higher-order (three-way and beyond)
interactions. Detecting those requires an estimate of the joint entropy
`H(X_S)`, which breaks the interlacing/monotonicity structure that the
certification relies on. `lambda_min(D_S)` and `log det(D_S)` remain tractable,
certificate-friendly surrogates.

## 7. Structure

**[`docs/algorithm.md`](docs/algorithm.md)** describes the algorithm in detail (formulas, bounds,
certification, complexities, parameters).

```
src/
  matrix.rs   N x M data (column-major), centering/scaling, degenerate columns
  gen.rs      synthetic generators (iid, equi, AR(1), blocks, factors) + RNG
              - compiled only for unit tests and examples/prof.rs
  packed.rs   packed lower triangle: parallel construction, matvec, swaps
  op.rs       SymOp, packed/implicit representations, submatrices, GEMV Z x w / Z^T x w
  inverse.rs  maintained inverse (initial Cholesky + rank-1 update), operator -A^-1
  lanczos.rs  reorthogonalized Lanczos + constraints, bisection/Sturm
  jacobi.rs   dense cyclic Jacobi (eigenvectors of the tridiagonal)
  dense.rs    dense symmetric eigendecomposition (whole-spectrum fast path)
  exact.rs    exact branch and bound over subsets of a fixed size
  greedy.rs   certified cascade, secular greedy, local swaps, bounds, KKT
  lib.rs      public surface of the crate
crates/
  sifters-python/         native PyO3 extension -> module sifters._sifters
python/
  sifters/__init__.py     Python API: select / path / curve, input conversion
  sifters/dependence.py   dependence matrices: rank/copula, dcor, HSIC, NMI
  sifters/dopt.py         D-optimal solver (max log det, greedy + exact B&B)
  sifters/__init__.pyi    typing stubs (+ _sifters.pyi, py.typed)
scripts/
  make_figures.py       generates the quality / performance README figures
  bench_numpy.py        timing vs naive numpy greedy (docs/img/fig_temps_vs_numpy.png)
  compare_baseline.py   quality/time comparison vs naive baselines (+ shared make_data)
  independence_recovery.py  ground-truth demo (recover the independent features)
  make_recovery_figures.py  generates docs/img/fig_recovery_*.png (needs matplotlib)
tests/
  test_sifters.py         binding tests (numpy as oracle)
  test_dependence.py    tests of the nonlinear dependence extension
  test_independence_recovery.py  ground-truth recovery tests
bench/        Python benchmark harnesses (see bench/README.md)
docs/
  algorithm.md          the algorithm in detail (formulas, bounds, invariants)
  python_api.md         full Python API reference
  img/                  README figures + raw measurements
examples/
  prof.rs               Rust profiling driver (keeps Python out of the profile)
pyproject.toml          maturin build backend (pip install . / maturin develop)
pixi.toml               dev environment + tasks
```

## 8. Documentation

- [`docs/algorithm.md`](docs/algorithm.md) - the algorithm in detail (formulas,
  bounds, certification, invariants, parameters);
- [`docs/python_api.md`](docs/python_api.md) - full Python API reference;
- [`bench/README.md`](bench/README.md) - benchmark harnesses and scenarios;
- [`bench/final_report.md`](bench/final_report.md) - historical exact-path
  measurements (CLI era);
- [`CHANGELOG.md`](CHANGELOG.md) - release history and known caveats.

## 9. Known limitations

* **Singular case (`M > N`)**: no inverse -> direct evaluations at 60-200 iterations.
  A suitable preconditioner remains to be found.
* **Myopic greedy**: the nested family is not globally optimal;
  `swap_passes=n` corrects locally by 1-for-1 swaps, and
  `forward_seeds=n` (mode `forward`) tries several starting variables and keeps the
  best family - the residual gap to the global optimum then goes from ~11 % to ~0.02 %
  on an AR(1) structure at `K = 3` (section 4).
* **Certification**: it uses valid upper bounds, but whose margin
  depends on the accuracy of the `p` eigenpairs; with `eval="direct"` and
  `iters_warm=60` the evaluations are not converged (use `eval="inverse"`,
  or increase `iters_warm`).
* **Python binding**: native PyO3 extension (`crates/sifters-python`), GIL released during
  the computation; installation requires a Rust build chain (`maturin`, see section 1).
* **Forward prefilter**: explicit heuristic, disabled by default; enabling it
  speeds up large `M` but can shave `lambda_min` by 4 to 12 % (section 4).
* **Exact mode**: it targets a single `K` and breaks the "whole family in one
  pass" promise, so the nested `curve`/`steps` still come from the greedy walk
  that seeded it. The worst case is exponential and the cost is driven by `K`:
  past the budget the search returns a certified gap instead of a proof
  (section 4).
* **Dependence measures** (section 6): `dcor` and `hsic` cost `O(M^2 N^2)` and
  materialize `M` matrices of size `N^2`; they are meant for moderate `N` /
  `M`. All measures are **pairwise**, so higher-order interactions are not
  captured. The D-optimal solver is a Python reference (not the certified Rust
  engine) and its branch and bound is exponential in the worst case.

## 10. License

Dual licensed under either of

* Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
* MIT license ([`LICENSE-MIT`](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
