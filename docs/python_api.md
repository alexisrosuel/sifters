# `sifters` Python API Reference

`sifters` is a native PyO3 extension over a safe, multi-threaded Rust engine. It
exposes one selection kernel — maximize the smallest eigenvalue
`lambda_min(R_S)` of the correlation matrix of a subset, the **E-optimal**
criterion — plus a dependence-aware wrapper and a pure-Python D-optimal
reference solver.

| Function | Purpose |
|---|---|
| `select` | **Primary**: keep `k` variables (E-optimal, certified) |
| `path` | Build a nested family of subsets over all sizes |
| `curve` | Shortcut: the `{K: lambda_min}` curve directly |
| `as_matrix` | Convert any array-like to a contiguous 2-D `float64` matrix |
| `dependence_matrix` | Pairwise dependence matrix for any kind of link |
| `select_from_dependence` | E- or D-optimal selection from a dependence matrix |
| `select_dependence` | End-to-end: `X` → dependence matrix → selection |
| `version` | Engine version string |

- **Module name:** `sifters` (native part: `sifters._sifters`)
- **Version:** `sifters.__version__`, also returned by `sifters.version()`
- **Python requirement:** `>= 3.9`
- **Array convention:** inputs are read through the buffer protocol and
  converted to contiguous `float64`; `numpy` is **optional** (it is only used to
  convert lists and DataFrames).
- **Indexing:** `X` is `(n_observations, n_variables)`. Every index in a result
  (`subset`, `order`, `steps[i].index`, ...) refers to a column of that input.
- **Threading:** the GIL is released during the computation; `threads=N` sizes a
  dedicated Rayon pool.
- **Validation:** bad arguments raise `ValueError`; engine failures (for
  example "no usable column") raise `sifters.SiftersError`, a subclass of
  `RuntimeError`. A Rust panic is never allowed to cross the boundary.

---

## Input contracts

- `X` must be a non-empty 2-D array-like `(n, m)` with `n >= 1`, `m >= 1`.
- Columns with zero variance (constant columns) are **dropped** during
  standardization and listed in `Selection.dropped`; the reported indices refer
  to the original columns. If every column is degenerate, `SiftersError` is raised.
- With `center=True` (the default) the columns are centered and scaled to unit
  norm, which gives the **correlation** matrix; `center=False` skips the
  centering and gives the **cosine** matrix.
- A finite `k` is required for `select`; `1 <= k <= m_usable`.
- The concentration ratio and spectral assumptions of the estimator do **not**
  apply here: the engine works on any symmetric matrix with a unit diagonal, as
  long as `select_from_dependence` is given a Gram factorization.

---

## Selection

### `select(X, k, *, direction="auto", ...)`

Select `k` variables out of `X` under the E-optimal criterion.

```python
import numpy as np, sifters

X = np.random.default_rng(0).normal(size=(500, 300))   # (observations, variables)

r = sifters.select(X, k=30)                       # direction="auto"
r = sifters.select(X, k=30, exact=True)           # proven optimum at k
r = sifters.select(X, k=30, prefilter=True)       # heuristic speed-up
r = sifters.select(X, k=30, forward_seeds=8)      # multi-start forward
```

Core parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `k` | — | Number of variables to keep |
| `direction` | `"auto"` | `"auto"` picks `"forward"` when `3k < m`, else `"backward"`; or force `"forward"` / `"backward"` |
| `center` | `True` | Center the columns (correlation) or not (cosine) |
| `verify` | `True` | Re-validate `lambda_min` with a strict cold Lanczos (tol `1e-13`); in `"backward"` mode also re-validates every step of the curve |
| `eval` | `"auto"` | Candidate evaluation: `"inverse"` maintains `R^-1` and converges in ~20 Lanczos iterations instead of 100–150 |
| `representation` | `"auto"` | `"packed"` materializes the correlation; `"implicit"` applies `Z_S^T (Z_S x)` and is optimal when `n << k` |

Solver and performance knobs:

| Parameter | Default | Meaning |
|---|---|---|
| `low_rank` | `4` | Number of eigenpairs used by the Temple bound |
| `tol` | `1e-10` | Lanczos tolerance |
| `iters_warm` | `60` | Iterations when a warm start is available |
| `iters_cold` | `400` | Iterations from a cold start |
| `max_exact` | `0` | Cap on candidates evaluated exactly per step (`0` = certified) |
| `batch` | `8` | Candidate batch size |
| `mem_budget_mb` | `4096` | Memory budget for the materialized representation |
| `block_rows` | `64` | Block height for the parallel construction |
| `threads` | `None` | Size of the dedicated Rayon pool (`None`/`0` = global pool) |

Traversal options:

| Parameter | Default | Meaning |
|---|---|---|
| `prefilter` | `False` | In `forward` mode, evaluate only the `forward_top` best secular candidates. **Disabled by default** so the walk is exactly E-optimal; enabling it is faster on large `m` but costs a few percent of `lambda_min` |
| `forward_top` | `0` | Prefilter width. `0` evaluates every candidate; when `prefilter=True` it defaults to `16` |
| `forward_first` | `None` | Imposed first variable (`forward` mode); disables multi-start |
| `forward_seeds` | `1` | Number of starting variables tried (`forward` mode). Start 1 is the historical heuristic, then the best one-step-lookahead variables; the best final family is kept. The starts are nested, so raising it can never degrade the result |
| `swap_passes` | `0` | Number of 1-for-1 local swap passes (`backward` mode) |
| `swap_top` | `8` | Candidates considered per swap |
| `swap_from` | `None` | First size at which swaps are allowed |

Exact mode:

| Parameter | Default | Meaning |
|---|---|---|
| `exact` | `False` | Prove optimality at `k` by branch and bound over all `k`-subsets, seeded by the greedy incumbent |
| `exact_time_s` | `30.0` | Wall-clock budget in seconds (`<= 0` = unlimited) |
| `exact_max_evals` | `0` | Cap on `lambda_min` evaluations (`0` = no cap) |

With `exact=True`, the returned `subset` / `lambda_min` are the optimum and
`proved_optimal` says whether the proof completed; otherwise `gap_certified`
bounds the remaining relative improvement. The nested family (`curve`,
`steps`) still comes from the greedy walk that seeded the search.

Progress:

| Parameter | Default | Meaning |
|---|---|---|
| `progress` | `None` | Called with a `Step` after each step. Returning `False` interrupts the walk cleanly and returns the partial result; `None` keeps going. `Ctrl-C` is intercepted and raises `KeyboardInterrupt` |

Returns a `Selection`.

---

### `path(X, *, kmin=1, kmax=None, direction="backward", ...)`

Build a nested family of subsets.

- `direction="backward"` (default) starts from all `m` variables and eliminates
  down to `kmin`. **Every step is certified.**
- `direction="forward"` starts from a singleton and grows up to `kmax`.

All other keywords are those of `select`. Returns a `Selection` whose
`curve` gives `lambda_min` for every visited `K`.

```python
r = sifters.path(X, direction="backward", kmin=1)   # the whole certified family
r = sifters.path(X, direction="forward", kmax=100, forward_seeds=4)
r.subset_at(42)          # the subset of size 42, no recomputation
```

---

### `curve(X, **kwargs) -> dict[int, float]`

Shortcut that returns the `{K: lambda_min}` curve directly. Accepts the same
keywords as `path`.

```python
c = sifters.curve(X, kmin=5, kmax=80)
```

---

### `as_matrix(X) -> numpy.ndarray`

Return a contiguous 2-D `float64` view (C order) of `X`. Useful to check or
explicitly prepare an input; the selection functions already perform this
conversion. `numpy` is used when installed.

---

## Result objects

### `Selection`

Complete result of a selection: nested family, curve and diagnostics.

| Attribute | Meaning |
|---|---|
| `direction` | Mode actually used |
| `k` | Requested size |
| `subset` | Selected column indices, increasing |
| `lambda_min` | Smallest eigenvalue of `R_S` (verified) |
| `lambda_residual` | Residual of the re-validation Lanczos |
| `curve` | `{K: lambda_min}` over the visited sizes |
| `seconds` | Total wall-clock time |
| `path_seconds` | Time spent in the greedy walk |
| `load_seconds` | Time spent loading/standardizing `X` |
| `order` | Order in which variables were removed/added |
| `initial_subset` | Starting subset of the walk |
| `n_observations`, `m_total`, `m_usable` | Sizes of the problem |
| `dropped` | Degenerate columns removed during standardization |
| `steps` | `list[Step]`, one per traversal step |
| `certified_steps` | Number of steps proven optimal |
| `total_exact_evals` | Exact candidate evaluations |
| `total_lanczos_iters` | Cumulative Lanczos iterations |
| `representation`, `eval` | Representations actually chosen |
| `proved_optimal` | True when `exact=True` proved the optimum |
| `gap_certified` | Certified relative gap otherwise |
| `exact_evals` | Evaluations spent in the exact search |

| Method | Meaning |
|---|---|
| `subset_at(k)` | Subset of any visited size (`None` if not visited) |
| `lambda_at(k)` | Verified `lambda_min` at size `k` |
| `lambda_raw_at(k)` | Raw (unverified) `lambda_min` at size `k` |
| `to_dict()` | JSON-serializable copy of the whole result |
| `len(r)` | Length of the family |

### `Step`

One step of the greedy walk (removal, addition or local swap). Indices are
given in the numbering of the input columns, before any removal of degenerate
columns.

`k_before`, `k_after`, `index`, `lambda_min`, `lambda_verified`, `upper_bound`,
`rayleigh_bound`, `exact_evals`, `candidates`, `certified`, `residual`,
`lanczos_iters`.

### `SiftersError`

Error raised by the numerical engine. Subclass of `RuntimeError`.

---

## Dependence-aware selection

Because the engine only needs a symmetric PSD matrix with a unit diagonal, any
**dependence matrix** `D` (with `D_ij = 0` meaning "columns `i` and `j` are
independent") can replace the correlation matrix. This makes the selection
robust to nonlinear and non-monotone links.

### `dependence_matrix(X, *, method="dcor", **kwargs)`

Pairwise dependence matrix `D` (unit diagonal). `method` is one of:

| `method` | Link it sees | Cost |
|---|---|---|
| `"linear"` | linear (Pearson correlation) | `O(m^2 n)` |
| `"rank"` | any **monotone** link (Gaussian copula / normal scores) | `O(m^2 n)` |
| `"dcor"` | any link, `0` iff independent (default) | `O(m^2 n^2)`, `O(m n^2)` memory |
| `"hsic"` | any link (normalized RBF kernel), PSD by construction | `O(m^2 n^2)` |
| `"nmi"` | any link, histogram estimate | `O(m^2 n)` |

Extra keywords go to the underlying measure (`bins` for `nmi`, `sigma` /
`kernel` for `hsic`). See `sifters.dependence` below.

### `select_from_dependence(D, k, *, criterion="E", **kwargs)`

- `criterion="E"` (default) maximizes `lambda_min(D_S)` with the certified Rust
  engine and returns a `Selection`.
- `criterion="D"` maximizes `log det(D_S)`, i.e. minimizes the Gaussian total
  correlation, with the Python reference solver and returns a
  `sifters.dopt.DOptResult`.

For the E criterion, `D` is projected to the nearest PSD unit-diagonal matrix,
factored as `D = B^T B`, and `B` is passed to `select` with `center=False`. All
other keywords (`direction`, `exact`, ...) are forwarded to `select`.

### `select_dependence(X, k, *, method="dcor", criterion="E", measure_kwargs=None, **kwargs)`

End-to-end shortcut: `dependence_matrix(X, method=...)` then
`select_from_dependence`.

```python
r = sifters.select_dependence(X, k=50, method="dcor", exact=True)
```

---

## `sifters.dependence`

Pure-Python implementations of the dependence measures and the PSD helpers.

| Function | Meaning |
|---|---|
| `dependence_matrix(X, *, method="dcor", **kwargs)` | Dispatch to one of `METHODS` |
| `normal_scores(X)` | Gaussian-copula / rank transform of every column |
| `distance_correlation(X)` | Pairwise distance correlation (Szekely et al.) |
| `hsic(X, *, sigma=None, kernel="rbf")` | Normalized HSIC matrix (`sigma=None` → median heuristic) |
| `normalized_mutual_information(X, *, bins="auto", normalize="sqrt")` | Pairwise NMI from 2-D histograms (`normalize="sqrt"` or `"min"`) |
| `project_psd(D, *, eps=1e-10)` | Nearest PSD matrix with a unit diagonal (eigenvalue clipping) |
| `factor(D, *, eps=1e-10, project=True)` | Factor `D = B^T B` with unit-norm columns of `B` |
| `select_from_dependence(D, k, *, project=True, **kwargs)` | E-optimal selection on `D` |
| `select_dependence(X, k, *, method="dcor", measure_kwargs=None, project=True, **kwargs)` | End-to-end shortcut |

`METHODS = ("linear", "rank", "dcor", "hsic", "nmi")`.

`dcor` and `nmi` are not guaranteed PSD: `project_psd` clips the negative
eigenvalues and rescales the diagonal back to `1` (a positive-diagonal
congruence, so the PSD property is preserved). Matrices that are already PSD
with a unit diagonal (`linear`, `rank`, `hsic`) are returned almost unchanged.

```python
D = sifters.dependence_matrix(X, method="dcor")
r = sifters.dependence.select_from_dependence(D, k=50, exact=True)
```

### Invariance to feature scaling

With the default settings all five measures are invariant to a per-feature
positive affine rescaling `x_j -> a_j x_j + b_j` (`a_j > 0`): the dependence
matrix (and therefore the selected subset) is unchanged. Two caveats:

- `hsic` with an explicit `sigma` **loses** the invariance (the bandwidth must
  follow the amplitude of the feature); the default median heuristic keeps it.
- The two **signed** measures (`linear`, `rank`) change when a feature is
  reflected (`x_j -> -x_j`). `dcor`, `hsic` and `nmi` measure the magnitude of
  the dependence and are invariant to a reflection.

`select(X, center=True)` (the default) is scale-invariant because it uses the
correlation matrix; `center=False` uses the cosine matrix, which is invariant
to a scaling of each column but **not** to a shift.

---

## `sifters.dopt`

The D-optimal criterion maximizes `log det(D_S)`, i.e. minimizes the Gaussian
total correlation `TC(S) = -0.5 log det(D_S)`. It is a pure-Python reference
solver: a forward greedy with a rank-1 inverse update, and an exact branch and
bound valid because `det(D_S)` is non-increasing as the set grows.

| Function | Meaning |
|---|---|
| `select(D, k, *, exact=False, ...)` | Greedy or branch-and-bound D-optimal selection |
| `greedy(D, k, *, project=True, family=False)` | Forward greedy for `max log det(D_S)` |
| `optimal(D, k, *, project=True, time_limit_s=30.0, node_limit=0, incumbent=None, tol=1e-12)` | Exact branch and bound |
| `select_from_data(X, k, *, method="dcor", exact=False, ...)` | `X` → dependence matrix → D-optimal selection |
| `logdet(D, subset)` | `log det(D_S)` (`0.0` for the empty set) |
| `total_correlation(D, subset)` | `-0.5 * log det(D_S)` |

### `DOptResult`

| Attribute | Meaning |
|---|---|
| `subset` | Selected variable indices |
| `k` | Requested size |
| `logdet` | `log det(D_S)` of the selected subset (`<= 0`) |
| `det` | `det(D_S)`, in `[0, 1]` |
| `curve` | `{size: log det}` of the nested greedy family (when available) |
| `proved_optimal` | True when the branch and bound exhausted the search space |
| `nodes` | Number of branch-and-bound nodes visited |
| `seconds` | Wall-clock time of the call |

```python
d = sifters.dopt.select(D, k=50, exact=True)
d.subset, d.det, d.proved_optimal, d.total_correlation
```

`time_limit_s` (`<= 0` for none) and `node_limit` (`0` for none) bound the
search; if a limit is hit, `proved_optimal` is False and the returned subset is
the best incumbent found so far.

---

## Notes and caveats

- **Pairwise is not joint.** `D = I` means "all pairs are independent", which is
  equivalent to the joint being the product of the marginals for a pairwise
  Markov random field or a Gaussian copula, but not for a general distribution
  with genuine higher-order interactions. `lambda_min(D_S)` and `log det(D_S)`
  remain tractable, certificate-friendly surrogates.
- **Myopic greedy.** The nested family is not globally optimal. `swap_passes`
  corrects locally, `forward_seeds` tries several starts, and `exact=True`
  proves the optimum at a single `k`.
- **Certification margin.** The bounds are valid, but their margin depends on
  the accuracy of the computed eigenpairs. With `eval="direct"` and the default
  `iters_warm=60`, the evaluations are not converged; prefer `eval="inverse"` or
  raise `iters_warm`.
- **Cost.** The exact search is exponential in the worst case and is driven by
  `k` far more than by `m`; past the budget it returns a certified gap instead
  of a proof. `dcor` and `hsic` cost `O(m^2 n^2)` and materialize `m` matrices
  of size `n^2`, so they are meant for moderate `n` and `m`.
- **Singular case (`m > n`).** No inverse is available, so the maintained-inverse
  path is disabled and evaluations fall back to direct iterations.
