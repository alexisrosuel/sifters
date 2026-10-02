# The `sifters` algorithm, in detail

This document describes precisely what the package does: the formulas, the procedures,
the invariants and the guarantees. For usage, refer to the [README](../README.md);
here we go into the code (`src/`).

---

## 1. The problem

Let `X` be a data matrix `N x M` (N observations, M variables). We center it
then normalize each column to norm 1, which gives `Z` and the **correlation
matrix**

```
R = Z^T Z in R^{MxM} ,   R_ii = 1.
```

Losing a variable (ignoring it) is a **permutation** of rows/columns of `R` that
we keep as the leading block. When we write `S \subseteq {0,...,M-1}` and `R_S`, we mean the
corresponding principal submatrix.

```
Objective: for each size K, find S of cardinality K maximizing  lambda_min(R_S).
```

This is the **E-optimal** criterion. Two useful reformulations:

* `lambda_min(R_S) = lambda_min(Z_S^T Z_S) = sigma_min(Z_S)^2` -- E-optimality = maximal `sigma_min`;
* `lambda_min(R_S) = 1 / lambda_max(R_S^-1)` -- this is the lever of evaluation via the inverse (section 5).

The problem is NP-hard. `sifters` does not look for the global optimum: it builds
a **nested family**

```
mode backward :  S_M \supset S_{M-1} \supset ... \supset S_{k_min}
mode forward  :  S_1 \subset S_2 \subset ... \subset S_{k_max}
```

by a greedy method, but a greedy method in which **each backward elimination step is proven
optimal** (valid upper bounds allow pruning without ever making a mistake).

> Result: an answer for *all* sizes K from a single computation, and a `certified`
> flag per step.

---

## 2. Overview

```
X (N x M)
  |  centering + normalization (section 3)           matrix.rs
  v
Z (N x M)  ------------------------------+
  |                                     |
  |  repr = packed  (section 8)         |  repr = implicit  (section 8)
  v                                     v
R = Z^T Z (packed)                      R x = Z_S^T (Z_S x)
  |                                     |
  +--------------+----------------------+
                 |
     +-----------+------------+
     |  mode backward (section 4)  |  mode forward (section 6)
     |  certified cascade          |  bounded greedy (+ optional prefilter)
     +-----------+------------+
                 |  at each step: warm Lanczos (section 9)
                 |                 maintained inverse (section 5, optional)
                 |                 local swaps (section 10, optional)
                 |  exact = True (section 7, optional)
                 v
        nested family + lambda_min(K) curve + certified flags
        (+ proven optimum at k, or a certified gap)
```

The flow of the Python extension (`sifters._sifters.run`, crates/sifters-python/src/lib.rs) is:
array conversion (numpy buffers) -> `standardize` -> `Dataset::new` (repr) ->
`greedy::backward` or `greedy::forward` -> optional exact branch and bound
(`exact::exact_at`, section 7) -> optional strict cold revalidation ->
construction of the `Selection` object (family, curve, steps, proof).

The **matrix contract** used everywhere below is "symmetric PSD with unit
diagonal", not "linear correlation". Any dependence matrix can therefore replace
`R = Z^T Z` by passing a factor `B` with `B^T B = D` and `center=False`
(section 16), which is how the nonlinear measures (distance correlation, HSIC,
NMI, Gaussian copula) and the D-optimal criterion plug into the same engine.

---

## 3. Preprocessing: centering, normalization, degenerate columns

`DataMatrix::standardize` (src/matrix.rs):

1. if `center` (default), subtract the mean of each column;
2. compute `||z_j||` and divide by the norm -> columns of norm 1;
3. **eliminate the zero-variance columns**: threshold
   `sqrt(N) * eps_machine * 100`. The removed indices are kept and remapped at the end
   (`keep_map`) so that the returned subsets speak the original numbering.

Everything is done column-major, in parallel (`rayon`). After this step,
`R = Z^T Z` is indeed the correlation matrix (or the cosine matrix if `center=False`).

---

## 4. Backward elimination: the certified cascade

This is the heart of the package (`greedy::backward`, `step_eliminate`, src/greedy.rs).

### 4.1 Step notation

At a given step, the current set `S` has `k` variables and we have its `p`
smallest eigenpairs (`p = low_rank`, default 4):

```
R u_l = lambda_l u_l ,   lambda_1 <= lambda_2 <= ... <= lambda_p ,   ||u_l|| = 1.
```

Removing the variable `i` gives `R_{-i}`. By **Cauchy interlacing**:

```
lambda_1(R) <= lambda_1(R_{-i}) <= lambda_2(R) <= lambda_2(R_{-i}) <= ...
```

so only lambda_1 can increase, and it stays in `[lambda_1, lambda_2]`.

### 4.2 The *exact* secular equation

The eigenvalues of `R_{-i}` are the roots of

```
S_i(mu) = sum_{l=1..k} u_l(i)^2 / (lambda_l - mu) = 0.
```

*Justification*: `det(R_{-i} - mu I) = det(R - mu I) * [(R-mu I)^-1]_ii` and
`[(R-mu I)^-1]_ii = sum_l u_l(i)^2/(lambda_l - mu)`. The poles `lambda_l` being excluded, the remaining
zeros are those of `S_i`.

On `(lambda_1, lambda_2)`, `S_i` is **strictly increasing**, from `-inf` (at `lambda_1^+`) to `+inf`
(at `lambda_2^-`): it therefore admits a **unique root** there, which is exactly
`lambda_min(R_{-i})`. If `u_1(i) = 0`, then `lambda_min(R_{-i}) = lambda_1` (the vector `u_1`
restricted is an eigenvector) and `secular_root` returns `None`.

`secular_root` (src/greedy.rs) solves this equation by **80 bisections** on
`(lambda_1, lambda_2)` (or `(lambda_1, lambda_1+1)` if `p = 1`). It is used only to
**build the seed** for Lanczos, not to decide: it is an upper bound, not an
exact value.

The **exact eigenvector** of `R_{-i}` is obtained by

```
x = (R - mu I)^-1 e_i = sum_l (u_l(i)/(lambda_l - mu)) u_l ,     x_i = 1.
```

This is `trial_vector`: restricted to `{0..k}\{i}`, it is the exact eigenvector of
`R_{-i}` evaluated at the true root `mu`. It is an almost perfect seed for Lanczos.

### 4.3 Bound 1: Rayleigh (O(1) per candidate)

For **any** unit vector `u`, we strip `u` of its `i` component and
renormalize. The Rayleigh quotient of this trial vector **upper-bounds**
`lambda_min(R_{-i})` (the min over the whole space is smaller than the value at a point):

```
rho_i = ( rq - 2*u_i*y_i + u_i^2 ) / (1 - u_i^2)  >=  lambda_min(R_{-i}),
     with y = R u  and  rq = u^T R u.
```

The formula uses `R_ii = 1`. It is `O(1)` but **loose**: it reaches
`lambda_1 + m_i` with `m_i = u_1(i)^2` (mass of the eigenvector), so it prunes
almost nothing. Hence the following bound.

### 4.4 Bound 2: p-dimensional spectral Temple

We take up `S_i(mu) = 0` at the root `mu*` and isolate the dominant term:

```
m_i / (mu* - lambda_1) = sum_{l=2..p} u_l(i)^2/(lambda_l - mu*) + tail(mu*).
```

We lower-bound each of the two right-hand terms.

**Low-dimensional term.** Since `mu* >= lambda_1`, `lambda_l - mu* <= lambda_l - lambda_1`, so

```
sum_{l=2..p} u_l(i)^2/(lambda_l - mu*)  >=  B_i := sum_{l=2..p} u_l(i)^2/(lambda_l - lambda_1).
```

**Tail, by Cauchy-Schwarz.** With

```
T_i = 1 - sum_{l<=p} u_l(i)^2,        Sig_i = (1 - lambda_1) - sum_{l=2..p} u_l(i)^2 (lambda_l - lambda_1),
```

we have the exact identity `Sig_i = sum_{l>p} u_l(i)^2 (lambda_l - lambda_1)` (it uses
`sum_l u_l(i)^2 = 1` and `sum_l u_l(i)^2 lambda_l = R_ii = 1`), and the Cauchy-Schwarz
"Titu" inequality applied to `a_l = u_l(i)^2`, `b_l = lambda_l - mu*` gives

```
tail(mu*) >= T_i^2 / (Sig_i - (mu*-lambda_1) T_i) >= T_i^2 / Sig_i.
```

**Conclusion.**

```
G_i = B_i + T_i^2/Sig_i,        ub_i = lambda_1 + m_i / G_i  >=  lambda_min(R_{-i}).
```

This is `LowSpectrum::upper_bound`. For `p = 1` it gives Rayleigh again. The test
`temple_bound_is_valid_and_tighter` checks on dense cases that `ub_i` dominates
the exact value always and that the mean gap is smaller than that of Rayleigh.

**Margin for the residuals.** The Lanczos residuals `||R u_l - lambda_l u_l||` are subtracted
from `T_i` and added to `Sig_i`:

```
slack = 2*p*max_l ||R u_l - lambda_l u_l||
t     = max(0, 1 - sum_{l<=p} u_l(i)^2 - slack)
Sig   = (1 - lambda_1) - sum_{l=2..p} u_l(i)^2 (lambda_l - lambda_1) - slack*(1 + (lambda_p - lambda_1))
```

Both operations **enlarge** the bound (they decrease `G_i`), so they remain
conservative. Special case `t <= 10^-12` (typically `p = k`, no tail): the
tail term is dropped and we take `lambda_1 + m_i/B_i`.

Finally `upper_bound` returns `min(ub_i(Temple), rho_i)`, then bounds the result
from below by `lambda_1` (numerical safeguard).

### 4.5 Selection and certification

A step = **this loop**:

```
1. y = R u_1 ;  rq = u_1^T y                                 (1 matvec)
2. for i = 0..k-1 :  ray_i = rayleigh(rq, u_1[i], y[i])
                     ub_i  = min(ub_i(Temple), ray_i)        O(p*k)
3. sort the candidates by DECREASING bound
4. evaluate exactly in BATCHES of `batch` (default 8), in that order:
       - if  best_value >= bound_of_next_candidate - margin:  STOP, certified
       - if  max_exact reached: STOP
5. return the exact argmax and its eigenpair
```

**Why this is correct.** All the `ub_i` are valid upper bounds of
`lambda_min(R_{-i})`. Sorting by decreasing bound then evaluating exactly: as soon as the
best **realized** value exceeds the largest **remaining** bound, no candidate
not evaluated can do better. The step is then optimal.

The margin is `tol * 50 * (1 + |rq|)`. Ties are broken by a tolerance
`tol * 10 * (1 + |value|)` so as not to let numerical noise decide between
indistinguishable candidates.

`StepRecord` keeps for traceability: the retained bound (`upper`), the Rayleigh
bound alone (`rayleigh`), the number of candidates evaluated exactly
(`exact_evals`), the total number of candidates, the `certified` flag, the Lanczos
residual and the cumulative iterations.

In backward mode **without a cap** (`max_exact=0`), the loop necessarily goes all the
way or stops on the criterion: `certified = true` at each step.

### 4.6 State update

After choosing `i*`:

1. `ds.swap(i*, k-1)` -- the variable leaves the leading block; the live set goes to
   `k-1` (no recompaction, see section 8); the permutation is also applied to `Z`, to
   packed `R` and to the inverse `W` (section 5);
2. seeds for the new spectrum: each former eigenvector is stripped of its
   `i*` component then permuted (`restrict_and_permute`); `seeds[0]` is replaced by
   the exact eigenvector of the winner (much better);
3. `low_spectrum(k-1, p, seeds, known = winner, cold = false)` recomputes the `p`
   smallest eigenpairs **warm**;
4. if `verify`: a **strict cold** Lanczos (`tol 1e-13`, 4000 iterations) recomputes
   `lambda_min` and feeds `lambda_verified` -- this is the value displayed in the curve.

Important point: even when the exact evaluation of the candidates goes through the inverse
(section 5), the **low spectrum used by the bounds is computed in direct mode**, so that
the bounds are identical to `eval="direct"`.

---

## 5. Evaluation via the maintained inverse (`eval="inverse"`)

### 5.1 Idea

`lambda_min(A) = 1/lambda_max(A^-1)`: the **largest** eigenvalue of `A^-1` equals
`1/lambda_min(A)`. Inverting **amplifies** the relative gaps at the bottom of the spectrum:

```
{0.0050, 0.0057, 0.0061, ...}   -- inverse -->   {200, 175, 164, ...}
 gaps ~12%
```

Lanczos therefore converges in **a few iterations** on `A^-1` where 60 to
200 were needed on `A` -- this is the "shift-invert" effect, **without a factorization per candidate**.

### 5.2 Rank-1 update (no factorization per candidate)

Let `W = R^-1` and `A = R_{-i}` be the block with `i` removed. Partitioning
`R = [[A, b],[b^T, R_ii]]`, the block inverse identity gives

```
A^-1 = W_{-i,-i} - (1/W_ii) w w^T ,      w = W_{.,i} with i removed.
```

Three operations, each `O(k^2)` or `O(k)`:

| operation | code | cost |
|---|---|---|
| initial Cholesky (`W = R^-1`) | `InverseSym::from_packed` | `O(M^3)` once |
| permutation `i <-> j` of the leading block | `swap_leading` | `O(k)` |
| removal of the last index | `downdate_last` | `O(k^2)` |
| matvec `W_{-d,-d} x - (w^T x) w/W_dd` | `mul` | `O(k^2)` |

The `mul` matvec splits its dot products into **two contiguous segments** on
either side of the removed index `d`, instead of testing `u < d` at each element:
the `O(k^2)` loop becomes vectorizable.

`from_packed` returns `None` if the block is not positive definite (`M > N` singular);
we then fall back on the direct evaluation. `downdate_error_growth` checks that
`W R - I` stays below `10^-6` after hundreds of updates.

### 5.3 The `-A^-1` operator

`NegInvSubOp` applies `x -> -A^-1 x` (via `mul(..., negate = true)`). Its **smallest**
eigenvalue `theta_min` equals `-1/lambda_min(A)`, hence the conversion

```
lambda_min(R_{-i}) = -1 / theta_min(-A^-1).
```

`to_lambda` implements this conversion. The test `inverse_eval_matches_direct_eval`
checks that both routes give the same values and vectors up to the tolerance.

---

## 6. Forward selection: the bordered greedy

`greedy::forward` starts from a singleton and adds one variable per step.

### 6.1 First variable

`best_first_feature`:

* if `R` is materialized (`packed`): the index whose maximum correlation in absolute
  value with the others is the smallest -- a Gershgorin-type heuristic,
  `O(M^2)` (used up to `M <= 20 000`);
* otherwise (implicit): the column farthest from the centroid, `O(NM)`.

### 6.2 Ranking by bordered secular loss

The bordered matrix `B = [[R_S, c],[c^T, 1]]` (`c = Z_S^T z_j`) has as eigenvalues
those of `R_S` **plus** the roots of

```
1 - mu = sum_l g_l^2 / (lambda_l - mu),      g_l = u_l^T c.
```

The **first-order loss** when adding `j` equals `sum_l g_l^2/(lambda_l - lambda_1)`: each
eigen-direction contributes more the closer it is to `lambda_1`. This is the
`loss` score of `forward`, and it is what distinguishes `sifters` from the correlation filter, which
only looks at `|g_1|`.

The computation of the `g_{l,.}` for **all** the columns is done in two vectorized
and parallel passes: `w_l = Z_S u_l` (`z_loading`), then
`g_{l,j} = z_j * w_l` for all `j` (`z_dot_all`, `O(NM)` per eigenpair, in
parallel over the columns).

### 6.3 Prefilter (optional) and exact evaluation

```
mode exact (forward_top = 0, default, CERTIFIED)   mode prefilter (forward_top = N)
  ub_j  = secular root of the COMPLETE spectrum    scored = sort by INCREASING loss
          (p = k: the bound is then EXACT)         take   = min(N, M-k)
  sort by DECREASING ub                            evaluate exactly the `take`
  evaluate in batches, stop as soon as             first ones, take the best
  best_realized >= next_ub - margin
```

No automatic switch based on size: by default (`forward_top = 0`) the forward
selection is the **certified** E-optimal greedy, in the sense that each step
stops on the optimality criterion (`certified` flag) instead of evaluating the
`M-k` candidates. The prefilter is enabled explicitly -- `prefilter=True` in the
Python binding (width 16, or `forward_top=N`); it then evaluates only the `N`
best of the secular score and loses the certification.

**Why `p = k` in exact mode.** With only the first `p` eigenpairs,
the secular root remains an upper bound (the tail terms are positive below
`lambda_1`) but loose: the mean gap to the exact value is ~10^-3, which lets through
dozens of candidates through the certification. With the **whole** spectrum of `R_S`, the
secular sum is exact, so the root is *exactly* `lambda_min` of the bordered
matrix (`forward_secular_bound_quality`: gap ~10^-15, max 4.5*10^-12 at production
settings). The certification then stops at the first batch (`batch`), and the cost
of the spectrum -- `k` warm Lanczos, only once per step -- is much lower than the
evaluations it saves (measured: 5.5x to 7.8x faster). `low_rank` remains
the setting of the prefilter and of the backward cascade.

Each retained candidate is evaluated exactly by warm Lanczos on `S union {j}`:

* if `R` is **materialized**, the operator is `BorderedHeadOp`, which applies
  directly `[[R_S, c],[c^T, 1]]`: `c` is the row `j` of the packed triangle, read in
  a **contiguous** way (`j >= k`), and the matvec costs `O(k^2)`;
* otherwise (`implicit`) we fall back on `GatheredZOp`, which builds
  `R_{S union {j}} x = C^T (Cx)` with `C = [Z_S, z_j]`, i.e. `O(Nk)` per matvec.

The second is the limiting factor of the exact forward greedy as soon as `N` is large
(`2N >~ k`), hence the first.

The seed is the **KKT trial vector**

```
mu  = root of  1 - mu - sum_l g_l^2/(lambda_l - mu) = 0   (root below lambda_1; upper bound if truncated)
y   = -sum_l (g_l/(lambda_l - mu)) u_l ,     seed = [ y ; 1 ],
```

fallback to `[u_1 ; 1]` if the root is not usable or if `||y|| > 10^6`.

**Important consequence for the guarantee**: with `forward_top = 0` (default), each
step is certified optimal (same upper bounds as the backward cascade, section 4.4) -- but
without evaluating the `M-k` candidates. With the prefilter enabled, it evaluates only the `N`
best of the secular score: this is a **heuristic**, faster (cost per step
~constant instead of growing with `M`) but whose `lambda_min` may be lower by
a few percent (section 4 of the README) -- benchmarked in `scripts/bench_numpy.py`.

### 6.4 State after addition

`ds.swap(k, j)` places the retained variable at position `k`; the previous
eigenvectors are extended by a `0` and serve as warm seeds; the winner feeds
`known`.

### 6.5 Multi-start (`forward_seeds`)

The forward greedy is myopic and, above all, **its quality depends strongly on the starting
variable**: all single variables have `lambda_min = 1`, so the objective does not decide
the first step -- it is [`best_first_feature`](../src/greedy.rs) (Gershgorin) that decides,
and a bad seed propagates through the whole family.

`forward_seeds = N` (default `1`) tries up to `N` starting variables:

1. `forward_seed_ranking` ranks the variables by **one-step lookahead**, that is,
   the best `lambda_min` reachable at `k = 2`:
   `score(j) = 1 - min_{i != j} |r_ij|` (closed form for a 2x2 correlation matrix).
   In the `implicit` representation (correlation not materialized), the exact computation
   would cost `O(N M^2)`: we fall back on the distance to the centroid, `O(N M)`;
2. the list of starts is `best_first_feature` (historical heuristic) followed by the
   best variables of this ranking, without duplicates. The set of starts is therefore
   **nested**: `forward_seeds = 1` reproduces exactly the historical greedy and
   increasing `N` can never degrade the final `lambda_min`;
3. the [`forward`](../src/greedy.rs) greedy is replayed from each retained seed;
4. the family kept is the one whose final `lambda_min` is the largest (at equal size),
   with the winning seed exposed by `Selection.initial_subset`.

Warning: since the family is chosen on the **final** `lambda_min`, the intermediate sizes `K`
may vary from one `N` to another, but never degrade at the last `K`.

The cost is multiplied by the number of starts. Since the permutations are involutions,
`forward_traced` records the sequence of `ds.swap` and replays it backwards: the
input state of the `Dataset` is restored exactly, without rebuilding the correlation
(`O(N M^2)`). An imposed `forward_first` disables multi-start.

---

## 7. Exact mode: branch and bound at a fixed size

Everything above builds a nested family greedily, which is myopic: `certified`
qualifies a *step*, never the whole subset. `src/exact.rs` closes that gap for a
target size `k`, and answers one of two things: the proven optimum, or a
certified bound on how far the best subset found still is from it.

Entry point: `greedy`-seeded `exact::exact_at(z, k, cfg, budget, incumbent)`,
reached from Python through `select(..., exact=True)`.

### 7.1 The bound is free

`lambda_min` is **monotonically non-increasing** when the set grows. If
`S subset T`, then `R_S` is a principal submatrix of `R_T`, and Cauchy
interlacing gives

```
lambda_min(R_S) >= lambda_min(R_T).
```

So the value of a *partial* set upper-bounds the value of **every** completion of
it. A depth-first exploration that adds one variable at a time can therefore
prune the whole subtree of `S` as soon as `f(S) <= best`, where `best` is the
best complete `k`-subset found so far. The subtlety is that this bound costs
nothing extra: the search has to evaluate `f(S + {j})` anyway to order its
children, and that is exactly the quantity the bound needs.

### 7.2 The search

* an explicit stack (never recursion), each node holding the selected positions,
  the first candidate index still allowed, its bound and the Lanczos seed of its
  parent;
* children are `S + {j}` for `j` in `first..m`, evaluated in parallel with
  `rayon`, then sorted by **decreasing** bound, so the most promising subtree is
  popped first;
* a child is kept only if its bound beats `best + margin`; the others cannot lead
  anywhere better and are dropped without exploring them;
* columns are visited in increasing index, so every `k`-subset is generated
  exactly once.

The stack is bounded by `O(m k)` nodes: along the current path, at most `m`
siblings are deferred per level, and the depth is at most `k`.

### 7.3 The certified gap

A `BTreeMap` keyed by the (order-preserving) bit pattern of the bounds holds a
multiset of the bounds of the **open** nodes. Its largest key is a valid upper
bound `UB` of the optimum, because any better subset would have to be a
completion of some open node. Hence:

```
LB = best (a real subset)        UB = max over open nodes
gap = (UB - LB) / LB
```

The search stops and declares `proved_optimal` as soon as `UB <= best + margin`;
if the wall-clock or evaluation budget runs out first, it returns the best subset
found together with that gap. Interrupting the search is therefore never a total
loss: the answer degrades into a *certified approximation*.

### 7.4 Numerical contract

A Ritz value of a Krylov subspace bounds `lambda_min` **from above**, never from
below (the Rayleigh quotient is minimized over a subspace of the whole space).
Pruning with such bounds is thus conservative: the search can never discard a
subtree containing a strictly better subset, it can only keep subtrees it could
have pruned. The comparison uses the usual margin
`tol * 50 * (1 + |lambda|)`, and the subset finally returned is re-evaluated with
a strict cold Lanczos (`tol = 1e-13`, 4000 iterations) before being reported.

### 7.5 Cost, and why the incumbent matters

The worst case is exponential, but the bound is only as good as `best`: a
near-optimal incumbent collapses the tree. That is why the greedy family is the
intended provider, and why `forward_seeds` (section 6.5) is a *performance*
parameter as much as a quality one. Measured with `forward_seeds=8`
(`bench/exact.py`, `N = 400`), proving optimality takes:

| structure | K | M=20 | M=30 | M=40 | M=60 | M=80 |
|---|---|---|---|---|---|---|
| iid | 5 | 0.03 s | 0.03 s | 0.04 s | 0.11 s | 0.20 s |
| iid | 8 | 0.21 s | 0.79 s | 1.3 s | 4.7 s | 13.4 s |
| AR(1) rho=0.9 | 5 | 0.04 s | 0.09 s | 0.25 s | 1.2 s | 1.1 s |
| AR(1) rho=0.9 | 8 | 0.28 s | 1.6 s | 8.0 s | 43.5 s | *gap 0.77 %* |

`K` drives the cost far more than `M`, and a fast-decaying spectrum (AR(1))
prunes better than a flat one (iid). Past the budget the answer is still useful:
at `M = 80, K = 8` on AR(1) the search times out but returns a subset 19.7 %
better than the greedy incumbent, with a certified gap of 0.77 %.

---

## 8. Representations and costs

Two representations, chosen by `Repr::Auto` (`Dataset::new`):

| | `packed` | `implicit` |
|---|---|---|
| storage | lower triangle of `R`, `M(M+1)/2` f64 | `Z` alone (`N*M`) |
| matvec `R_S x` | `O(k^2)` (`matvec_head`) | `O(Nk)`: `Z_S^T (Z_S x)` |
| kept by `Auto` if | fits in `mem_budget_mb` | otherwise |
| theoretical advantage | `k` small | `2N ~<= k` |
| construction | `O(NM^2)`, blocked GEMM | -- |

In both cases **the active set is always the leading block** `0..k`. Removing
a variable is just a swap `i <-> k-1`:

* `Z`: swap of two contiguous columns, `O(N)`;
* `PackedSym::swap_leading`: scans the rows up to `M` to permute the two
  indices, i.e. `O(M)`; in return **no recompaction** `O(M^2)` takes place, the
  live part remaining the leading block;
* `InverseSym::swap_leading`: `O(k)` (bounded to the leading block).

The construction of `R` is a parallel blocked GEMM: row blocks (`block_rows`,
default 64) and column blocks of 64, with one row block staying resident in cache
during the sweep.

Small shared numerical utility (src/lib.rs): dot product and norm
unrolled x4 to favor vectorization, `axpy` with a short-circuit on `alpha = 0`.

---

## 9. The spectral solver: reorthogonalized Lanczos

`lanczos::smallest_eigenpair_constrained` (src/lanczos.rs) is the single
entry point; it finds the **smallest** eigenvalue of a `SymOp` operator.

### 9.1 The loop

Classical Lanczos with:

* **full reorthogonalization** against the whole basis (stability), whose
  **second pass is conditional**: it is executed only if the first made
  the norm of `w` drop below `1/sqrt(2)` of its input value -- a sign of catastrophic
  cancellation, hence a loss of orthogonality. The warm iterations, where `w`
  loses little, thus save a full pass over the basis;
* **explicit projection** against the constraint vectors `constraints` at each
  step (deflation, section 9.2);
* **warm start**: the seed is the eigenvector of the previous step, or the
  secular KKT vector;
* **convergence test on the Ritz residual**: at each iteration `j`, the
  smallest eigenvalue `theta_j` of the tridiagonal `T_j` is estimated by
  `tridiag_smallest` (bisection + Sturm sequence, 40 steps), then the residual
  `||A u - theta_j u|| = beta_j * |y_j|` (`y_j` = last component of the eigenvector of `T_j`,
  obtained by inverse iteration, `O(j)`) is compared to `tol * (1 + |theta_j|)`. A mere
  stabilization test `|theta_j - theta_{j-1}|` can trigger on a plateau while the
  residual is still large: the Ritz value -- an **upper bound** of `lambda_min` -- is then
  overestimated, which corrupts the certification of the greedy step;
* **invariant subspace detection**: if `beta <= 10^-9 * (1 + |alpha|)`, `w` is nothing but
  rounding noise; we stop before creating spurious eigenvalues.

**End.** The smallest eigenvalue of `T_m` is recomputed by bisection (60 steps),
its eigenvector by inverse iteration (`tridiag_smallest_eigenvector`, `O(m)` -- instead
of the Jacobi diagonalization `O(m^3)` which cost more than all the rest
of the warm loop); the **exact residual** `||A u - theta u||` is computed on the
final vector (a quality indicator of the *vector*, not of the value).

### 9.2 Why the deflation is "explicit"

To find the 2nd, 3rd... eigenpair, `low_spectrum_inv` calls Lanczos again while
forbidding the space already covered. Two possible implementations:

* split on the projected operator `(I-P)A`: the trial vector keeps a
  spurious component that gets amplified by `alpha/beta` when `beta` is small -> unstable;
* **retained here**: orthogonalize each new basis vector against the
  constraints. This is stable by construction.

### 9.3 Auxiliary dense solvers (src/jacobi.rs)

* `eigen_sym`: cyclic Jacobi, `O(n^3)`, up to 100 sweeps, stops when
  `sum_{p<q} a_pq^2 <= 10^-30 * sum a_ii^2`. Very robust, reserved for tests and small
  sizes;
* `tridiag_smallest`: bisection on `[-r-1, r+1]` where `r` is the Gershgorin radius
  of the tridiagonal; the interval test only needs the predicate "at least one
  eigenvalue below `mu`", hence `sturm_has_negative` (Sturm sequence with an **early
  exit** at the first negative pivot -- the Sturm counter is increasing);
* `tridiag_smallest_eigenvector` / `tridiag_inverse_iterate`: eigenvector of the
  smallest eigenvalue by inverse iteration (LU of `T - theta*I` factorized once,
  `O(n)` per iteration);
* `tridiag_rayleigh`: Rayleigh quotient `y^T T y`, `O(n)`.

The Lanczos iteration table **never** uses `O(j^3)` re-diagonalization
in the warm loop.

---

## 10. Local 1-for-1 swaps (`swap_passes`)

Local improvement at constant size `K` (`refine_swaps`), enabled if
`swap_passes > 0` and only for `k <= swap_from`:

```
for each pass:
  rho_i = Rayleigh bound of the removal of i in S       (outgoing candidates)
  g_j = u_1^T c_j = w * z_j   for j not in S,  w = Z_S u_1  (incoming candidates)
  form the `swap-top x swap-top` pairs (outgoing i, incoming j),
  score them by increasing rho_i - |g_j|, keep the `batch` best,
  evaluate each exactly (GatheredZOp with del = i, extra = j),
  accept the best if the value strictly improves, otherwise STOP.
```

This is a discrete gradient ascent, **monotone by construction** (only an actual
improvement is accepted) but not certified.

---

## 11. Complexity

Notation: `k` = current size, `e` = candidates evaluated exactly per step,
`t` = Lanczos iterations per candidate.

**Backward** (M -> k_min), per step:

| item | cost |
|---|---|
| Rayleigh bound | 1 matvec + `O(k)` |
| Temple bounds | `O(p*k)` |
| exact evaluations | `e * t *` matvec |
| warm low spectrum | `p` constrained Lanczos |
| permutation `i* <-> k-1` | `O(M)` (packed) + `O(N)` (`Z`) |
| (inverse) `W` update | `O(k^2)` |

with matvec `O(k^2)` (packed) or `O(Nk)` (implicit). The key point is that **`e` stays
small**: sorting by decreasing bound means that certification often arrives after
one or two candidates, instead of the `k` of an exhaustive search.

**Forward** (1 -> k_max), per step: `O(p*NM)` for all the scores, `O(M log M)` of
sorting, `O(M*p*J)` for the secular bounds (`J` = bisection step of
`addition_secular`), then `e * t *` matvec for the evaluations. Since the forward
certification, **`e` stays small** as in the backward case: `take = M-k` is no longer
the default behavior, sorting by decreasing secular bound makes it possible to
stop as soon as the best realized value exceeds the largest remaining bound.
Better, in exact mode the bound is **exact** (`p = k`), so `e` drops to the floor
`batch` (8) and the dominant cost becomes the full spectrum of `R_S`: `O(k)` Lanczos
warm per step, i.e. `O(k^3)` flops. With the prefilter (`forward_top = N > 0`),
`e = N` but without guarantee.

**Memory footprint.** `packed`: `M(M+1)/2` f64 + `W` (`M^2`) if the inverse is active;
`implicit`: `N*M` only.

---

## 12. Guarantees, and what limits them

**Guaranteed.**

* The Rayleigh and Temple bounds are **valid upper bounds** of
  `lambda_min(R_{-i})`; the residuals are incorporated conservatively. Tests:
  `rayleigh_upper_is_valid_for_any_unit_vector`,
  `temple_bound_is_valid_and_tighter`.
* The truncated secular root of the bordered matrix is a valid upper bound of the
  new `lambda_min`, which certifies the exact forward selection. Test:
  `forward_secular_bound_quality` (validity + gap measured against the exact value).
* In backward mode without a cap on candidates, each step stops on the certification
  criterion: the removal chosen is that of the exact greedy. Tests:
  `backward_matches_brute_force_greedy` (comparison with brute force via `eigvalsh`).
* `forward` with `forward_top = 0` achieves the same exact greedy, but certified and
  frugal. Test: `forward_matches_brute_force_greedy`.
* `implicit` and `packed` give the same results; inverse and direct too. Tests:
  `backward_implicit_matches_packed`, `inverse_eval_matches_direct_eval`.
* With `exact=True`, the branch and bound of section 7 returns the **global
  optimum** at the requested size when it terminates. Test:
  `exact_matches_brute_force` (exhaustive enumeration over every `k`-subset).

**Not guaranteed / limitations.**

* The greedy is **myopic**: the nested family is not globally optimal
  (the local swaps of section 10 partially correct this, and section 7 proves the
  optimum at a single `k`). `exact=True` is the only mode that claims global
  optimality, and only for the requested size.
* `eval="direct"` with `iters_warm=60`: the evaluations may not be
  converged, in which case the certification must be taken with the tolerance margin;
  `eval="inverse"` or a larger `iters_warm` fixes the problem. The solver
  now stops on the Ritz **residual**, which makes this limitation visible
  (`converged = false`) rather than silent.
* `M > N`: `R` is singular, no inverse (`from_packed` -> `None`), we fall back on
  direct evaluations.
* `forward` with the prefilter enabled (`prefilter=True`, `forward_top=N > 0`):
  heuristic, `lambda_min` not guaranteed (see section 6.3).
* When two candidates have `lambda_min` values within the certification margin
  (`tol*50*(1+|lambda|)`) of each other, the choice is arbitrary: over a long
  family, this indeterminacy can make the sequence of subsets diverge. The
  steps remain optimal *up to the margin* (verified step by step against brute force
  on i.i.d. and correlated data).
* **The elimination bound remains loose.** `deletion_bound_quality` measures the mean
  bound-exact gap as a fraction of the range of the candidate `lambda_min` values:
  **40.6%** at `p = 4`, 24.1% at `p = 8`, 12.3% at `p = 16` -- i.e. a decay in `1/p`. This is
  what forces evaluating exactly ~15 to 25% of the candidates at each backward step.
  Making it exact would require `p = k` (full spectrum, cost of `k` Lanczos per step);
  the profitable route would be a **full spectrum maintained incrementally** from one step
  to the next (Cauchy interlacing + secular equation), which would bring the cost
  per step down to `O(k^3)` instead of `O(e*t*k^2)`. Not implemented to date.

---

## 13. Parameters that change the behavior

Names of the arguments of `sifters.select` / `sifters.path` (Python binding, the only interface).

| parameter | effect | default |
|---|---|---|
| `direction="backward"\|"forward"` | direction of the traversal | `"auto"` (`select`) / `"backward"` (`path`) |
| `kmin` / `kmax` | bounds of the traversal | `1` / `M` |
| `tol` | Lanczos tolerance, certification margin | `1e-10` |
| `iters_cold` | cold iterations | `400` |
| `iters_warm` | warm iterations | `60` |
| `low_rank` | number `p` of eigenpairs for Temple | `4` |
| `max_exact` | cap on candidates evaluated per step (`0` = certified) | `0` |
| `eval="auto"\|"direct"\|"inverse"` | evaluation route | `"auto"` |
| `forward_top` | width of the forward prefilter (`0` = disabled, certified exact greedy) | `0` |
| `prefilter` | enables the forward prefilter (width 16) | `False` |
| `forward_first` | first variable imposed | heuristic |
| `forward_seeds` | number of starts tried in forward selection (`1` = historical greedy, `>= M` = all) | `1` |
| `exact` | proves the optimum at `k` by branch and bound (section 7) | `False` |
| `exact_time_s` | wall-clock budget of the exact search (`<= 0` = unlimited) | `30.0` |
| `exact_max_evals` | cap on `lambda_min` evaluations of the exact search (`0` = none) | `0` |
| `swap_passes`, `swap_top`, `swap_from` | local swaps | `0`, `8`, `kmin` |
| `representation="auto"\|"packed"\|"implicit"` | representation | `"auto"` |
| `mem_budget_mb`, `block_rows`, `threads` | performance | `4096`, `64`, all |
| `verify` | strict cold revalidation | `True` |

`eval="auto"` chooses the inverse if `R` is materialized and `M <= 3000`
(`inverse_max_m`).

---

## 14. Outputs and traceability

* **curve** `(K, lambda_min)`: `lambda_verified` when cold revalidation is active,
  otherwise the warm Ritz value;
* **per step** (`Selection.steps`): entering/leaving variable, retained upper bound,
  Rayleigh bound, number of candidates evaluated, `certified` flag, residual,
  iterations;
* **`Selection.to_dict()`**: `direction`, `k`, `subset`, `lambda_min`, `curve`, `order`,
  `initial_subset`, `seconds`, `path_seconds`, `load_seconds`, `total_exact_evals`,
  `total_lanczos_iters`, `certified_steps`, `representation`, `eval`, `proved_optimal`,
  `gap_certified`, `exact_evals`, `steps`... The benchmarks (`bench/run.py`) write this
  dictionary as JSON.
* **exact mode** (`exact=True`): `proved_optimal` is true when the optimum at `k` was
  proven, `gap_certified` is the remaining relative gap (`0.0` when proven, `inf`
  when no proof was attempted) and `exact_evals` counts the `lambda_min`
  evaluations spent by the branch and bound.

Strict cold revalidation (`verify=True`) places the chosen variables at the head and
restarts a very strict **cold** Lanczos (`tol 1e-13`, 4000 iterations). The value
displayed is therefore a Ritz value certified up to the tolerance, independently of
any warm start.

Point of attention: `Dataset::new` resets `active` to `0..m` whereas `ds.z` has been
**permuted** by the traversal. The reconstruction must therefore reuse the current
permutation -- this is what `strict_lambda` (crates/sifters-python/src/lib.rs) does, which
recovers each variable via `ds.active` and not via the identity order of the clone. Otherwise
the position -> original index correspondence would be broken and the revalidation
would apply to a different subset.

`order` and `initial_subset` are enough to reconstruct any `S_K` of the family
(`Selection::subset_at` of the Python binding).

---

## 15. Where to read what

| file | content |
|---|---|
| `src/greedy.rs` | cascade, bounds, certification, forward greedy, swaps |
| `src/exact.rs` | exact branch and bound at a fixed size, certified gap |
| `src/lanczos.rs` | reorthogonalized Lanczos + deflation + Sturm stopping |
| `src/jacobi.rs` | dense Jacobi, tridiagonal bisection, Sturm sequence |
| `src/inverse.rs` | maintained inverse, `-A^-1`, rank-1 update |
| `src/op.rs` | `SymOp`, `Dataset`, `SubOp`, `GatheredZOp`, `z_loading`/`z_dot_all` |
| `src/packed.rs` | packed lower triangle, blocked GEMM, matvec, permutations |
| `src/matrix.rs` | standardization, degenerate columns |
| `src/gen.rs` | synthetic test generators, RNG xorshift64* (unit tests and `examples/prof.rs`) |
| `crates/sifters-python/src/lib.rs` | PyO3 extension: `run`, `Selection`, `Step`, GIL, threads |
| `python/sifters/__init__.py` | Python API (`select`/`path`/`curve`), input conversion |
| `python/sifters/dependence.py` | dependence matrices (rank/copula, dcor, HSIC, NMI), PSD projection, factorization |
| `python/sifters/dopt.py` | D-optimal solver (greedy + exact branch and bound on `log det`) |
| `scripts/independence_recovery.py` | ground-truth generator: independent sources, nonlinear links, recovery report |
| `scripts/make_recovery_figures.py` | figures `docs/img/fig_recovery_setup.png` and `docs/img/fig_recovery_result.png` |
| `tests/test_sifters.py` | binding tests (numpy as oracle) |
| `tests/test_dependence.py` | tests of the nonlinear dependence extension |
| `tests/test_independence_recovery.py` | ground-truth recovery tests (one feature per independent source) |
| `scripts/bench_numpy.py` | timing comparison vs naive numpy greedy |
| `scripts/compare_baseline.py` | quality vs correlation heuristics (+ `make_data`) |
| `bench/run.py`, `bench/compare.py` | Python benchmarks (scenarios, A/B) |
| `bench/exact.py` | exact mode: proof time, evaluations, certified gap |

---

## 16. Generalization: any dependence measure, and the D-optimal criterion

The E-optimal engine makes no use of "correlation" beyond the **contract on the
matrix**: a symmetric PSD matrix with a unit diagonal, i.e. the Gram matrix of
normalized columns. Sections 4 to 7 therefore extend verbatim to any
**dependence matrix** `D` (``D_ij = 0`` meaning "columns `i` and `j` are
independent") provided it is projected to the nearest PSD unit-diagonal matrix.

### 16.1 From the dependence matrix to the engine

Given `D` (unit diagonal), compute an eigendecomposition `D = V diag(w) V^T`,
project `w <- max(w, eps)` (nearest PSD, `eps > 0` makes it definite), rescale
the diagonal to 1 (a positive-diagonal congruence, so PSD is preserved), then
factor

```text
D = B^T B ,   B = diag(sqrt(w)) V^T ,   ||B_(:,j)|| = sqrt(D_jj) = 1 .
```

Passing `B` to `select(..., center=False)` makes the engine see exactly `D` as
its correlation matrix: `standardize` only renormalizes the columns (already of
norm 1) and the Gram is unchanged. Everything downstream - the secular equation,
the Temple bound, the certified cascade, the exact branch and bound, the
maintained inverse - is unchanged. This is what
`sifters.select_from_dependence(D, k)` does.

The measures provided (`sifters.dependence`):

| method | statistic | `D_ij = 0` iff | PSD by construction |
|---|---|---|---|
| `linear` | Pearson correlation | linear independence | yes |
| `rank` | normal scores (Gaussian copula) | monotone-link independence | yes |
| `dcor` | distance correlation | independence | no (projected) |
| `hsic` | normalized RBF HSIC | independence | yes (Gram of centered kernels) |
| `nmi` | normalized mutual information | independence | no (projected) |

For `trace(D_S) = K` and a unit diagonal, `lambda_min(D_S) >= 1 - eps` forces
every eigenvalue of `D_S` into `[1 - eps, 1 + (K-1) eps]` (if `lambda_max = L`
then `K = trace >= L + (K-1)(1-eps)`), hence

```text
det(D_S) in [(1-eps)^K, 1]   and   TC(S) = -0.5 log det(D_S) <= -K/2 log(1-eps) .
```

So the certified E-optimal criterion is a **hard surrogate** for the Gaussian
total correlation, at no algorithmic cost.

### 16.2 D-optimal: maximizing `log det` directly

The Gaussian total correlation (multi-information) `TC(S) = sum_i H(X_i) - H(X_S)`
equals `-0.5 log det(D_S)`, so "joint closest to the product of the marginals"
is literally `max log det(D_S)`. Two facts make this tractable.

**Monotone determinant.** For a PSD unit-diagonal `D` and `j` outside `S`,

```text
det(D_{S u j}) = det(D_S) * (1 - c^T D_S^{-1} c) ,   c = D_{S,j} ,
```

and the Schur complement `s_j = 1 - c^T D_S^{-1} c` lies in `[0, 1]` because
`D_{S u j}` is PSD. Adding a variable can therefore only decrease the
determinant: `det(D_{S'}) <= det(D_S)` for every superset `S'` of `S`. This is
the exact analogue of Cauchy interlacing for `lambda_min`.

**Bordered inverse.** With `A = D_S` invertible and `u = A^{-1} c`,

```text
[[A, c], [c^T, 1]]^{-1} = [[ A^{-1} + u u^T / s ,  -u / s ],
                           [ -u^T / s          ,   1 / s  ]] ,   s = 1 - c^T u .
```

This gives (i) the forward greedy score of every candidate in `O(k^2)` once
`u = A^{-1} c` is formed, and (ii) the exact determinant ratio
`det(D_{S u j}) / det(D_S) = s_j`. The greedy (`sifters.dopt.greedy`) picks the
largest `s_j` at each step and maintains `A^{-1}` by the rank-1 update above:
`O(M k^2)` per step, `O(M k^3)` overall, no factorization.

**Exact branch and bound** (`sifters.dopt.optimal`). The monotonicity gives the
pruning rule: a partial set `S` with `log det(D_S) <= best` cannot beat the
incumbent, because every completion of `S` has a smaller determinant. The search
is a DFS that includes or excludes variables, ordered by decreasing `s_j` (the
greedy order) so that a strong incumbent appears on the first leaf; it is seeded
by the greedy subset. `proved_optimal` reports whether the space was exhausted
within the node/time budget.

**E-optimal versus D-optimal.**

```text
E-optimal : max lambda_min(D_S)  ->  controls the worst direction
D-optimal : max log det(D_S)     ->  minimizes the total correlation exactly
```

E-optimality implies a small total correlation (section 16.1) but not the
converse: a subset can have a large determinant and one nearly-singular
direction. D-optimality does not bound `lambda_min`. The Rust engine implements
the certified E-optimal path; `sifters.dopt` is a pure-Python reference for the
D-optimal path (greedy + exact branch and bound).

### 16.3 Pairwise versus joint

Both criteria use a **pairwise** dependence matrix. `D = I` means that all pairs
are independent, and this implies joint independence exactly when the
distribution is a pairwise Markov random field or a Gaussian copula. For a
generic distribution with a genuine three-way interaction, a pairwise matrix
cannot see it (all pairwise marginals can factorize while the joint does not).
Capturing these requires an estimate of `H(X_S)`, which is neither a Gram
matrix nor monotone under set growth: the interlacing and Schur arguments above
no longer hold and the certification would be lost. The pairwise criteria remain
the tractable, certificate-compatible surrogates.
