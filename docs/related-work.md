# Prior art, positioning, and what is actually claimed

This package is easy to misread. It optimizes a criterion that is classical, with
algorithms that are classical, in a way that is not. This document says plainly
what belongs to the literature and what belongs to `sifters`, so that a reviewer
does not have to guess.

## 1. The criterion is not new, and it has a name

Maximizing `lambda_min(R_S)` over `k`-subsets is **E-optimality** from optimal
design theory: the E-optimal design maximizes the smallest eigenvalue of the
information matrix.

* F. Pukelsheim, *Optimal Design of Experiments*, Wiley 1993 / SIAM 2006
  (the standard reference for the E-, D-, A- and G-optimality criteria).
* The exact (cardinality-constrained) theory is its own chapter:
  C. L. Atwood, "Optimal design: Exact theory", in *Handbook of Statistics*,
  vol. 13, 1996.

Written on a Gram matrix, the same objective is the **sparse eigenvalue problem**,
which is known to be NP-hard and connected to the densest-`k`-subgraph problem:

* L. El Ghaoui, A. d'Aspremont et al., *Subset selection bounds* --
  <https://www.di.ens.fr/~aspremon/PDF/SubsetSelectionBounds.pdf>
  (the convex/SDP relaxation of the same objective, with approximation bounds).

And the *maximization of a determinant* -- the D-optimal cousin -- is not a
machine-learning invention either: its greedy is **pivoted Cholesky**, i.e. QR
with column pivoting, and it coincides with the MAP inference of a determinantal
point process:

* A. Kulesza, B. Taskar, *Determinantal Point Processes for Machine Learning*,
  Foundations and Trends in ML, 2012.
* *Lazy and Fast Greedy MAP Inference for Determinantal Point Processes*,
  NeurIPS 2022 -- <https://proceedings.neurips.cc/paper_files/paper/2022/file/127179162bfe4c422325ee7d05ad9cd8-Paper-Conference.pdf>
* *Submodular meets Spectral: Greedy Algorithms for Subset Selection*,
  ICML 2011 -- <http://www.icml-2011.org/papers/542_icmlpaper.pdf>

The D-optimal greedy has a `(1 - 1/e)` guarantee because `log det` is monotone
submodular. **E-optimality has no such guarantee**: `lambda_min` is neither
monotone nor submodular in the subset, which is exactly why the greedy gap
reported in section 4 of the README is 10-30 % in the backward direction and not
a constant factor. This asymmetry is a fact about the problem, not a defect of
this implementation.

`scripts/cssp_baselines.py` implements that equivalence and
`tests/test_cssp_baselines.py` asserts it: `pivoted_cholesky(R, k)` and
`qr_pivot(Z, k)` return the same pivot order, and each pivot maximizes
`det(R_{S u j}) / det(R_S)` by brute force.

## 2. The comparison target is not the reconstruction criterion

The Column Subset Selection Problem asks for columns of `A` whose span is close to
`A` in Frobenius or spectral norm: `min ||A - A_S A_S^+ A||`. That is a different
objective from "the `k` columns whose Gram matrix is best conditioned", although
the two families touch.

* C. Boutsidis, P. Drineas, M. W. Mahoney, *Near-optimal column-based matrix
  reconstruction*, SIAM J. Comput. 43(2), 2014 -- <https://arxiv.org/abs/1103.0995>
* P. Drineas, M. W. Mahoney, S. Muthukrishnan, *Relative-error CUR matrix
  decompositions*, 2007 -- <https://arxiv.org/abs/0708.3696>

Two consequences worth being explicit about:

1. **Randomized CSSP sampling degenerates here.** The standard sampler draws
   column `j` with probability `||A^(j)||^2 / ||A||_F^2`. `sifters` normalizes
   every column to unit norm before looking at the Gram, so those probabilities
   are all equal and the algorithm becomes *uniform sampling*. That is why
   `uniform_sample` in `scripts/cssp_baselines.py` is not a strawman: it is the
   CSSP sampler in this geometry.
2. **The relevant classical competitor is strong RRQR, not leverage scores.**
   Gu & Eisenstat's strong rank-revealing QR returns `k` columns with

   ```
   sigma_i(Z_S) >= sigma_i(Z) / sqrt(1 + f^2 k (M - k)),   1 <= i <= k
   ```

   * M. Gu, S. Eisenstat, *Efficient algorithms for computing a strong
     rank-revealing QR factorization*, SIAM J. Sci. Comput. 17(4), 1996.
   * Stated in this form in the randomized-vs-deterministic survey of subset
     selection, and in the RRQR literature.

   With `i = k` and `lambda_min(R_S) = sigma_min(Z_S)^2`, this is a *global,
   deterministic, multiplicative* guarantee on the very objective `sifters`
   optimizes. It is implemented as `rrqr_strong` / `rrqr_bound`, and the
   measurement of its slack is one of the more useful results here.

## 3. The guarantee is global but vacuous; the per-step certificate is tight

The honest reading of the numbers in `docs/measurements.md`:

* the Gu & Eisenstat floor is **valid** (asserted in
  `tests/test_cssp_baselines.py::test_rrqr_bound_never_exceeds_what_rrqr_achieves`)
  and **non-binding**: at `N=400, M=200` it sits 32x to 2359x below what
  column-pivoted QR already achieves, and guarantees an efficiency
  `eta <= 0.001` where the algorithm reaches `eta = 0.03..0.46`. The `k(M-k)`
  factor in the `sqrt` is what kills it at these sizes;
* the `sifters` forward certificate is **exact and tight**: the certified walk
  evaluates 8 candidates per step out of `M - k` available (4.6 % at `M = 200`),
  and reproduces the exhaustive `eigvalsh` greedy to `<= 1e-15` on every tested
  dataset.

So the two guarantees are not substitutes. RRQR's is a worst-case statement that
does not bite in practice; `sifters`' is a statement about the *cost of exactness*
at each step. Section 4 of the README reports both.

## 4. What is actually new here

Not the criterion, not the greedy, not the branch and bound, not the exchange
algorithm. Specifically:

1. **A certified exact greedy step for E-optimality.** Candidates are ranked by a
   valid upper bound and evaluated in decreasing order, stopping as soon as the
   best realized value beats the largest remaining bound; the step is then
   *provably* the exact greedy step. The bounds are the p-dimensional Temple
   bound with a Cauchy-Schwarz tail and conservative residual slack
   (`docs/algorithm.md` section 4.4), and in the forward direction the full
   spectrum (`p = k`) makes the secular bound **exact**, which collapses the
   number of evaluations to the batch size. We are not aware of a published
   statement of this certification scheme for E-optimal subset selection.
2. **The cost of the certificate is measured and published.** Baseline greedy
   E-optimal selection evaluates every candidate at every step. Section 4 and
   `docs/measurements.md` report the ratio, and separate the two directions
   honestly: forward prunes to ~5 % of candidates, backward only to 26-41 %.
   That asymmetry (the backward deletion bound is weak because it depends on the
   tail of the spectrum) is documented rather than hidden.
3. **One nested family for every K, with a per-step certificate.** The usual
   answers in this literature are for a single `k`: sampling gives one subset,
   an SDP relaxation gives one design, the exact theory gives one `k`. `sifters`
   returns `S_1 < S_2 < ... < S_kmax` (or the decreasing family) in one
   computation, with `lambda_min(K)` for every visited size, and
   `Selection.subset_at(K)` reconstructs any of them.
4. **Anytime exact mode.** `select(..., exact=True)` runs a branch and bound over
   the `k`-subsets whose bound is free (the value of a partial set upper-bounds
   every completion, by Cauchy interlacing) and either proves the optimum or
   returns a **certified relative gap**. Branch and bound for D-optimal design is
   classical (Welch 1982, *Branch-and-bound search for experimental designs based
   on D optimality*); transferring it to E-optimality with an anytime certified
   gap is an engineering contribution, not a theoretical one.
5. **The criterion as a black box on a PSD unit-diagonal matrix.** The engine only
   ever needs the Gram contract, so distance correlation, HSIC, normalized mutual
   information or a Gaussian copula can replace the correlation matrix and *every*
   guarantee survives (section 6 of the README). The CSSP/RRQR literature is
   linear-algebraic and has no analogue of "run the certified selector on a
   nonlinear dependence matrix"; this is the most application-relevant part of the
   package, and `scripts/independence_recovery.py` is its ground-truth test.
6. **A production implementation.** Safe Rust (`#![forbid(unsafe_code)]`), rayon,
   packed/implicit representations, a maintained inverse for shift-invert-like
   acceleration, a PyO3 extension with the GIL released, typed stubs, 43 Rust and
   120 Python tests, four CI gates. There is no Rust-native E-optimal column
   selector on crates.io or PyPI that we know of.

## 5. What is *not* claimed

* Not a new criterion, not a new hardness result, not a new relaxation.
* Not a global guarantee for the greedy family: `certified` qualifies a **step**,
  never the whole subset. `exact=True` is the only mode that claims global
  optimality, and only at the requested `k`.
* Not an approximation ratio. E-optimality has no `(1 - 1/e)`; the measured gap to
  the brute-force optimum is in section 4 of the README and is a
  structure-dependent 1-13 % for the forward walk, 10-30 % for the backward one.
* Not a win on every dataset. The measurements say the opposite, and
  `docs/measurements.md` reports where the classical methods win.

## 6. Where a theoretical contribution could still be made

Two gaps we did not close, in decreasing order of tractability:

1. **Complexity of the certificate.** How many exact evaluations does a certified
   greedy step require, as a function of `p`, `k` and the spectrum? The observed
   behaviour (a 1/p decay of the bound's slack in the backward direction; a
   collapse to the batch size in the forward direction once `p = k`) is
   empirical. A worst-case or average-case bound would turn the informal claim
   "e stays small" into a theorem.
2. **A global ratio for E-optimal greedy.** Either a direct bound, or one routed
   through the SDP value of the sparse eigenvalue problem (d'Aspremont et al.).
   The non-submodularity of `lambda_min` makes this genuinely open as far as we
   know, and the measured gaps suggest a structure-dependent ratio at best.
