# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Nonlinear dependence selection** (section 6 of the README, section 16 of
  [`docs/algorithm.md`](docs/algorithm.md)). The engine only ever needs a
  symmetric PSD matrix with unit diagonal, so any dependence matrix `D`
  (`D_ij = 0` iff independent) can replace the linear correlation matrix, with
  no change to the certified core:
  - `sifters.dependence_matrix(X, method=...)` with `linear`, `rank` (Gaussian
    copula / normal scores, any monotone link), `dcor` (distance correlation),
    `hsic` (normalized RBF HSIC, PSD by construction) and `nmi` (normalized
    mutual information from 2D histograms);
  - `sifters.select_from_dependence(D, k, ...)` projects `D` to the nearest PSD
    unit-diagonal matrix, factors it as `D = B^T B` and calls the existing engine
    with `center=False`, so the certificate, the exact branch and bound and every
    `select` option keep working;
  - `sifters.select_dependence(X, k, method=..., criterion="E"|"D")` end-to-end.
- **D-optimal solver** (`sifters.dopt`) for the literal "closest to the product of
  the marginals" objective, the Gaussian total correlation
  `TC(S) = -0.5 log det(D_S)`: `greedy` (forward selection with a rank-1
  bordered-inverse update) and `optimal` (exact branch and bound, valid because
  `det(D_S)` is non-increasing as the set grows). `DOptResult` carries
  `subset`, `k`, `logdet`, `det`, `curve`, `proved_optimal`, `nodes`, `seconds`
  (`to_dict()`-style `total_correlation` included).
- `tests/test_dependence.py`: tests of the measures (independence, nonlinear
  detection, PSD-ness), of the factorization, and of both solvers against brute
  force.
- `scripts/independence_recovery.py`: runnable ground-truth example. Features are
  generated from independent latent sources through nonlinear/non-monotone links
  plus pure-noise features; asking the selector for the true number of
  independent directions must return one feature per source.
  `tests/test_independence_recovery.py` asserts it for `dcor`, `hsic`, `nmi`
  and the D-optimal criterion, and documents that `linear` / `rank` are fooled
  by the non-monotone links.
- `scripts/make_recovery_figures.py` and the two README figures
  `docs/img/fig_recovery_setup.png` (generative model, distance correlation vs
  linear correlation) and `docs/img/fig_recovery_result.png` (features picked per
  source, by method). README section 6 reports the experiment, scored on the
  true distance-correlation matrix: the nonlinear measures recover the 7
  independent directions, `linear` scores 0.475 against 0.953 for `dcor`.

### Changed — package layout and documentation tidy-up

Aligned the repository with the `fast_rmt_shrinkage` layout. No estimator,
measured number or public API changed.

- Long-form documentation now lives under `docs/`: `ALGORITHME.md` moved to
  [`docs/algorithm.md`](docs/algorithm.md), and the README figures moved to
  `docs/img/` next to their raw measurements.
- New [`docs/python_api.md`](docs/python_api.md): the complete Python API
  reference (`select` / `path` / `curve`, the `Selection` and `Step` objects,
  the dependence measures and the D-optimal solver).
- Python sources are split by role: the importable package stays in
  `python/sifters/`, the generators and figure scripts moved to `scripts/`, and
  the `pytest` suite to `tests/` (`testpaths = ["tests"]`, `pythonpath =
  ["scripts"]` in `pyproject.toml`).
- New `examples/README.md` documenting the Rust profiling driver `prof.rs`.
- New `pixi.toml` with the dev tasks (`build`, `test`, `test-py`, `lint`,
  `lint-py`, `type-py`, `fmt`, `bench`, `figures`, ...), and `ruff` / `mypy`
  configuration in `pyproject.toml`.
- CI now covers the whole workspace: `cargo clippy --workspace --all-targets`
  (the PyO3 binding crate was previously never linted), `cargo doc` with
  `RUSTDOCFLAGS=-D warnings`, `ruff check .` (explicit rule selection),
  `mypy scripts tests`, and a Python matrix building a wheel per interpreter
  (`3.9`, `3.12`, `3.14`). Added the `release.yml` (tag -> wheels + sdist ->
  PyPI, trusted publishing) and `testpypi.yml` (manual dry run) workflows.
- The tree is now `rustfmt`-clean and `clippy -D warnings`-clean: a `cargo fmt
  --all` pass over `src/` and `crates/`, and a fix to the implicit-QL loop bound
  in `src/dense.rs` (`while m <= n - 1` -> `while m < n`).
- README: CI/licence badges, an updated structure tree, a *Documentation*
  section, a *Development environment* (pixi) section, and every path reference
  updated to the new layout.

### Notes

- The dependence matrices are invariant to a per-feature positive affine
  rescaling `x_j -> a_j x_j + b_j` (`a_j > 0`), so the selection does not depend
  on the units of the features. `hsic` with an explicit `sigma` is the
  exception (the default median heuristic adapts the bandwidth); `linear` and
  `rank` are signed measures and change under a feature reflection, unlike
  `dcor`, `hsic` and `nmi`.

## [0.1.0] - 2026-10-01

First public release. The package, its documentation and its tests are entirely
in English.

### Added

- **Exact mode** — `sifters.select(X, k=..., exact=True)` maximizes `lambda_min`
  over all subsets of the requested size with a branch and bound, and returns
  either the proven optimum or a certified optimality gap:
  - `Selection.proved_optimal` (bool), `Selection.gap_certified` (relative gap,
    `0.0` when proven, `inf` when no exact search was run),
    `Selection.exact_evals` (search cost);
  - `exact_time_s` and `exact_max_evals` bound the search; the pruning bound is
    free because `lambda_min` is monotonically non-increasing as the set grows
    (Cauchy interlacing), so a partial set upper-bounds all its completions;
  - the greedy family supplies the incumbent, which is what makes the proof
    cheap.
- **Multi-start forward selection** — `forward_seeds=N` tries several starting
  variables and keeps the family with the largest final `lambda_min`. The first
  start is the historical heuristic, then come the best variables of the
  one-step lookahead ranking (`lambda_min` at `k = 2`, i.e. `1 - min_j |r_ij|`).
  The start set is nested, so increasing `N` can never degrade the result, and
  `forward_seeds=1` reproduces the historical greedy walk exactly.
- Python package (`python/sifters`) with type stubs, `sifters.select`,
  `sifters.path`, `sifters.curve`, `sifters.as_matrix`, progress callbacks and
  `Ctrl-C` handling.
- Rust library exposing the greedy walk, the certified backward cascade, the
  exact branch and bound, the Lanczos eigensolver, the bordered/secular
  machinery and the packed/implicit correlation representations.
- Benchmark harness under `bench/` and figure scripts under `python/`.
- Dual licence (MIT or Apache-2.0) and a CI workflow running the Rust tests, the
  Python tests and an ASCII/English check.

### Notes

- The greedy walk is myopic: `certified` qualifies each **step**, not the whole
  subset. Only `exact=True` proves global optimality, at the requested size.
- `python/` requires Python >= 3.9; the Rust library requires Rust >= 1.75.

[0.1.0]: https://github.com/alexisrosuel/sifters/releases/tag/v0.1.0
