"""Tests for the classical CSSP baselines and the claims they are used to support.

These tests lock two kinds of statement:

* the baselines really are what their docstrings say (pivoted Cholesky *is* the
  exact D-optimal greedy; QRCP uses the same pivot rule; the Fedorov result is a
  local optimum; the VIF bound is on the full matrix, not on a subset);
* the comparison in ``scripts/compare_cssp.py`` is not rigged -- the certified
  forward walk of `sifters` must reproduce the exhaustive eigen-oracle greedy,
  and the strong-RRQR floor must never exceed what RRQR actually achieves.
"""

from __future__ import annotations

import numpy as np
import pytest

import sifters
from compare_baseline import make_data
from cssp_baselines import (
    ceiling,
    correlation,
    efficiency,
    eig_greedy_backward,
    eig_greedy_forward,
    fedorov_exchange,
    lam_min,
    pivoted_cholesky,
    qr_pivot,
    rrqr_bound,
    rrqr_growth,
    rrqr_strong,
    standardize,
    uniform_sample,
    vif_topk,
)

KINDS = ("iid", "ar", "blocks", "factor")
TOL = 1e-9


def _dataset(kind: str, n: int = 150, m: int = 35, seed: int = 1) -> tuple:
    X = make_data(kind, n, m, 0.9, 0.05, 5, 4, seed)
    Z = standardize(X)
    return X, Z, Z.T @ Z


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_standardize_normalizes_columns() -> None:
    X = make_data("ar", 80, 20, 0.7, 0.0, 4, 3, 0)
    Z = standardize(X)
    assert np.allclose(np.linalg.norm(Z, axis=0), 1.0)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    R = correlation(X)
    assert R.shape == (20, 20)
    assert np.allclose(np.diag(R), 1.0)


def test_ceiling_bounds_every_subset() -> None:
    """``sigma_k(Z)^2`` is the largest ``lambda_min`` any ``k``-subset can reach."""
    rng = np.random.default_rng(0)
    for kind in KINDS:
        _X, _Z, R = _dataset(kind)
        m = R.shape[0]
        for k in (3, 8, 15):
            cap = ceiling(R, k)
            for _ in range(20):
                S = rng.choice(m, size=k, replace=False)
                assert lam_min(R, S) <= cap + TOL


def test_efficiency_of_every_baseline_is_at_most_one() -> None:
    rng = np.random.default_rng(3)
    for kind in KINDS:
        _X, Z, R = _dataset(kind)
        m = R.shape[0]
        for k in (4, 10, 18):
            cap = ceiling(R, k)
            subsets = {
                "qr_pivot": qr_pivot(Z, k),
                "pivoted_cholesky": pivoted_cholesky(R, k),
                "vif_topk": vif_topk(R, k, ridge=1e-10),
                "uniform": uniform_sample(m, k, rng),
                "rrqr": rrqr_strong(Z, k)[0],
                "fedorov": fedorov_exchange(R, k, rng, passes=3)[0],
                "oracle": eig_greedy_forward(R, k)[0],
            }
            for name, S in subsets.items():
                assert len(S) == k, (name, k, len(S))
                assert 0.0 < efficiency(lam_min(R, S), cap) <= 1.0 + TOL, (kind, k, name)


# --------------------------------------------------------------------------- #
# pivoted Cholesky == QRCP == exact D-optimal greedy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", KINDS)
def test_qrcp_and_pivoted_cholesky_share_the_pivot_rule(kind: str) -> None:
    _X, Z, R = _dataset(kind)
    for k in (3, 10, 20):
        assert sorted(qr_pivot(Z, k)) == sorted(pivoted_cholesky(R, k))


@pytest.mark.parametrize("kind", KINDS)
def test_pivoted_cholesky_is_the_exact_d_optimal_greedy(kind: str) -> None:
    """Every pivot must maximize ``det(R_{S u j}) / det(R_S)``, by brute force."""
    _X, _Z, R = _dataset(kind)
    m = R.shape[0]
    order = pivoted_cholesky(R, 12)
    S: list[int] = []
    for step, p in enumerate(order):
        rest = [j for j in range(m) if j not in S]
        gains = [np.linalg.slogdet(R[np.ix_(S + [j], S + [j])])[1] for j in rest]
        assert gains[rest.index(p)] >= max(gains) - 1e-9, (kind, step, p)
        if step == 0:
            # diag(R) is constant on a correlation matrix: the objective is
            # degenerate, so `argmax` is decided by rounding noise.
            d = np.diag(R)
            assert d.max() - d.min() < 1e-12
        S.append(p)


def test_pivoted_cholesky_first_step_is_exact_when_the_diagonal_differs() -> None:
    """With a non-degenerate diagonal the same rule is well posed from step 0."""
    rng = np.random.default_rng(11)
    B = rng.normal(size=(60, 14))
    R = B.T @ B  # distinct diagonal entries on purpose
    m = R.shape[0]
    order = pivoted_cholesky(R, 6)
    S: list[int] = []
    for step, p in enumerate(order):
        rest = [j for j in range(m) if j not in S]
        gains = [np.linalg.slogdet(R[np.ix_(S + [j], S + [j])])[1] for j in rest]
        assert gains[rest.index(p)] >= max(gains) - 1e-9, (step, p)
        S.append(p)


def test_sifters_dopt_greedy_agrees_with_its_own_subset() -> None:
    """``sifters.dopt.greedy`` scores the subset it returns (sanity of the seam)."""
    from sifters import dopt

    for kind in KINDS:
        _X, _Z, R = _dataset(kind)
        res = dopt.greedy(R, 8, project=False)
        assert np.isclose(res.logdet, dopt.logdet(R, res.subset), atol=1e-10)
        # it is a greedy on the maximum determinant, so it beats a random draw
        rng = np.random.default_rng(0)
        random_subset = uniform_sample(R.shape[0], 8, rng)
        assert res.logdet >= dopt.logdet(R, random_subset) - 1e-10


# --------------------------------------------------------------------------- #
# strong RRQR: the guarantee, and its slack
# --------------------------------------------------------------------------- #
def test_rrqr_bound_never_exceeds_what_rrqr_achieves() -> None:
    for kind in KINDS:
        _X, Z, R = _dataset(kind)
        for k in (4, 10, 18):
            for f in (1.0, 0.3, 0.05):
                S, _swaps, bound, _extra = rrqr_strong(Z, k, f=f)
                assert bound <= lam_min(R, S) + TOL, (kind, k, f, bound, lam_min(R, S))


def test_rrqr_bound_is_monotone_in_f_and_matches_its_formula() -> None:
    _X, Z, R = _dataset("ar")
    k = 8
    cap = ceiling(R, k)
    for f in (0.5, 1.0, 2.0):
        expected = cap / (1.0 + f * f * k * (R.shape[0] - k))
        assert np.isclose(rrqr_bound(R, k, f=f), expected, rtol=1e-12)
    assert rrqr_bound(R, k, f=0.5) > rrqr_bound(R, k, f=1.0) > rrqr_bound(R, k, f=2.0)


def test_rrqr_reports_the_effective_growth_that_makes_the_bound_valid() -> None:
    """``f_eff = max(f, growth)``, and the reported growth matches a recomputation."""
    _X, Z, R = _dataset("blocks")
    for k in (5, 12):
        S, swaps, bound, extra = rrqr_strong(Z, k, f=0.05)
        growth = rrqr_growth(Z, S)
        assert np.isclose(extra["growth"], growth, rtol=1e-6)
        assert extra["f_eff"] >= growth - TOL
        assert bound <= lam_min(R, S) + TOL
        assert swaps <= 4 * k


def test_rrqr_guarantee_is_loose_in_this_regime() -> None:
    """The documented, honest finding: the floor is orders of magnitude below the
    value RRQR actually reaches, so the guarantee is not the deciding criterion."""
    _X, Z, R = _dataset("factor")
    k = 15
    S, _swaps, bound, _extra = rrqr_strong(Z, k, f=1.0)
    assert lam_min(R, S) > 10.0 * bound


# --------------------------------------------------------------------------- #
# Fedorov exchange
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ("iid", "ar"))
def test_fedorov_result_is_a_local_optimum(kind: str) -> None:
    _X, _Z, R = _dataset(kind)
    m = R.shape[0]
    k = 8
    rng = np.random.default_rng(5)
    S, _stats = fedorov_exchange(R, k, rng, passes=20, restarts=2)
    assert len(S) == k
    value = lam_min(R, S)
    rest = [j for j in range(m) if j not in set(S)]
    for i in S:
        base = [c for c in S if c != i]
        for j in rest:
            assert lam_min(R, base + [j]) <= value + 1e-9, (kind, i, j)


def test_fedorov_never_worse_than_its_initial_subset() -> None:
    _X, _Z, R = _dataset("ar")
    k = 10
    init = pivoted_cholesky(R, k)
    rng = np.random.default_rng(1)
    S, _stats = fedorov_exchange(R, k, rng, init=init, passes=20, restarts=1)
    assert lam_min(R, S) >= lam_min(R, init) - 1e-9


# --------------------------------------------------------------------------- #
# VIF: the bound that is true, and the one that is not
# --------------------------------------------------------------------------- #
def test_vif_only_certifies_ill_conditioning() -> None:
    """``lambda_min(R) <= 1 / max_j VIF_j`` -- an *upper* bound.

    A large VIF proves the matrix is badly conditioned (that is what the
    diagnostic is for); a small VIF proves nothing, and the inequality is
    typically strict.  This is why VIF top-k is a screen and not a selector.
    """
    strict = 0
    for kind in KINDS:
        _X, _Z, R = _dataset(kind)
        vif = np.diag(np.linalg.inv(R))
        upper = 1.0 / vif.max()
        assert lam_min(R, range(R.shape[0])) <= upper + TOL, kind
        if lam_min(R, range(R.shape[0])) < upper - 1e-6:
            strict += 1
    assert strict == len(KINDS)  # the bound is strict, i.e. it certifies little


def test_vif_topk_returns_a_subset_of_the_right_size() -> None:
    _X, _Z, R = _dataset("iid")
    S = vif_topk(R, 7, ridge=1e-10)
    assert len(S) == 7
    assert len(set(S)) == 7
    assert S == sorted(S)


# --------------------------------------------------------------------------- #
# the comparison itself: the certified walk must be the exhaustive greedy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", KINDS)
def test_certified_forward_walk_reproduces_the_exhaustive_oracle(kind: str) -> None:
    """This is the claim the whole package rests on, checked against the oracle.

    ``sifters`` certifies each *step* of the forward walk; if the certificate were
    wrong, the certified walk would drift away from the brute-force greedy.  The
    exhaustive ``eigvalsh`` oracle here evaluates every candidate at every step.
    """
    X, _Z, R = _dataset(kind, n=250, m=30, seed=4)
    kmax = 10
    _order, info = eig_greedy_forward(R, kmax)
    sel = sifters.path(X, direction="forward", kmax=kmax)
    for k in range(2, kmax + 1):
        value = sel.lambda_at(k)
        subset = sel.subset_at(k)
        assert value is not None and subset is not None
        assert np.isclose(value, info["curve"][k], rtol=1e-9, atol=1e-12), (kind, k)
        assert sorted(subset) == sorted(eig_greedy_forward(R, k)[0]), (kind, k)


@pytest.mark.parametrize("kind", ("iid", "ar"))
def test_certified_backward_walk_reproduces_the_exhaustive_oracle(kind: str) -> None:
    X, _Z, R = _dataset(kind, n=200, m=22, seed=2)
    kmin = 14
    _order, info = eig_greedy_backward(R, kmin)
    sel = sifters.path(X, direction="backward", kmin=kmin)
    for k in range(kmin, R.shape[0] + 1):
        value = sel.lambda_at(k)
        assert value is not None
        assert np.isclose(value, info["curve"][k], rtol=1e-9, atol=1e-12), (kind, k)


def test_uniform_sample_is_a_proper_subset() -> None:
    rng = np.random.default_rng(0)
    S = uniform_sample(30, 8, rng)
    assert len(S) == 8 and len(set(S)) == 8
