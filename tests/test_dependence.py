"""Tests for the nonlinear dependence extension (`sifters.dependence`, `sifters.dopt`)."""
import itertools
import json
from statistics import NormalDist

import numpy as np
import pytest

import sifters
from sifters import dependence as dep
from sifters import dopt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _psd_unit(m, seed=0, shrink=0.4):
    """Random PSD matrix with a unit diagonal."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, m))
    D = A @ A.T
    d = np.sqrt(np.diag(D))
    D = D / np.outer(d, d)
    return (1.0 - shrink) * D + shrink * np.eye(m)


def _brute_lmin(D, k):
    return max(
        float(np.linalg.eigvalsh(D[np.ix_(c, c)])[0])
        for c in itertools.combinations(range(D.shape[0]), k)
    )


def _brute_logdet(D, k):
    best, arg = -np.inf, None
    for c in itertools.combinations(range(D.shape[0]), k):
        _sign, ld = np.linalg.slogdet(D[np.ix_(c, c)])
        if ld > best:
            best, arg = float(ld), c
    return best, arg


def _nonlinear_trap(seed=11, n=800):
    """x, y = x**2 (strong nonlinear link, ~0 correlation), u = f(w)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = x ** 2 + 0.05 * rng.normal(size=n)
    w = rng.normal(size=n)
    u = 0.15 * w + 0.10 * rng.normal(size=n)
    X = np.column_stack([x, y, u, w])
    return (X - X.mean(0)) / X.std(0)


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def test_normal_scores_match_stdlib_quantiles():
    p = np.linspace(0.001, 0.999, 41)
    expected = np.array([NormalDist().inv_cdf(float(v)) for v in p])
    assert np.allclose(dep._norm_ppf(p), expected, atol=1e-8)


def test_normal_scores_are_standardized_and_monotone():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)) ** 3
    Z = dep.normal_scores(X)
    assert np.allclose(Z.mean(0), 0.0, atol=1e-12)
    assert np.allclose(Z.std(0), 1.0, atol=1e-12)
    # ranks are preserved column-wise (monotone transform)
    for j in range(3):
        assert np.array_equal(np.argsort(X[:, j]), np.argsort(Z[:, j]))


# ---------------------------------------------------------------------------
# Dependence measures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", dep.METHODS)
def test_dependence_matrix_is_symmetric_unit_diagonal(method):
    X = np.random.default_rng(1).normal(size=(150, 6))
    D = sifters.dependence_matrix(X, method=method)
    assert D.shape == (6, 6)
    assert np.allclose(D, D.T, atol=1e-12)
    assert np.allclose(np.diag(D), 1.0, atol=1e-12)
    assert np.abs(D - np.eye(6)).max() <= 1.0 + 1e-12


@pytest.mark.parametrize("method", dep.METHODS)
def test_measures_are_near_zero_for_independent_columns(method):
    X = np.random.default_rng(2).normal(size=(600, 5))
    D = sifters.dependence_matrix(X, method=method)
    off = D[np.triu_indices(5, 1)]
    assert np.abs(off).max() < 0.25, (method, off)


@pytest.mark.parametrize(
    "method,link",
    [("rank", "exp"), ("dcor", "square"), ("hsic", "square"), ("nmi", "square")],
)
def test_measures_detect_a_nonlinear_link(method, link):
    rng = np.random.default_rng(3)
    x = rng.normal(size=500)
    y = np.exp(x) if link == "exp" else x ** 2
    X = np.column_stack([x, y, rng.normal(size=500)])
    D = sifters.dependence_matrix(X, method=method)
    assert abs(D[0, 1]) > 0.3, (method, D[0, 1])
    assert abs(D[0, 2]) < 0.2


def test_linear_measure_is_blind_to_the_nonlinear_link():
    rng = np.random.default_rng(3)
    x = rng.normal(size=800)
    X = np.column_stack([x, x ** 2])
    D = sifters.dependence_matrix(X, method="linear")
    assert abs(D[0, 1]) < 0.1
    assert abs(sifters.dependence_matrix(X, method="dcor")[0, 1]) > 0.3


# ---------------------------------------------------------------------------
# Invariance to the scaling of each feature
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", dep.METHODS)
def test_measures_are_invariant_to_per_feature_affine_scaling(method):
    """x_j -> a_j x_j + b_j with a_j > 0 must leave D (and the selection) intact."""
    rng = np.random.default_rng(31)
    X = rng.normal(size=(250, 6))
    a = np.exp(rng.uniform(-3.0, 3.0, 6))   # amplitudes 0.05 .. 20
    b = rng.uniform(-5.0, 5.0, 6)
    D0 = sifters.dependence_matrix(X, method=method)
    D1 = sifters.dependence_matrix(X * a + b, method=method)
    assert np.allclose(D0, D1, atol=1e-10), np.abs(D0 - D1).max()


@pytest.mark.parametrize("method", dep.METHODS)
def test_selection_is_invariant_to_per_feature_scaling(method):
    rng = np.random.default_rng(32)
    X = rng.normal(size=(250, 7))
    a = np.exp(rng.uniform(-3.0, 3.0, 7))
    b = rng.uniform(-4.0, 4.0, 7)
    r0 = sifters.select_dependence(X, 3, method=method, direction="backward", exact=True)
    r1 = sifters.select_dependence(X * a + b, 3, method=method, direction="backward", exact=True)
    assert r0.subset == r1.subset
    assert abs(r0.lambda_min - r1.lambda_min) < 1e-9


@pytest.mark.parametrize("method", ["dcor", "hsic", "nmi"])
def test_unsigned_measures_are_reflection_invariant(method):
    """Negating a subset of the features must not change D (unsigned measures)."""
    rng = np.random.default_rng(33)
    X = rng.normal(size=(250, 5))
    flip = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    D0 = sifters.dependence_matrix(X, method=method)
    D1 = sifters.dependence_matrix(X * flip, method=method)
    assert np.allclose(D0, D1, atol=1e-10)


def test_hsic_scale_invariance_relies_on_the_median_heuristic():
    """A fixed ``sigma`` breaks the invariance; the default (median) does not."""
    rng = np.random.default_rng(34)
    X = rng.normal(size=(200, 4))
    a = np.exp(rng.uniform(-2.0, 2.0, 4))
    D0 = sifters.dependence_matrix(X, method="hsic")
    D1 = sifters.dependence_matrix(X * a, method="hsic")
    assert np.allclose(D0, D1, atol=1e-10)

    F0 = sifters.dependence_matrix(X, method="hsic", sigma=1.0)
    F1 = sifters.dependence_matrix(X * a, method="hsic", sigma=1.0)
    assert np.abs(F0 - F1).max() > 1e-3


def test_hsic_matrix_is_psd_by_construction():
    X = np.random.default_rng(4).normal(size=(120, 5))
    D = dep.hsic(X)
    assert np.linalg.eigvalsh(D).min() > -1e-10


# ---------------------------------------------------------------------------
# PSD projection and factorization
# ---------------------------------------------------------------------------

def test_project_psd_returns_a_unit_diagonal_psd_matrix():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(6, 6))
    D = A @ A.T
    D = 0.7 * D / np.abs(D).max() + 0.3 * np.eye(6)
    D[0, 1] = D[1, 0] = 2.0  # deliberately not PSD
    P = dep.project_psd(D)
    assert np.allclose(P, P.T)
    assert np.allclose(np.diag(P), 1.0)
    assert np.linalg.eigvalsh(P).min() > 0.0


def test_factor_reconstructs_the_projected_matrix():
    D = _psd_unit(7, seed=6)
    B = dep.factor(D)
    assert B.shape == (7, 7)
    assert np.allclose(np.linalg.norm(B, axis=0), 1.0)
    assert np.allclose(B.T @ B, dep.project_psd(D), atol=1e-9)


# ---------------------------------------------------------------------------
# E-optimal selection on a dependence matrix
# ---------------------------------------------------------------------------

def test_select_from_dependence_matches_brute_force():
    D = _psd_unit(9, seed=7)
    for k in (2, 3, 4):
        r = sifters.select_from_dependence(D, k, direction="backward", exact=True)
        assert r.proved_optimal
        assert r.subset == sorted(r.subset)
        assert abs(r.lambda_min - _brute_lmin(D, k)) < 1e-9


def test_lambda_min_is_really_the_dependence_eigenvalue():
    D = _psd_unit(8, seed=8)
    r = sifters.select_from_dependence(D, 4, direction="backward")
    got = float(np.linalg.eigvalsh(D[np.ix_(r.subset, r.subset)])[0])
    assert abs(got - r.lambda_min) < 1e-9


def test_dependence_selection_beats_the_linear_one_on_a_nonlinear_trap():
    X = _nonlinear_trap()
    D_lin = sifters.dependence_matrix(X, method="linear")
    D_dcor = sifters.dependence_matrix(X, method="dcor")
    # the linear criterion thinks x and x**2 are independent and keeps both
    r_lin = sifters.select_from_dependence(D_lin, 2, direction="backward", exact=True)
    assert set(r_lin.subset) == {0, 1}
    # the distance-correlation criterion avoids the nonlinear pair
    r_dcor = sifters.select_from_dependence(D_dcor, 2, direction="backward", exact=True)
    real_lin = float(D_dcor[np.ix_(r_lin.subset, r_lin.subset)][0, 1])
    real_dcor = float(D_dcor[np.ix_(r_dcor.subset, r_dcor.subset)][0, 1])
    assert real_dcor < 0.5 * real_lin


def test_select_dependence_end_to_end():
    X = _nonlinear_trap()
    r = sifters.select_dependence(X, 2, method="dcor", direction="backward", exact=True)
    assert len(r.subset) == 2
    assert 0.0 < r.lambda_min <= 1.0


# ---------------------------------------------------------------------------
# D-optimal selection (max log det)
# ---------------------------------------------------------------------------

def test_dopt_optimal_matches_brute_force():
    for seed, m, k in ((10, 8, 3), (11, 9, 4)):
        D = _psd_unit(m, seed=seed)
        res = dopt.select(D, k, exact=True)
        best, _arg = _brute_logdet(D, k)
        assert res.proved_optimal
        assert abs(res.logdet - best) < 1e-9
        assert np.exp(res.logdet) <= 1.0 + 1e-12


def test_dopt_greedy_is_a_lower_bound_of_the_optimum():
    D = _psd_unit(10, seed=12)
    g = dopt.greedy(D, 4)
    o = dopt.optimal(D, 4)
    assert o.logdet + 1e-9 >= g.logdet
    assert o.det <= 1.0 + 1e-12


def test_dopt_identity_gives_determinant_one():
    D = np.eye(6)
    res = dopt.select(D, 3, exact=True)
    assert abs(res.logdet) < 1e-12
    assert abs(res.det - 1.0) < 1e-12
    assert res.proved_optimal


def test_dopt_curve_is_monotone_decreasing():
    D = _psd_unit(12, seed=13)
    res = dopt.greedy(D, 6, family=True)
    sizes = sorted(res.curve)
    assert sizes == list(range(1, 7))
    vals = [res.curve[s] for s in sizes]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-12


def test_dopt_total_correlation_relation():
    D = _psd_unit(7, seed=14)
    idx = [0, 2, 5]
    ld = dopt.logdet(D, idx)
    assert abs(dopt.total_correlation(D, idx) + 0.5 * ld) < 1e-12
    _sign, expected = np.linalg.slogdet(D[np.ix_(idx, idx)])
    assert abs(ld - expected) < 1e-12


def test_dopt_reports_a_non_proof_when_the_node_limit_is_hit():
    D = _psd_unit(10, seed=15)
    res = dopt.select(D, 5, exact=True, node_limit=1)
    assert not res.proved_optimal


def test_dopt_select_from_data_and_criterion_switch():
    X = _nonlinear_trap(seed=21)
    res_e = sifters.select_dependence(X, 2, method="dcor", criterion="E", exact=True)
    res_d = sifters.select_dependence(X, 2, method="dcor", criterion="D", exact=True)
    assert hasattr(res_e, "lambda_min")
    assert isinstance(res_d, dopt.DOptResult)
    assert res_d.criterion == "D"


def test_dopt_to_dict_is_json_serializable():
    D = _psd_unit(6, seed=16)
    payload = dopt.select(D, 3, exact=True).to_dict()
    text = json.dumps(payload)
    assert json.loads(text)["k"] == 3


def test_dopt_rejects_bad_input():
    with pytest.raises(ValueError):
        dopt.select(np.zeros((3, 4)), 2)
    with pytest.raises(ValueError):
        sifters.dependence_matrix(np.zeros((5, 3)), method="unknown")
