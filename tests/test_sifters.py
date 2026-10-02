"""Tests for the native Python binding `sifters`.

Run with:

    python -m pytest tests/test_sifters.py       # or simply `pytest`
    python tests/test_sifters.py                 # equivalent, no pytest collection

numpy is used as the oracle for every spectral value.
"""
import itertools
import time
import threading

import numpy as np
import pytest

import sifters


def _data(n=200, m=60, rho=0.5, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n, 1))
    return np.sqrt(rho) * g + np.sqrt(1.0 - rho) * rng.normal(size=(n, m))


def _corr(X):
    """Correlation matrix of X (same conventions as the engine)."""
    Z = X - X.mean(0)
    nrm = np.linalg.norm(Z, axis=0)
    nrm[nrm == 0] = 1.0
    Z = Z / nrm
    return Z.T @ Z


def _lmin(R, idx):
    return float(np.linalg.eigvalsh(R[np.ix_(list(idx), list(idx))])[0])


# ---------------------------------------------------------------------------
# Correctness and quality
# ---------------------------------------------------------------------------

def test_select_shapes():
    X = _data()
    r = sifters.select(X, k=10)
    assert len(r.subset) == 10
    assert len(set(r.subset)) == 10
    assert 0.0 < r.lambda_min <= 1.0
    assert 0.3 < r.lambda_min < 1.0
    assert r.n_observations == 200
    assert r.m_total == 60
    assert len(r) == r.k == 10


def test_curve_is_monotone():
    X = _data()
    r = sifters.path(X, kmin=5)
    ks = sorted(r.curve)
    vals = [r.curve[k] for k in ks]
    # lambda_min decreases as K grows (interlacing theorem)
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-6


def test_matches_numpy_eigenvalue_both_directions():
    """`lambda_min` (cold-revalidated by default) must match numpy, and not just
    the warm Ritz value of the walk."""
    X = _data()
    R = _corr(X)
    for direction in ("forward", "backward"):
        for verify in (True, False):
            r = sifters.select(X, k=12, direction=direction, verify=verify)
            exact = np.linalg.eigvalsh(R[np.ix_(r.subset, r.subset)])[0]
            assert abs(exact - r.lambda_min) < 1e-9, (direction, verify, exact, r.lambda_min)


def test_subset_at_matches_numpy_for_every_size():
    X = _data(n=200, m=40, seed=7)
    R = _corr(X)
    r = sifters.path(X, kmin=3, direction="backward")
    for k in range(3, 41):
        subset = r.subset_at(k)
        assert subset is not None and len(subset) == k
        exact = np.linalg.eigvalsh(R[np.ix_(subset, subset)])[0]
        # curve steps are cold-revalidated: tight tolerance
        assert abs(exact - r.curve[k]) < 1e-8, (k, exact, r.curve[k])
    assert r.subset_at(2) is None  # outside the walk
    assert r.lambda_at(3) is not None


def test_forward_exact_matches_numpy_greedy():
    """The prefilter is off by default: the forward greedy must then coincide
    exactly with a naive E-optimal greedy (numpy)."""
    X = _data(n=300, m=600, seed=3)
    r = sifters.select(X, k=6, direction="forward", prefilter=False)
    assert r.initial_subset, "the binding must expose the starting variable"

    R = _corr(X)
    sel = [int(r.initial_subset[0])]
    while len(sel) < 6:
        rest = [j for j in range(R.shape[0]) if j not in sel]
        best = max(rest, key=lambda j: np.linalg.eigvalsh(R[np.ix_(sel + [j], sel + [j])])[0])
        sel.append(int(best))
    assert abs(np.linalg.eigvalsh(R[np.ix_(sel, sel)])[0] - r.lambda_min) < 1e-6


def test_forward_seeds_multi_start():
    """``forward_seeds=N`` tries N starts and keeps the best family.

    With ``N = M`` (every start) the result must coincide with the best walk
    obtained by imposing each ``forward_first``; with ``N = 1``, with the default
    walk.
    """
    X = _data(n=300, m=18, rho=0.8, seed=4)
    k = 6
    base = sifters.path(X, direction="forward", kmax=k)
    assert sifters.path(X, direction="forward", kmax=k, forward_seeds=1).subset == base.subset

    best = max(
        sifters.path(X, direction="forward", kmax=k, forward_first=j).lambda_at(k)
        for j in range(X.shape[1])
    )
    multi = sifters.path(X, direction="forward", kmax=k, forward_seeds=18)
    assert multi.lambda_at(k) == pytest.approx(best, abs=1e-12)
    # starts are nested: the final lambda_min cannot decrease with N
    values = [
        sifters.path(X, direction="forward", kmax=k, forward_seeds=n).lambda_at(k)
        for n in range(1, 19)
    ]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
    # the winning start is traceable, and the subset has the right size
    assert len(multi.initial_subset) == 1
    assert len(multi.subset_at(k)) == k


def test_forward_seeds_is_ignored_when_first_is_imposed():
    """An imposed start disables multi-start."""
    X = _data(n=200, m=14, rho=0.5, seed=5)
    a = sifters.path(X, direction="forward", kmax=5, forward_first=3, forward_seeds=9)
    b = sifters.path(X, direction="forward", kmax=5, forward_first=3)
    assert a.subset == b.subset


def test_prefilter_is_opt_in():
    """The prefilter is an explicit heuristic, disabled by default. The exact
    mode certifies every step (next test); it does not necessarily yield a larger
    final lambda_min, since the greedy walk is myopic: a heuristic deviation can
    end better."""
    X = _data(n=200, m=600, seed=1)
    exact = sifters.select(X, k=6, direction="forward", prefilter=False)
    filtre = sifters.select(X, k=6, direction="forward", prefilter=True)
    assert len(filtre.subset) == 6
    assert exact.certified_steps == len(exact.steps)


def test_forward_exact_is_stepwise_optimal():
    """Certified forward selection: at each step, the retained addition really
    maximizes lambda_min among *all* remaining candidates (numpy brute force)."""
    X = _data(n=300, m=40, rho=0.4, seed=3)
    r = sifters.select(X, k=12, direction="forward", prefilter=False)
    assert r.certified_steps == len(r.steps)

    Z = X - X.mean(0)
    Z = Z / np.linalg.norm(Z, axis=0)
    R = Z.T @ Z
    m = X.shape[1]
    added = [st.index for st in r.steps]
    first = (set(r.subset) - set(added)).pop()

    live = list(range(m))
    p0 = live.index(first)
    live[0], live[p0] = live[p0], live[0]
    for step, st in enumerate(r.steps):
        k = step + 1
        best = max(
            np.linalg.eigvalsh(R[np.ix_(live[:k] + [live[j]], live[:k] + [live[j]])])[0]
            for j in range(k, m)
        )
        assert abs(best - st.lambda_min) < 1e-8, (k, best, st.lambda_min)
        pos = live.index(st.index)
        live[k], live[pos] = live[pos], live[k]


def test_eval_and_representation_agree():
    """`inverse`/`direct` and `packed`/`implicit` must give the same result."""
    X = _data(n=300, m=80, seed=11)
    ref = sifters.select(X, k=15, direction="backward", eval="direct", representation="packed")
    for kwargs in (
        {"eval": "inverse"},
        {"representation": "implicit"},
        {"eval": "inverse", "representation": "implicit"},
    ):
        r = sifters.select(X, k=15, direction="backward", **kwargs)
        assert r.subset == ref.subset
        assert abs(r.lambda_min - ref.lambda_min) < 1e-9


def test_constant_columns_are_dropped_without_shifting_indices():
    X = _data(n=150, m=20, seed=5)
    X = np.column_stack([X[:, :3], np.full(150, 4.0), X[:, 3:]])
    r = sifters.select(X, k=5, direction="forward")
    assert r.dropped == [3]
    assert r.m_total == 21 and r.m_usable == 20
    assert 3 not in r.subset
    assert all(0 <= j < 21 for j in r.subset)
    R = _corr(X)
    assert abs(np.linalg.eigvalsh(R[np.ix_(r.subset, r.subset)])[0] - r.lambda_min) < 1e-9


# ---------------------------------------------------------------------------
# Exact mode (branch and bound)
# ---------------------------------------------------------------------------

def test_exact_proves_the_optimum_at_k():
    """`select(exact=True)` must return the global optimum at the requested k,
    checked against a brute-force enumeration of every subset."""
    X = _data(n=250, m=16, rho=0.8, seed=12)
    R = _corr(X)
    m = X.shape[1]
    for k in (2, 4, 6):
        r = sifters.select(X, k=k, direction="forward", exact=True)
        assert r.proved_optimal
        assert r.gap_certified == 0.0
        assert len(r.subset) == k
        want = max(_lmin(R, c) for c in itertools.combinations(range(m), k))
        assert abs(r.lambda_min - want) < 1e-8, (k, r.lambda_min, want)


def test_exact_never_returns_a_worse_subset_than_greedy():
    X = _data(n=250, m=22, rho=0.9, seed=13)
    for k in (4, 7):
        greedy = sifters.select(X, k=k, direction="forward")
        exact = sifters.select(X, k=k, direction="forward", exact=True)
        assert exact.lambda_min >= greedy.lambda_min - 1e-12
        assert len(exact.subset) == k


def test_exact_accepts_a_backward_incumbent():
    """The exact proof is direction-independent: it must find the same optimum
    whichever walk seeded it."""
    X = _data(n=200, m=14, rho=0.7, seed=14)
    fwd = sifters.select(X, k=5, direction="forward", exact=True)
    bwd = sifters.select(X, k=5, direction="backward", exact=True)
    assert abs(fwd.lambda_min - bwd.lambda_min) < 1e-8


def test_exact_reports_a_certified_gap_when_budget_is_exhausted():
    X = _data(n=250, m=26, rho=0.9, seed=15)
    k = 6
    full = sifters.select(X, k=k, direction="forward", exact=True)
    assert full.proved_optimal
    limited = sifters.select(
        X, k=k, direction="forward", exact=True, exact_max_evals=100
    )
    assert not limited.proved_optimal
    assert len(limited.subset) == k
    assert limited.exact_evals <= 100
    assert limited.gap_certified >= 0.0
    # the certified upper bound must contain the true optimum
    upper = limited.lambda_min * (1.0 + limited.gap_certified)
    assert full.lambda_min <= upper + 1e-9
    # without exact mode nothing is claimed
    plain = sifters.select(X, k=k, direction="forward")
    assert plain.proved_optimal is False
    assert plain.gap_certified == float("inf")


def test_exact_is_exported_in_to_dict():
    X = _data(n=150, m=12, seed=16)
    d = sifters.select(X, k=4, direction="forward", exact=True).to_dict()
    assert d["proved_optimal"] is True
    assert d["gap_certified"] == 0.0
    assert d["exact_evals"] > 0


# ---------------------------------------------------------------------------
# Accepted / rejected inputs
# ---------------------------------------------------------------------------

def test_accepts_various_array_like_inputs():
    X = _data(n=120, m=25, seed=2)
    ref = sifters.select(X, k=5, direction="forward")

    # same values, different storage: identical result
    for name, variant in {
        "fortran": np.asfortranarray(X),
        "transpose_view": np.ascontiguousarray(X.T).T,
        "list": X.tolist(),
    }.items():
        r = sifters.select(variant, k=5, direction="forward")
        assert r.subset == ref.subset, name
        assert abs(r.lambda_min - ref.lambda_min) < 1e-9, name

    # dtypes and non-contiguous views: the package converts, the result must be
    # the one of the same values passed as contiguous `float64`.
    for name, variant in {
        "non_contiguous": np.ascontiguousarray(X.T).T,
        "float32": X.astype(np.float32),
        "int": (X * 100).astype(np.int64),
        "row_slice": np.repeat(X, 2, axis=0)[::2],
    }.items():
        converted = np.ascontiguousarray(np.asarray(variant, dtype=np.float64))
        assert converted.shape == X.shape
        expected = sifters.select(converted, k=5, direction="forward")
        r = sifters.select(variant, k=5, direction="forward")
        assert r.subset == expected.subset, name
        assert abs(r.lambda_min - expected.lambda_min) < 1e-9, name


def test_as_matrix_returns_contiguous_float64():
    X = _data(n=10, m=4, seed=0)
    a = sifters.as_matrix(X.astype(np.float32).T.T)
    assert a.dtype == np.float64 and a.flags["C_CONTIGUOUS"]
    assert a.shape == X.shape


def test_rejects_bad_inputs():
    X = _data(n=60, m=10, seed=0)
    with pytest.raises(ValueError, match="2D"):
        sifters.select(X[:, 0], k=2)
    with pytest.raises(ValueError, match="2D"):
        sifters.select(X[None, :, :], k=2)
    with pytest.raises(TypeError, match="file path"):
        sifters.select("data.csv", k=2)
    with pytest.raises(sifters.SiftersError, match="non-finite"):
        bad = X.copy()
        bad[0, 0] = np.nan
        sifters.select(bad, k=2)
    with pytest.raises(sifters.SiftersError, match="non-finite"):
        bad = X.copy()
        bad[0, 0] = np.inf
        sifters.select(bad, k=2)
    with pytest.raises(ValueError, match=r"k/k(min|max)"):
        sifters.select(X, k=0)
    with pytest.raises(ValueError, match=r"k/k(min|max)"):
        sifters.select(X, k=11)
    with pytest.raises(ValueError, match="direction"):
        sifters.select(X, k=2, direction="sideways")
    with pytest.raises(ValueError, match="eval"):
        sifters.select(X, k=2, eval="nope")
    with pytest.raises(ValueError, match="repr"):
        sifters.select(X, k=2, representation="nope")


def test_extension_rejects_float32_without_conversion():
    """The native module requires a float64 buffer: converting is the job of the
    Python package (which also avoids a copy when the input already is float64)."""
    with pytest.raises(TypeError):
        sifters._sifters.run(np.zeros((4, 3), dtype=np.float32))


def test_legacy_api_is_gone():
    with pytest.raises(AttributeError, match="binary_path"):
        sifters.binary_path
    with pytest.raises(AttributeError, match="Selection"):
        sifters.Result
    with pytest.raises(AttributeError):
        sifters.i_do_not_exist


# ---------------------------------------------------------------------------
# Progress, interruption, threads
# ---------------------------------------------------------------------------

def test_progress_callback_receives_steps():
    X = _data(n=120, m=30, seed=4)
    seen = []
    r = sifters.select(X, k=8, direction="forward", progress=lambda s: seen.append(s))
    assert len(seen) == len(r.steps) == 7
    assert [s.k_after for s in seen] == [s.k_after for s in r.steps]
    assert all(isinstance(s, sifters.Step) for s in seen)
    assert "add" in repr(seen[0])


def test_progress_returning_none_does_not_stop():
    """A purely informative callback (returning `None`) must not interrupt."""
    X = _data(n=120, m=30, seed=4)
    r = sifters.select(X, k=8, direction="forward", progress=lambda s: None)
    assert len(r.steps) == 7


def test_progress_can_stop_early():
    X = _data(n=150, m=40, seed=6)
    calls = []

    def cb(step):
        calls.append(step.k_after)
        return False if len(calls) >= 3 else None

    r = sifters.path(X, kmin=1, direction="backward", progress=cb)
    assert len(calls) == len(r.steps) == 3
    assert r.k == 40 - 3  # the partial result stays consistent
    assert len(r.subset) == r.k


def test_threads_parameter_gives_identical_results():
    X = _data(n=200, m=120, seed=8)
    ref = sifters.select(X, k=20, direction="forward")
    par = sifters.select(X, k=20, direction="forward", threads=2)
    assert par.subset == ref.subset
    assert abs(par.lambda_min - ref.lambda_min) < 1e-9


def test_gil_is_released_during_computation():
    """The computation releases the GIL: another Python thread must keep running."""
    X = _data(n=400, m=2000, seed=9)
    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(time.monotonic())
            time.sleep(0.001)

    t = threading.Thread(target=ticker)
    t.start()
    try:
        sifters.select(X, k=40, direction="forward", prefilter=True)
    finally:
        stop.set()
        t.join()
    # without releasing the GIL, the thread could not run at all
    assert len(ticks) >= 1


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_to_dict_is_consistent():
    X = _data(n=120, m=30, seed=3)
    r = sifters.select(X, k=6, direction="backward")
    d = r.to_dict()
    assert d["k"] == r.k == 6
    assert d["subset"] == r.subset
    assert d["direction"] == "backward"
    assert abs(d["lambda_min"] - r.lambda_min) < 1e-15
    assert dict(d["curve"]) == dict(r.curve)
    assert len(d["steps"]) == len(r.steps)
    import json

    json.dumps(d)  # serializable as is


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # pragma: no cover - utility
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    print("Python binding OK" if not failures else f"{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
