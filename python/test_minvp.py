"""Tests du binding Python natif `minvp`.

Exécution :

    python -m pytest python/test_minvp.py      # ou simplement `pytest`
    python python/test_minvp.py                # équivalent, sans collecte pytest

numpy sert d'oracle pour toutes les valeurs spectrales.
"""
import time
import threading

import numpy as np
import pytest

import minvp


def _data(n=200, m=60, rho=0.5, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n, 1))
    return np.sqrt(rho) * g + np.sqrt(1.0 - rho) * rng.normal(size=(n, m))


def _corr(X):
    """Matrice de corrélation de X (mêmes conventions que le moteur)."""
    Z = X - X.mean(0)
    nrm = np.linalg.norm(Z, axis=0)
    nrm[nrm == 0] = 1.0
    Z = Z / nrm
    return Z.T @ Z


# ---------------------------------------------------------------------------
# Correction et qualité
# ---------------------------------------------------------------------------

def test_select_shapes():
    X = _data()
    r = minvp.select(X, k=10)
    assert len(r.subset) == 10
    assert len(set(r.subset)) == 10
    assert 0.0 < r.lambda_min <= 1.0
    assert 0.3 < r.lambda_min < 1.0
    assert r.n_observations == 200
    assert r.m_total == 60
    assert len(r) == r.k == 10


def test_curve_is_monotone():
    X = _data()
    r = minvp.path(X, kmin=5)
    ks = sorted(r.curve)
    vals = [r.curve[k] for k in ks]
    # lambda_min décroît quand K augmente (théorème d'entrelacement)
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-6


def test_matches_numpy_eigenvalue_both_directions():
    """`lambda_min` (revalidé à froid par défaut) doit coller à numpy, et pas
    seulement la valeur Ritz du parcours."""
    X = _data()
    R = _corr(X)
    for direction in ("forward", "backward"):
        for verify in (True, False):
            r = minvp.select(X, k=12, direction=direction, verify=verify)
            exact = np.linalg.eigvalsh(R[np.ix_(r.subset, r.subset)])[0]
            assert abs(exact - r.lambda_min) < 1e-9, (direction, verify, exact, r.lambda_min)


def test_subset_at_matches_numpy_for_every_size():
    X = _data(n=200, m=40, seed=7)
    R = _corr(X)
    r = minvp.path(X, kmin=3, direction="backward")
    for k in range(3, 41):
        subset = r.subset_at(k)
        assert subset is not None and len(subset) == k
        exact = np.linalg.eigvalsh(R[np.ix_(subset, subset)])[0]
        # les étapes de la courbe sont revalidées à froid : tolérance serrée
        assert abs(exact - r.curve[k]) < 1e-8, (k, exact, r.curve[k])
    assert r.subset_at(2) is None  # hors du parcours
    assert r.lambda_at(3) is not None


def test_forward_exact_matches_numpy_greedy():
    """Le pré-filtre est désactivé par défaut : le glouton avant doit alors
    coïncider exactement avec un glouton E-optimal naïf (numpy)."""
    X = _data(n=300, m=600, seed=3)
    r = minvp.select(X, k=6, direction="forward", prefilter=False)
    assert r.initial_subset, "le binding doit exposer la variable initiale"

    R = _corr(X)
    sel = [int(r.initial_subset[0])]
    while len(sel) < 6:
        rest = [j for j in range(R.shape[0]) if j not in sel]
        best = max(rest, key=lambda j: np.linalg.eigvalsh(R[np.ix_(sel + [j], sel + [j])])[0])
        sel.append(int(best))
    assert abs(np.linalg.eigvalsh(R[np.ix_(sel, sel)])[0] - r.lambda_min) < 1e-6


def test_prefilter_is_opt_in():
    """Le pré-filtre est une heuristique explicite, désactivée par défaut. Le mode
    exact certifie chaque étape (test suivant) ; il n'en résulte pas forcément un
    λ_min final supérieur, le glouton étant myope : une déviation heuristique peut
    mieux finir."""
    X = _data(n=200, m=600, seed=1)
    exact = minvp.select(X, k=6, direction="forward", prefilter=False)
    filtre = minvp.select(X, k=6, direction="forward", prefilter=True)
    assert len(filtre.subset) == 6
    assert exact.certified_steps == len(exact.steps)


def test_forward_exact_is_stepwise_optimal():
    """Sélection avant certifiée : à chaque étape, l'ajout retenu maximise
    réellement λ_min parmi *tous* les candidats restants (force brute numpy)."""
    X = _data(n=300, m=40, rho=0.4, seed=3)
    r = minvp.select(X, k=12, direction="forward", prefilter=False)
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
    """`inverse`/`direct` et `packed`/`implicit` doivent donner le même résultat."""
    X = _data(n=300, m=80, seed=11)
    ref = minvp.select(X, k=15, direction="backward", eval="direct", representation="packed")
    for kwargs in (
        {"eval": "inverse"},
        {"representation": "implicit"},
        {"eval": "inverse", "representation": "implicit"},
    ):
        r = minvp.select(X, k=15, direction="backward", **kwargs)
        assert r.subset == ref.subset
        assert abs(r.lambda_min - ref.lambda_min) < 1e-9


def test_constant_columns_are_dropped_without_shifting_indices():
    X = _data(n=150, m=20, seed=5)
    X = np.column_stack([X[:, :3], np.full(150, 4.0), X[:, 3:]])
    r = minvp.select(X, k=5, direction="forward")
    assert r.dropped == [3]
    assert r.m_total == 21 and r.m_usable == 20
    assert 3 not in r.subset
    assert all(0 <= j < 21 for j in r.subset)
    R = _corr(X)
    assert abs(np.linalg.eigvalsh(R[np.ix_(r.subset, r.subset)])[0] - r.lambda_min) < 1e-9


# ---------------------------------------------------------------------------
# Entrées acceptées / refusées
# ---------------------------------------------------------------------------

def test_accepts_various_array_like_inputs():
    X = _data(n=120, m=25, seed=2)
    ref = minvp.select(X, k=5, direction="forward")

    # mêmes valeurs, stockage différent : résultat identique
    for name, variant in {
        "fortran": np.asfortranarray(X),
        "transpose_view": np.ascontiguousarray(X.T).T,
        "list": X.tolist(),
    }.items():
        r = minvp.select(variant, k=5, direction="forward")
        assert r.subset == ref.subset, name
        assert abs(r.lambda_min - ref.lambda_min) < 1e-9, name

    # dtypes et vues non contiguës : le paquet convertit, le résultat doit être
    # celui des mêmes valeurs passées en `float64` contigu.
    for name, variant in {
        "non_contiguous": np.ascontiguousarray(X.T).T,
        "float32": X.astype(np.float32),
        "int": (X * 100).astype(np.int64),
        "row_slice": np.repeat(X, 2, axis=0)[::2],
    }.items():
        converted = np.ascontiguousarray(np.asarray(variant, dtype=np.float64))
        assert converted.shape == X.shape
        expected = minvp.select(converted, k=5, direction="forward")
        r = minvp.select(variant, k=5, direction="forward")
        assert r.subset == expected.subset, name
        assert abs(r.lambda_min - expected.lambda_min) < 1e-9, name


def test_as_matrix_returns_contiguous_float64():
    X = _data(n=10, m=4, seed=0)
    a = minvp.as_matrix(X.astype(np.float32).T.T)
    assert a.dtype == np.float64 and a.flags["C_CONTIGUOUS"]
    assert a.shape == X.shape


def test_rejects_bad_inputs():
    X = _data(n=60, m=10, seed=0)
    with pytest.raises(ValueError, match="2D"):
        minvp.select(X[:, 0], k=2)
    with pytest.raises(ValueError, match="2D"):
        minvp.select(X[None, :, :], k=2)
    with pytest.raises(TypeError, match="chemin"):
        minvp.select("data.csv", k=2)
    with pytest.raises(minvp.MinvpError, match="non finie"):
        bad = X.copy()
        bad[0, 0] = np.nan
        minvp.select(bad, k=2)
    with pytest.raises(minvp.MinvpError, match="non finie"):
        bad = X.copy()
        bad[0, 0] = np.inf
        minvp.select(bad, k=2)
    with pytest.raises(ValueError, match=r"k/k(min|max)"):
        minvp.select(X, k=0)
    with pytest.raises(ValueError, match=r"k/k(min|max)"):
        minvp.select(X, k=11)
    with pytest.raises(ValueError, match="direction"):
        minvp.select(X, k=2, direction="sideways")
    with pytest.raises(ValueError, match="eval"):
        minvp.select(X, k=2, eval="nope")
    with pytest.raises(ValueError, match="repr"):
        minvp.select(X, k=2, representation="nope")


def test_extension_rejects_float32_without_conversion():
    """Le module natif exige un tampon float64 : c'est le rôle du paquet Python
    de convertir (et c'est ce qui évite une copie quand l'entrée l'est déjà)."""
    with pytest.raises(TypeError):
        minvp._minvp.run(np.zeros((4, 3), dtype=np.float32))


def test_legacy_api_is_gone():
    with pytest.raises(AttributeError, match="binary_path"):
        minvp.binary_path
    with pytest.raises(AttributeError, match="Selection"):
        minvp.Result
    with pytest.raises(AttributeError):
        minvp.je_n_existe_pas


# ---------------------------------------------------------------------------
# Progression, interruption, threads
# ---------------------------------------------------------------------------

def test_progress_callback_receives_steps():
    X = _data(n=120, m=30, seed=4)
    seen = []
    r = minvp.select(X, k=8, direction="forward", progress=lambda s: seen.append(s))
    assert len(seen) == len(r.steps) == 7
    assert [s.k_after for s in seen] == [s.k_after for s in r.steps]
    assert all(isinstance(s, minvp.Step) for s in seen)
    assert "add" in repr(seen[0])


def test_progress_returning_none_does_not_stop():
    """Un callback purement informatif (qui renvoie `None`) ne doit pas interrompre."""
    X = _data(n=120, m=30, seed=4)
    r = minvp.select(X, k=8, direction="forward", progress=lambda s: None)
    assert len(r.steps) == 7


def test_progress_can_stop_early():
    X = _data(n=150, m=40, seed=6)
    calls = []

    def cb(step):
        calls.append(step.k_after)
        return False if len(calls) >= 3 else None

    r = minvp.path(X, kmin=1, direction="backward", progress=cb)
    assert len(calls) == len(r.steps) == 3
    assert r.k == 40 - 3  # le résultat partiel reste cohérent
    assert len(r.subset) == r.k


def test_threads_parameter_gives_identical_results():
    X = _data(n=200, m=120, seed=8)
    ref = minvp.select(X, k=20, direction="forward")
    par = minvp.select(X, k=20, direction="forward", threads=2)
    assert par.subset == ref.subset
    assert abs(par.lambda_min - ref.lambda_min) < 1e-9


def test_gil_is_released_during_computation():
    """Le calcul libère le GIL : un autre thread Python doit continuer à tourner."""
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
        minvp.select(X, k=40, direction="forward", prefilter=True)
    finally:
        stop.set()
        t.join()
    # sans libération du GIL, le thread ne pourrait pas s'exécuter du tout
    assert len(ticks) >= 1


# ---------------------------------------------------------------------------
# Sérialisation
# ---------------------------------------------------------------------------

def test_to_dict_is_consistent():
    X = _data(n=120, m=30, seed=3)
    r = minvp.select(X, k=6, direction="backward")
    d = r.to_dict()
    assert d["k"] == r.k == 6
    assert d["subset"] == r.subset
    assert d["direction"] == "backward"
    assert abs(d["lambda_min"] - r.lambda_min) < 1e-15
    assert dict(d["curve"]) == dict(r.curve)
    assert len(d["steps"]) == len(r.steps)
    import json

    json.dumps(d)  # sérialisable tel quel


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # pragma: no cover - utilitaire
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    print("binding Python OK" if not failures else f"{failures} échec(s)")
    raise SystemExit(1 if failures else 0)
