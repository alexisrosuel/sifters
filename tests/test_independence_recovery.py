"""Ground-truth tests: the selector must recover the truly independent features.

The generator is in ``scripts/independence_recovery.py``.  A group of features is
built from a single latent source through possibly nonlinear links; the true
number of independent directions is the number of groups.  Asking for that ``k``
must return exactly one feature per group.
"""
import numpy as np
import pytest

import sifters
from independence_recovery import dependence_summary, make_ground_truth, recovery_report


def _ground_truth(**kwargs):
    kwargs.setdefault("n", 600)
    return make_ground_truth(**kwargs)


def test_generator_has_the_expected_structure():
    X, groups = _ground_truth(n=400, n_groups=3, per_group=2, n_singletons=2, seed=1)
    assert X.shape == (400, 3 * 2 + 2)
    assert groups.shape == (8,)
    assert set(groups.tolist()) == {0, 1, 2, 3, 4}
    assert np.allclose(X.mean(0), 0.0, atol=1e-12)
    assert np.allclose(X.std(0), 1.0, atol=1e-12)


def test_dependence_separates_within_groups_from_across_groups():
    X, groups = _ground_truth(seed=2)
    for method in ("dcor", "hsic", "nmi"):
        within, across = dependence_summary(X, groups, method)
        assert within > 3.0 * across, (method, within, across)


@pytest.mark.parametrize("method", ["dcor", "hsic", "nmi"])
def test_nonlinear_methods_recover_one_feature_per_group(method):
    X, groups = _ground_truth(seed=0)
    rep = recovery_report(X, groups, method=method, direction="backward", exact=True)
    assert rep["one_per_group"], rep
    assert sorted(rep["picks_per_group"].values()) == [1] * rep["n_groups"]


@pytest.mark.parametrize("method", ["linear", "rank"])
def test_signed_linear_methods_are_fooled_by_non_monotone_links(method):
    """s and s**2 are uncorrelated: the linear/copula criteria keep both."""
    X, groups = _ground_truth(seed=0)
    rep = recovery_report(X, groups, method=method, direction="backward", exact=True)
    assert not rep["one_per_group"], rep
    assert max(rep["picks_per_group"].values()) >= 2


def test_rank_recovers_when_the_links_are_monotone():
    """The Gaussian copula handles every monotone link, and only those."""
    X, groups = _ground_truth(
        per_group=3, n_singletons=2, seed=3, links=("linear", "tanh")
    )
    rep = recovery_report(X, groups, method="rank", direction="backward", exact=True)
    assert rep["one_per_group"], rep


def test_dopt_recovers_the_groups_too():
    X, groups = _ground_truth(seed=4)
    k = int(groups.max()) + 1
    res = sifters.select_dependence(X, k, method="dcor", criterion="D", exact=True)
    picked = groups[np.asarray(res.subset, dtype=int)]
    assert len(np.unique(picked)) == k
    assert res.proved_optimal


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_recovery_is_stable_across_seeds(seed):
    X, groups = _ground_truth(seed=seed)
    rep = recovery_report(X, groups, method="dcor", direction="backward", exact=True)
    assert rep["one_per_group"], (seed, rep)


def test_asking_for_fewer_features_gives_one_per_selected_group():
    X, groups = _ground_truth(seed=5)
    rep = recovery_report(X, groups, method="dcor", k=3, direction="backward", exact=True)
    assert len(rep["groups_selected"]) == 3
    assert max(rep["picks_per_group"].values()) == 1
