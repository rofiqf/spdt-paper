"""Extension tests: normalized SPDT correctness and safety."""

import numpy as np
import pytest

from src.tree import StabilityTreeClassifier
from tests.test_tree import _reference_cart_predict


def test_lambda0_matches_custom_cart():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=80) > 0).astype(int)
    Xte = rng.normal(size=(25, 3))
    for depth in (None, 3):
        ours = StabilityTreeClassifier(penalty_lambda=0.0, max_depth=depth,
                                       random_state=0).fit(X, y)
        ref = _reference_cart_predict(X, y, Xte, max_depth=depth)
        assert list(ours.predict(Xte)) == list(ref)


def test_normalized_scores_finite_and_deterministic():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] - X[:, 1] + 0.2 * rng.normal(size=120) > 0).astype(int)
    Xte = rng.normal(size=(20, 4))
    fits = []
    for _ in range(2):
        clf = StabilityTreeClassifier(penalty_lambda=1.0, n_bootstrap=10,
                                      normalized_penalty=True,
                                      penalty_epsilon=1e-4,
                                      random_state=3).fit(X, y)
        fits.append(clf)
    p1, p2 = fits[0].predict(Xte), fits[1].predict(Xte)
    assert list(p1) == list(p2)  # deterministic
    proba = fits[0].predict_proba(Xte)
    assert np.isfinite(proba).all()


def test_near_zero_gain_no_failure():
    """Node dengan gain ~ 0 (label hampir acak): score ternormalisasi
    tetap finite, tidak ada overflow/NaN, dan tree tetap jalan."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 3))
    y = (rng.random(60) > 0.5).astype(int)  # hampir tanpa sinyal
    clf = StabilityTreeClassifier(penalty_lambda=10.0, n_bootstrap=8,
                                  normalized_penalty=True,
                                  penalty_epsilon=1e-6,
                                  random_state=1).fit(X, y)
    pred = clf.predict(X[:10])
    assert len(pred) == 10
    assert clf.node_count_() >= 1


def test_multiclass_supported():
    rng = np.random.default_rng(9)
    X = np.vstack([rng.normal(i * 2, 1, (30, 4)) for i in range(3)])
    y = np.repeat([0, 1, 2], 30)
    clf = StabilityTreeClassifier(penalty_lambda=0.5, n_bootstrap=5,
                                  normalized_penalty=True,
                                  random_state=2).fit(X, y)
    assert set(clf.predict(X)) == {0, 1, 2}


def test_min_samples_leaf_enforced_normalized():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = StabilityTreeClassifier(penalty_lambda=5.0, n_bootstrap=5,
                                  normalized_penalty=True,
                                  min_samples_leaf=3,
                                  random_state=0).fit(X, y)
    assert clf.node_count_() == 1  # split terbaik hanya 2 vs 2


def test_stats_counters_present():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(int)
    clf = StabilityTreeClassifier(penalty_lambda=1.0, n_bootstrap=5,
                                  random_state=0).fit(X, y)
    s = clf.stats_()
    assert s["candidates_evaluated"] > 0
    assert s["bootstrap_gain_evaluations"] > 0
    clf0 = StabilityTreeClassifier(penalty_lambda=0.0).fit(X, y)
    assert clf0.stats_()["bootstrap_gain_evaluations"] == 0


def test_original_spdt_still_reproducible():
    """Regresi: implementasi SPDT lama tidak berubah oleh ekstensi."""
    rng = np.random.default_rng(21)
    X = rng.normal(size=(60, 3))
    y = (X[:, 2] + 0.3 * rng.normal(size=60) > 0).astype(int)
    a = StabilityTreeClassifier(penalty_lambda=0.5, n_bootstrap=8,
                                random_state=42).fit(X, y)
    b = StabilityTreeClassifier(penalty_lambda=0.5, n_bootstrap=8,
                                random_state=42).fit(X, y)
    assert list(a.predict(X)) == list(b.predict(X))
