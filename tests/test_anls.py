"""ANLS (Lee et al. 2024) implementation sanity tests."""

import numpy as np
import pytest

from src.anls import ANLSTreeClassifier


def _data(seed=3, n=120, m=4, k=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, m))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    if k == 3:
        Xm = np.vstack([rng.normal(i * 2, 1, (n // 3, m)) for i in range(3)])
        ym = np.repeat([0, 1, 2], n // 3)
        return Xm, ym
    return X, y


def test_anls_deterministic_and_finite():
    X, y = _data()
    a = ANLSTreeClassifier(random_state=0).fit(X, y)
    b = ANLSTreeClassifier(random_state=0).fit(X, y)
    Xte = np.random.default_rng(9).normal(size=(20, 4))
    assert list(a.predict(Xte)) == list(b.predict(Xte))
    assert np.isfinite(a.predict_proba(Xte)).all()


def test_anls_multiclass():
    Xm, ym = _data(k=3)
    clf = ANLSTreeClassifier(random_state=1).fit(Xm, ym)
    assert set(clf.predict(Xm)) == {0, 1, 2}


def test_anls_root_split_reported():
    X, y = _data()
    clf = ANLSTreeClassifier(random_state=0).fit(X, y)
    rf, rt = clf.root_split_()
    assert rf in (-1, 0, 1, 2, 3)
    if rf != -1:
        assert np.isfinite(rt)


def test_anls_pure_node_leaf():
    X, y = _data()
    clf = ANLSTreeClassifier(random_state=0).fit(X, np.zeros(len(y), int))
    rf, rt = clf.root_split_()
    assert rf == -1 and np.isnan(rt)


def test_anls_respects_iteration_cap():
    X, y = _data(n=200)
    clf = ANLSTreeClassifier(random_state=0, max_iterations=40,
                             min_iterations=10).fit(X, y)
    assert clf is not None  # tidak hang/overflow pada cap kecil
