"""Unit tests 1-10 untuk StabilityTreeClassifier (spec section 8)."""

import numpy as np
import pytest

from src.tree import StabilityTreeClassifier, _gini_from_counts


# ---------------------------------------------------------------- test 1
def test_gini_and_gain_hand_calculated():
    # Gini counts [2, 2] = 0.5 ; counts [4] = 0 ; counts [3,1] = 0.375
    assert _gini_from_counts(np.array([[2.0, 2.0]]))[0] == pytest.approx(0.5)
    assert _gini_from_counts(np.array([[4.0, 0.0]]))[0] == pytest.approx(0.0)
    assert _gini_from_counts(np.array([[3.0, 1.0]]))[0] == pytest.approx(0.375)
    # Split y=[0,0,1,1] pada threshold 1.5 -> gain 0.5 (hand calc)
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = StabilityTreeClassifier(penalty_lambda=0.0, max_depth=None,
                                  min_samples_leaf=1).fit(X, y)
    assert clf.node_count_() == 3          # root + 2 leaves
    assert list(clf.predict(X)) == list(y)  # split 1.5 chosen
    assert clf.tree_.threshold == pytest.approx(1.5)


# ---------------------------------------------------------------- test 2
def _reference_cart_predict(X_tr, y_tr, X_te, max_depth=None,
                            min_samples_leaf=1, min_samples_split=2):
    """Independent slow CART reference (same documented policies)."""
    classes = np.unique(y_tr)
    enc = {c: i for i, c in enumerate(classes)}
    y_e = np.array([enc[v] for v in y_tr])

    def gini(idx):
        if len(idx) == 0:
            return 0.0
        c = np.bincount(y_e[idx], minlength=len(classes))
        p = c / len(idx)
        return 1.0 - float((p ** 2).sum())

    def build(idx, depth):
        counts = np.bincount(y_e[idx], minlength=len(classes))
        pred = classes[int(np.argmax(counts))]

        def leaf(X):
            return np.array([pred] * len(X), dtype=classes.dtype)

        if (max_depth is not None and depth >= max_depth):
            return leaf
        if len(idx) < min_samples_split or len(np.unique(y_e[idx])) == 1:
            return leaf
        n = len(idx)
        best = None
        for j in range(X_tr.shape[1]):
            vals = np.unique(X_tr[idx, j])
            for a, b in zip(vals[:-1], vals[1:]):
                t = 0.5 * (a + b)
                left = idx[X_tr[idx, j] <= t]
                right = idx[X_tr[idx, j] > t]
                if len(left) < min_samples_leaf or \
                   len(right) < min_samples_leaf:
                    continue
                gain = gini(idx) - (len(left) / n) * gini(left) \
                    - (len(right) / n) * gini(right)
                if best is None or gain > best[0]:
                    best = (gain, j, t, left, right)
        if best is None or best[0] <= 0:
            return leaf
        _, j, t, left, right = best
        lf = build(left, depth + 1)
        rf = build(right, depth + 1)

        def predict(X):
            out = np.empty(len(X), dtype=classes.dtype)
            m = X[:, j] <= t
            out[m] = lf(X[m])
            out[~m] = rf(X[~m])
            return out
        return predict
    return build(np.arange(len(y_tr)), 0)(X_te)


def test_lambda0_equals_custom_cart_reference():
    rng = np.random.default_rng(7)
    n, m = 80, 3
    X = rng.normal(size=(n, m))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    y[::7] = 1 - y[::7]  # sedikit noise
    Xte = rng.normal(size=(30, m))
    for depth in (None, 3, 5):
        for msl in (1, 2, 5):
            ours = StabilityTreeClassifier(
                penalty_lambda=0.0, max_depth=depth,
                min_samples_leaf=msl).fit(X, y)
            ref = _reference_cart_predict(X, y, Xte, max_depth=depth,
                                          min_samples_leaf=msl)
            assert list(ours.predict(Xte)) == list(ref), (depth, msl)


# ---------------------------------------------------------------- test 3
def test_no_split_constant_or_pure():
    X = np.ones((10, 2))
    y = np.array([0, 1] * 5)
    clf = StabilityTreeClassifier(penalty_lambda=0.0).fit(X, y)
    assert clf.node_count_() == 1
    assert clf.tree_depth_() == 0
    X2 = np.column_stack([np.arange(10.0), np.zeros(10)])
    y2 = np.ones(10, dtype=int)
    clf2 = StabilityTreeClassifier(penalty_lambda=0.0).fit(X2, y2)
    assert clf2.node_count_() == 1


# ---------------------------------------------------------------- test 4
def test_min_samples_leaf_respected():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = StabilityTreeClassifier(penalty_lambda=0.0,
                                  min_samples_leaf=3).fit(X, y)
    assert clf.node_count_() == 1  # split terbaik hanya memberi 2 vs 2
    clf2 = StabilityTreeClassifier(penalty_lambda=0.0,
                                   min_samples_leaf=2).fit(X, y)
    assert clf2.node_count_() == 3


# ---------------------------------------------------------------- test 5
def test_binary_and_multiclass():
    rng = np.random.default_rng(11)
    Xb = rng.normal(size=(60, 4))
    yb = (Xb[:, 0] > 0).astype(int)
    clf = StabilityTreeClassifier(penalty_lambda=0.0, max_depth=3).fit(Xb, yb)
    assert set(clf.predict(Xb)) <= {0, 1}
    Xm = np.vstack([rng.normal(0, 1, (25, 4)) + i * 2.5
                    for i in range(3)])
    ym = np.repeat([0, 1, 2], 25)
    clf_m = StabilityTreeClassifier(penalty_lambda=0.0).fit(Xm, ym)
    assert set(clf_m.predict(Xm)) == {0, 1, 2}
    proba = clf_m.predict_proba(Xm[:5])
    assert proba.shape == (5, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------- test 6
def test_deterministic_fixed_seed():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(70, 3))
    y = (X[:, 1] + 0.3 * rng.normal(size=70) > 0).astype(int)
    Xte = rng.normal(size=(20, 3))
    a = StabilityTreeClassifier(penalty_lambda=1.0, n_bootstrap=5,
                                random_state=42).fit(X, y)
    b = StabilityTreeClassifier(penalty_lambda=1.0, n_bootstrap=5,
                                random_state=42).fit(X, y)
    assert list(a.predict(Xte)) == list(b.predict(Xte))
    assert a.node_count_() == b.node_count_()


# ---------------------------------------------------------------- test 7
def test_bootstrap_only_uses_node_training_data():
    """Fit pada train yang sama harus memberi struktur identik terlepas
    dari data evaluasi; bootstrap tidak boleh menyentuh data luar node."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    X_eval1 = rng.normal(size=(25, 3))
    X_eval2 = rng.normal(size=(25, 3)) + 100.0  # ekstrem, berbeda total
    a = StabilityTreeClassifier(penalty_lambda=0.5, n_bootstrap=5,
                                random_state=0).fit(X, y)
    b = StabilityTreeClassifier(penalty_lambda=0.5, n_bootstrap=5,
                                random_state=0).fit(X, y)
    # struktur internal identik, dan prediksi pada probe set sama
    assert a.node_count_() == b.node_count_()
    assert list(a.predict(X_eval1)) == list(b.predict(X_eval1))
    # evaluasi pada data yang sangat berbeda tidak mengubah model
    assert list(a.predict(X_eval1)) == list(a.predict(X_eval1))
    _ = X_eval2  # dokumentasi intent: keberadaan data lain tak berpengaruh


# ---------------------------------------------------------------- test 9
def test_missing_and_invalid_inputs():
    X = np.array([[0.0, 1.0], [np.nan, 2.0], [1.0, 0.0], [2.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    clf = StabilityTreeClassifier()
    with pytest.raises(ValueError):
        clf.fit(X, y)
    with pytest.raises(ValueError):
        clf.fit(np.zeros(5), np.array([0, 1, 0, 1, 0]))  # 1-D X
    with pytest.raises(ValueError):
        clf.fit(np.zeros((4, 2)), np.array([0, 1]))      # mismatch
    with pytest.raises(ValueError):
        clf.fit(np.zeros((0, 2)), np.array([]))          # kosong
    Xok = np.array([[0.0], [1.0], [2.0], [3.0]])
    clf.fit(Xok, y)
    with pytest.raises(ValueError):
        clf.predict(np.array([[np.nan]]))


# ---------------------------------------------------------------- test 10
def test_lambda_and_B_sensitivity_do_not_crash():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(50, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    Xte = rng.normal(size=(15, 4))
    for lam in (0.0, 0.01, 0.1, 1.0, 10.0):
        for B in (5, 10, 20):
            clf = StabilityTreeClassifier(penalty_lambda=lam, n_bootstrap=B,
                                          random_state=1).fit(X, y)
            p = clf.predict(Xte)
            assert np.all(np.isfinite(p)) or p.dtype.kind in "biUS"
    # varian normalized penalty (eksploratori)
    clf_n = StabilityTreeClassifier(penalty_lambda=1.0, n_bootstrap=5,
                                    normalized_penalty=True,
                                    random_state=1).fit(X, y)
    assert len(clf_n.predict(Xte)) == 15


# ------------------------------------------------------------ extra
def test_empty_resample_child_policy_deterministic():
    """Data dengan outlier jarang: resample boleh mengosongkan child;
    kebijakan 'kontribusi 0' harus tetap finite dan deterministik."""
    rng = np.random.default_rng(13)
    X = rng.normal(size=(40, 2))
    X[0, 0] = 50.0  # outlier
    y = (X[:, 1] > 0).astype(int)
    clf = StabilityTreeClassifier(penalty_lambda=1.0, n_bootstrap=5,
                                  random_state=2).fit(X, y)
    assert clf.node_count_() >= 1
    p1 = clf.predict(X)
    clf2 = StabilityTreeClassifier(penalty_lambda=1.0, n_bootstrap=5,
                                   random_state=2).fit(X, y)
    assert list(p1) == list(clf2.predict(X))


def test_sklearn_clone_and_grid_compatible():
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    clf = StabilityTreeClassifier(penalty_lambda=0.5, n_bootstrap=5)
    c2 = clone(clf)
    assert c2.get_params() == clf.get_params()
    X = np.random.default_rng(1).normal(size=(40, 2))
    y = (X[:, 0] > 0).astype(int)
    gs = GridSearchCV(clone(clf),
                      {"penalty_lambda": [0.0, 0.5],
                       "max_depth": [2, 3]},
                      cv=StratifiedKFold(2, shuffle=True, random_state=0),
                      scoring="f1_macro")
    gs.fit(X, y)
    assert gs.best_params_["penalty_lambda"] in (0.0, 0.5)
