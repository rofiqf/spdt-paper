"""Leakage tests: preprocessing hanya fit pada training, tuning tak
menyentuh outer test, dan outer splits identik antar metode."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.evaluate import inner_seed, outer_splits
from src.tree import StabilityTreeClassifier


def test_outer_splits_deterministic_and_identical():
    y = np.array([0, 1] * 25)
    cfg = {"cv": {"outer": {"seeds": [42, 123], "splits": 5, "repeats": 1},
                  "inner": {"splits": 3}}}
    a = list(outer_splits(y, cfg))
    b = list(outer_splits(y, cfg))
    assert len(a) == len(b) == 10  # 2 seeds x 5 folds
    for (s1, r1, f1, tr1, te1), (s2, r2, f2, tr2, te2) in zip(a, b):
        assert (s1, r1, f1) == (s2, r2, f2)
        assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2)
    # tidak ada sampel test yang bocor ke train dalam satu fold
    for _, _, _, tr, te in a:
        assert len(set(tr) & set(te)) == 0


def test_imputer_fitted_on_train_only():
    X_tr = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0],
                         "b": [10.0, 20.0, 10.0, 20.0, 10.0]})
    y_tr = np.array([0, 0, 1, 1, 0])
    imp = SimpleImputer(strategy="median").fit(X_tr)
    assert np.allclose(imp.statistics_, [3.0, 10.0])  # median train
    # test set berisi NaN + outlier ekstrem tidak mengubah statistics_
    X_te = pd.DataFrame({"a": [np.nan, 99999.0], "b": [-99999.0, np.nan]})
    Xt = imp.transform(X_te)
    assert np.allclose(imp.statistics_, [3.0, 10.0])
    assert Xt.shape == (2, 2)


def test_inner_tuning_invariant_to_outer_test_content():
    """Prosedur tuning yang sama dengan outer train yang sama harus
    menghasilkan best_params identik meski outer test diganti total —
    bukti outer test tidak memengaruhi seleksi hyperparameter."""
    rng = np.random.default_rng(21)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    y = (X["a"] + 0.2 * rng.normal(size=120) > 0).astype(int).values

    def tune(X_tr, y_tr, X_te):
        i_seed = inner_seed(42, 0, 0)
        cv = StratifiedKFold(3, shuffle=True, random_state=i_seed)
        gs = GridSearchCV(
            StabilityTreeClassifier(random_state=0),
            {"penalty_lambda": [0.0, 0.5], "n_bootstrap": [5],
             "max_depth": [2, 3]},
            scoring="f1_macro", cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr)
        return gs.best_params_

    y_perm = rng.permutation(y)
    idx_tr = np.arange(90)
    idx_te_a = np.arange(90, 120)
    idx_te_b = np.arange(30, 60)  # test alternatif berbeda
    p_a = tune(X.iloc[idx_tr], y_perm[idx_tr], X.iloc[idx_te_a])
    p_b = tune(X.iloc[idx_tr], y_perm[idx_tr], X.iloc[idx_te_b])
    assert p_a == p_b
