"""Reproducibility tests: seed sama -> hasil identik; engine run_dataset
reproducible end-to-end pada konfigurasi kecil."""

import numpy as np
import pandas as pd

from src.evaluate import build_model_specs, run_dataset
from src.tree import StabilityTreeClassifier


def test_same_seed_identical_predictions():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(90, 4))
    y = (X[:, 2] + 0.4 * rng.normal(size=90) > 0).astype(int)
    Xte = rng.normal(size=(25, 4))
    for lam in (0.1, 1.0, 10.0):
        a = StabilityTreeClassifier(penalty_lambda=lam, n_bootstrap=8,
                                    random_state=99).fit(X, y)
        b = StabilityTreeClassifier(penalty_lambda=lam, n_bootstrap=8,
                                    random_state=99).fit(X, y)
        assert list(a.predict(Xte)) == list(b.predict(Xte))
        pa, pb = a.predict_proba(Xte), b.predict_proba(Xte)
        assert np.allclose(pa, pb)


def test_run_dataset_reproducible(tmp_path):
    cfg = {
        "cv": {"outer": {"seeds": [42], "splits": 2, "repeats": 1},
               "inner": {"splits": 2}},
        "models": {
            "cart_custom": {"grid": {"max_depth": [2]},
                            "fixed": {"penalty_lambda": 0.0}},
            "spdt": {"grid": {"penalty_lambda": [0.1],
                              "n_bootstrap": [5],
                              "max_depth": [2]},
                     "fixed": {}},
        },
    }
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(60, 3)))
    y = pd.Series((X[0] > 0).astype(int).values)
    specs = build_model_specs(cfg)
    rows_a, _ = run_dataset("syn", X, y, cfg, specs, tmp_path / "a",
                            stability_resamples=0, verbose=False)
    rows_b, _ = run_dataset("syn", X, y, cfg, specs, tmp_path / "b",
                            stability_resamples=0, verbose=False)
    da, db = pd.DataFrame(rows_a), pd.DataFrame(rows_b)
    cols = ["f1_macro", "balanced_accuracy", "accuracy", "best_params"]
    pd.testing.assert_frame_equal(da[cols].reset_index(drop=True),
                                  db[cols].reset_index(drop=True))
    assert len(da) == 2 * 2  # 2 outer folds x 2 metode
