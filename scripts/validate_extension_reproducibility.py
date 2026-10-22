"""F4: bit-exact reproducibility probes for the extension experiment.

Recomputes the full staged-tuning pipeline for three stored folds and
compares predictions' f1_macro against normalized_penalty.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data as data_mod
from src import extension_evaluation as ee
from src.evaluate import outer_splits

PROBES = [("iris", "cart_custom", 42, 0, 0),
          ("wine", "spdt_norm", 2026, 2, 4),
          ("banknote", "anls", 123, 1, 2)]


def main():
    cfg = ee.default_cfg()
    cfg.update({"rf_trees": 100})
    stored = pd.read_csv(ee.EXT / "normalized_penalty.csv")
    all_ok = True
    for name, method, seed, repeat, fold in PROBES:
        X, y = data_mod.load_dataset(name)
        y_arr = np.asarray(y)
        for s, r, f, tr, te in outer_splits(y_arr, cfg):
            if (s, r, f) != (seed, repeat, fold):
                continue
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y_arr[tr], y_arr[te]
            cv_seed = ee.inner_seed(seed, repeat, fold)
            struct = ee.tune_structure(X_tr, y_tr, cv_seed)
            lam2 = ee.tune_lambda(X_tr, y_tr, struct, False, cv_seed,
                                   cfg["tune_B"], cfg["epsilon"])
            lam3 = ee.tune_lambda(X_tr, y_tr, struct, True, cv_seed,
                                   cfg["tune_B"], cfg["epsilon"])
            skl = ee.tune_sklearn(X_tr, y_tr, cv_seed)
            pipes = ee.make_method_pipes(struct, lam2, lam3, skl, cfg)
            p = clone(pipes[method])
            pred, _, _ = ee.fit_predict(p, X_tr, y_tr, X_te)
            f1_new = f1_score(y_te, pred, average="macro")
            row = stored[(stored.dataset == name)
                         & (stored.method == method)
                         & (stored.seed == seed)
                         & (stored.repeat == repeat)
                         & (stored.outer_fold == fold)].iloc[0]
            ok = abs(f1_new - row["f1_macro"]) < 1e-9
            all_ok &= ok
            print(f"{name}/{method}/s{seed}r{r}f{f}: "
                  f"stored={row['f1_macro']:.12f} "
                  f"recomputed={f1_new:.12f} {'PASS' if ok else 'FAIL'}")
    print("EXTENSION REPRODUCIBILITY:",
          "PASS" if all_ok else "FAIL — investigate")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
