"""Validasi reproducibility: rehitung beberapa outer fold tersimpan dan
bandingkan dengan fold_metrics.csv. PASS bila selisih < 1e-9.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data as data_mod
from src.evaluate import build_model_specs, inner_seed, outer_splits


def refit_and_score(X, y, config, spec, seed, repeat, fold):
    y_arr = np.asarray(y)
    splits = list(outer_splits(y_arr, config))
    tr, te = None, None
    for s, r, f, tr_i, te_i in splits:
        if (s, r, f) == (seed, repeat, fold):
            tr, te = tr_i, te_i
            break
    assert tr is not None
    est = spec["factory"](**spec["fixed"])
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("clf", est)])
    grid = {f"clf__{k}": v for k, v in spec["grid"].items()}
    i_seed = inner_seed(seed, repeat, fold)
    cv = StratifiedKFold(config["cv"]["inner"]["splits"], shuffle=True,
                         random_state=i_seed)
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
    gs.fit(X.iloc[tr], y_arr[tr])
    pred = gs.predict(X.iloc[te])
    return f1_score(y_arr[te], pred, average="macro")


def main():
    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    specs = build_model_specs(config)
    stored = pd.read_csv(ROOT / "results/raw/fold_metrics.csv")

    probes = [("iris", "cart_custom", 42, 0, 0),
              ("iris", "spdt", 2026, 2, 4),
              ("wine", "cart_sklearn", 123, 1, 2)]
    all_ok = True
    for dataset, method, seed, repeat, fold in probes:
        drop = config["datasets"][dataset].get("drop_columns")
        X, y = data_mod.load_dataset(dataset, drop_columns=drop)
        f1_new = refit_and_score(X, y, config, specs[method], seed, repeat,
                                 fold)
        row = stored[(stored.dataset == dataset)
                     & (stored.method == method)
                     & (stored.seed == seed)
                     & (stored.repeat == repeat)
                     & (stored.outer_fold == fold)].iloc[0]
        ok = abs(f1_new - row["f1_macro"]) < 1e-9
        all_ok &= ok
        print(f"{dataset}/{method}/seed{seed}/r{repeat}/f{fold}: "
              f"stored={row['f1_macro']:.12f} recomputed={f1_new:.12f} "
              f"{'PASS' if ok else 'FAIL'}")
    print("REPRODUCIBILITY:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
