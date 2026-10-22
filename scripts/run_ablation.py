"""Stage 7a: ablation terkontrol pada outer splits yang IDENTIK dengan main
run (dipakai ulang dari config). Variants:
  A0: lambda=0 (CART custom)
  A1: penalti aktif, B=10 tetap, lambda in {0.1, 1.0}
  A2: sensitivitas B in {5, 10, 20} pada lambda=1.0
  A3: sensitivitas depth/leaf (depth x leaf-size) pada lambda=1.0, B=10
Output: results/raw/ablation_folds.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data as data_mod
from src.evaluate import outer_splits
from src.tree import StabilityTreeClassifier


def variants():
    vs = {"A0_lam0": dict(penalty_lambda=0.0, n_bootstrap=10),
          "A1_lam0.1_B10": dict(penalty_lambda=0.1, n_bootstrap=10),
          "A1_lam1_B10": dict(penalty_lambda=1.0, n_bootstrap=10),
          "A2_B5": dict(penalty_lambda=1.0, n_bootstrap=5),
          "A2_B20": dict(penalty_lambda=1.0, n_bootstrap=20)}
    for d in (3, 8):
        for leaf in (1, 5):
            vs[f"A3_d{d}_leaf{leaf}"] = dict(
                penalty_lambda=1.0, n_bootstrap=10,
                max_depth=d, min_samples_leaf=leaf)
    return vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text())
    vs = variants()
    names = args.datasets or list(config["datasets"].keys())
    out = ROOT / "results" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        drop = config["datasets"][name].get("drop_columns")
        X, y = data_mod.load_dataset(name, drop_columns=drop)
        Xv = X.values
        yv = np.asarray(y)
        for seed, repeat, fold, tr, te in outer_splits(yv, config):
            imputer = SimpleImputer(strategy="median").fit(Xv[tr])
            Xi = imputer.transform(Xv)
            for vname, kw in vs.items():
                clf = StabilityTreeClassifier(random_state=0, **kw)
                t0 = time.perf_counter()
                clf.fit(Xi[tr], yv[tr])
                fit_s = time.perf_counter() - t0
                pred = clf.predict(Xi[te])
                rows.append({
                    "dataset": name, "variant": vname, "seed": seed,
                    "repeat": repeat, "outer_fold": fold,
                    **{k: v for k, v in kw.items()},
                    "f1_macro": f1_score(yv[te], pred, average="macro"),
                    "balanced_accuracy": balanced_accuracy_score(yv[te], pred),
                    "accuracy": accuracy_score(yv[te], pred),
                    "fit_seconds": fit_s,
                    "node_count": clf.node_count_(),
                    "tree_depth": clf.tree_depth_(),
                })
            print(f"[{name}] seed={seed} r={repeat} f={fold} ablation done",
                  flush=True)
        tmp = out / ".abl.tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.rename(out / "ablation_folds.csv")
    print(f"ablation rows: {len(rows)}")


if __name__ == "__main__":
    main()
