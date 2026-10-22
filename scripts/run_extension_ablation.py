"""Experiment E: controlled ablation on identical outer splits.

Uses per-fold frozen structure from logs/extension_tuned_params.json
(produced by the full extension run). Variants: lambda=0, original and
normalized penalty at several lambda, B sensitivity, epsilon sensitivity,
depth and leaf-size restrictions. Fixed-config fits only (no retuning).
"""

from __future__ import annotations

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


def variants(struct):
    vs = {"A0_lam0": dict(penalty_lambda=0.0, n_bootstrap=10, **struct),
          "A1_lam0.1": dict(penalty_lambda=0.1, n_bootstrap=10, **struct),
          "A1_lam1": dict(penalty_lambda=1.0, n_bootstrap=10, **struct),
          "A1_lam10": dict(penalty_lambda=10.0, n_bootstrap=10, **struct)}
    for B in (5, 20):
        vs[f"A2_B{B}"] = dict(penalty_lambda=1.0, n_bootstrap=B, **struct)
    for eps in (1e-3, 1e-6):
        vs[f"A2n_eps{eps:g}"] = dict(penalty_lambda=1.0, n_bootstrap=10,
                                     normalized_penalty=True,
                                     penalty_epsilon=eps, **struct)
    vs["A3n_lam0.1"] = dict(penalty_lambda=0.1, n_bootstrap=10,
                            normalized_penalty=True, **struct)
    vs["A3n_lam10"] = dict(penalty_lambda=10.0, n_bootstrap=10,
                           normalized_penalty=True, **struct)
    for d in (3, 8):
        for leaf in (1, 5):
            s2 = dict(struct)
            s2.update(max_depth=d, min_samples_leaf=leaf)
            vs[f"A4_d{d}_leaf{leaf}"] = dict(penalty_lambda=1.0,
                                             n_bootstrap=10, **s2)
    return vs


def main():
    cfg = {"cv": {"outer": {"seeds": [42, 123, 2026], "splits": 5,
                            "repeats": 3}, "inner": {"splits": 3}}}
    tuned = {(t["dataset"], t["seed"], t["repeat"], t["outer_fold"]): t
             for t in json.loads(
                 (ROOT / "logs/extension_tuned_params.json").read_text())}
    out = ROOT / "results" / "extension" / "ablation.csv"
    rows = []
    for name in ["iris", "wine", "seeds", "glass", "sonar", "ionosphere",
                 "wdbc", "banknote"]:
        X, y = data_mod.load_dataset(name)
        yv = np.asarray(y)
        Xv = X.values
        for seed, repeat, fold, tr, te in outer_splits(yv, cfg):
            t = tuned[(name, seed, repeat, fold)]
            struct = {k: t["struct"][k] for k in
                      ("max_depth", "min_samples_leaf", "min_samples_split")}
            imp = SimpleImputer(strategy="median").fit(Xv[tr])
            Xi = imp.transform(Xv)
            for vname, kw in variants(struct).items():
                clf = StabilityTreeClassifier(random_state=0, **kw)
                t0 = time.perf_counter()
                clf.fit(Xi[tr], yv[tr])
                fit_s = time.perf_counter() - t0
                pred = clf.predict(Xi[te])
                rows.append({"dataset": name, "variant": vname,
                             "seed": seed, "repeat": repeat,
                             "outer_fold": fold,
                             "f1_macro": f1_score(yv[te], pred,
                                                  average="macro"),
                             "balanced_accuracy":
                                 balanced_accuracy_score(yv[te], pred),
                             "accuracy": accuracy_score(yv[te], pred),
                             "fit_seconds": fit_s,
                             "node_count": clf.node_count_(),
                             "tree_depth": clf.tree_depth_()})
            print(f"[{name}] seed={seed} r={repeat} f={fold} ablation",
                  flush=True)
            tmp = out.with_suffix(".tmp")
            pd.DataFrame(rows).to_csv(tmp, index=False)
            tmp.rename(out)
    print("ablation rows:", len(rows))


if __name__ == "__main__":
    main()
