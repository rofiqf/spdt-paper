"""F3/E-2: depth-matched CART control.

For each stored fold, fit custom CART (lambda=0, default leaf/split) with
max_depth equal to the structure tuned for SPDT, evaluate on the same
outer test fold, and write paired comparisons vs SPDT and SPDT-norm
(from Experiment A). Tests whether the penalized trees' behavior is
explained by implicit depth restriction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data as data_mod
from src.evaluate import outer_splits
from src.tree import StabilityTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

OUTDIR = ROOT / "results" / "extension_e2"


def main():
    cfg = {"cv": {"outer": {"seeds": [42, 123, 2026], "splits": 5,
                            "repeats": 3}, "inner": {"splits": 3}}}
    tuned = {(t["dataset"], t["seed"], t["repeat"], t["outer_fold"]): t
             for t in json.loads(
                 (ROOT / "logs/extension_tuned_params.json").read_text())}
    a = pd.read_csv(ROOT / "results/extension/normalized_penalty.csv")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ["iris", "wine", "seeds", "glass", "sonar", "ionosphere",
                 "wdbc", "banknote"]:
        X, y = data_mod.load_dataset(name)
        yv = np.asarray(y)
        Xv = X.values
        for seed, repeat, fold, tr, te in outer_splits(yv, cfg):
            t = tuned[(name, seed, repeat, fold)]
            depth = t["struct"]["max_depth"]
            imp = SimpleImputer(strategy="median").fit(Xv[tr])
            Xi = imp.transform(Xv)
            clf = StabilityTreeClassifier(
                penalty_lambda=0.0, max_depth=depth, random_state=0)
            clf.fit(Xi[tr], yv[tr])
            pred = clf.predict(Xi[te])
            row_a = a[(a.dataset == name) & (a.seed == seed)
                      & (a.repeat == repeat) & (a.outer_fold == fold)]
            f1_spdt = row_a[row_a.method == "spdt"].f1_macro.iloc[0]
            f1_norm = row_a[row_a.method == "spdt_norm"].f1_macro.iloc[0]
            rows.append({"dataset": name, "seed": seed, "repeat": repeat,
                         "outer_fold": fold, "matched_depth": str(depth),
                         "f1_depth_matched":
                             f1_score(yv[te], pred, average="macro"),
                         "f1_spdt": f1_spdt, "f1_spdt_norm": f1_norm})
        print(f"[{name}] depth-matched done", flush=True)
    df = pd.DataFrame(rows)
    df["d_vs_spdt"] = df.f1_depth_matched - df.f1_spdt
    df["d_vs_norm"] = df.f1_depth_matched - df.f1_spdt_norm
    out = OUTDIR / "depth_matched.csv"
    df.to_csv(out, index=False)
    ds = df.groupby("dataset")[["d_vs_spdt", "d_vs_norm"]].mean()
    print(ds.round(4).to_string())
    print("mean d_vs_spdt=%+.4f  d_vs_norm=%+.4f"
          % (df.d_vs_spdt.mean(), df.d_vs_norm.mean()))
    print("written:", out)


if __name__ == "__main__":
    main()
