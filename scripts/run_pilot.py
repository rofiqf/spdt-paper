"""Stage 4: pilot terbatas (Iris, Wine, WDBC).

Tujuan: verifikasi kebenatan pipeline end-to-end, pengukuran runtime aktual
per kombinasi (lambda, B), dan DATA untuk menyempitkan grid sebelum freeze
protokol (Stage 5). BUKAN klaim temuan final.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data as data_mod
from src.evaluate import build_model_specs, run_dataset
from src.tree import StabilityTreeClassifier


def bench_fit(X, y, lam, B, depth=None, n=5):
    """Median waktu fit beberapa ulangan."""
    times = []
    for _ in range(n):
        clf = StabilityTreeClassifier(penalty_lambda=lam, n_bootstrap=B,
                                      max_depth=depth, random_state=0)
        t0 = time.perf_counter()
        clf.fit(X, y)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    config = {"cv": {"outer": {"seeds": [42], "splits": 3, "repeats": 1},
                     "inner": {"splits": 2}},
              "models": {
                  "cart_custom": {"grid": {"max_depth": [3, 5]},
                                  "fixed": {"penalty_lambda": 0.0}},
                  "spdt": {"grid": {"penalty_lambda": [0.01, 0.1, 1.0],
                                    "n_bootstrap": [10],
                                    "max_depth": [3, 5]},
                           "fixed": {}},
              }}
    datasets = ["iris", "wine", "wdbc"]
    out = ROOT / "results" / "summary"
    out.mkdir(parents=True, exist_ok=True)

    # --- benchmark runtime mentah per (dataset, lambda, B) ----------------
    bench_rows = []
    for name in datasets:
        X, y = data_mod.load_dataset(name)
        yv = np.asarray(y)
        for lam in (0.0, 0.01, 0.1, 1.0):
            for B in (5, 10, 20):
                t = bench_fit(X.values, yv, lam, B)
                bench_rows.append({"dataset": name, "lambda": lam, "B": B,
                                   "fit_median_seconds": t})
                print(f"bench {name} lam={lam} B={B}: {t:.3f}s", flush=True)
    bench_df = pd.DataFrame(bench_rows)
    bench_df.to_csv(out / "pilot_runtime.csv", index=False)

    # --- pilot nested CV mini --------------------------------------------
    specs = build_model_specs(config)
    all_rows, all_stab = [], []
    for name in datasets:
        X, y = data_mod.load_dataset(name)
        rows, stab = run_dataset(name, X, y, config, specs,
                                 ROOT / "results" / "raw" / "checkpoints",
                                 stability_resamples=5, verbose=True)
        all_rows += rows
        all_stab += stab
    pd.DataFrame(all_rows).to_csv(out / "pilot_metrics.csv", index=False)
    pd.DataFrame(all_stab).to_csv(out / "pilot_stability.csv", index=False)

    summary = {
        "pilot_datasets": datasets,
        "bench_max_seconds": float(bench_df["fit_median_seconds"].max()),
        "bench_by_lam_B": bench_df.groupby(["lambda", "B"])[
            "fit_median_seconds"].median().reset_index().to_dict("records"),
    }
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
