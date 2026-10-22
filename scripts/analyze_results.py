"""Stage 7b: agregasi hasil -> results/summary/ (semua file wajib spec).

Unit inferensi = dataset (rata-rata 45 outer fold), fold = deskriptif.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.statistics import (aggregate_dataset_level, friedman_test,
                            mean_ranks, paired_differences,
                            pairwise_wilcoxon_holm)

RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "summary"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(RAW / "fold_metrics.csv")
    agg = aggregate_dataset_level(folds)
    agg.to_csv(OUT / "dataset_metrics.csv", index=False)

    # paired differences (semua pasangan, per dataset)
    paired_differences(agg).to_csv(OUT / "paired_differences.csv",
                                   index=False)

    # mean ranks + Friedman pada metode primer (tanpa random_forest)
    primary = ["cart_sklearn", "cart_custom", "spdt"]
    avail = [m for m in primary if m in agg["method"].unique()]
    if len(avail) >= 3:
        wide = agg[agg["method"].isin(avail)].pivot(
            index="dataset", columns="method", values="f1_macro")
        ranks_tbl, wide2 = mean_ranks(agg[agg["method"].isin(avail)])
        ranks_tbl.to_csv(OUT / "mean_ranks.csv", index=False)
        fr = friedman_test(wide)
        rec = {"test": "friedman", "statistic": fr["statistic"],
               "df": fr["df"], "p_value": fr["p_value"],
               "n_datasets": fr["n_datasets"], "kendall_w": fr["kendall_w"],
               "decision": ("omnibus_significant_posthoc_justified"
                            if fr["p_value"] < 0.05 else
                            "no_omnibus_evidence_posthoc_not_run")}
        pd.DataFrame([rec]).to_csv(OUT / "statistical_tests.csv",
                                   index=False)
        if fr["p_value"] < 0.05:
            pw = pairwise_wilcoxon_holm(wide, focus="spdt")
            pw.to_csv(OUT / "posthoc_wilcoxon_holm.csv", index=False)

    # runtime
    folds.groupby("method", as_index=False).agg(
        fit_mean=("fit_seconds", "mean"),
        fit_median=("fit_seconds", "median"),
        predict_mean=("predict_seconds", "mean"),
        nodes_mean=("node_count", "mean"),
        depth_mean=("tree_depth", "mean"),
    ).to_csv(OUT / "runtime.csv", index=False)

    # stability
    stab_path = RAW / "stability_metrics.csv"
    if stab_path.exists():
        stab = pd.read_csv(stab_path)
        stab.groupby("method", as_index=False).agg(
            disagreement_mean=("disagreement", "mean"),
            disagreement_sd=("disagreement", "std"),
        ).to_csv(OUT / "stability.csv", index=False)
        stab.groupby(["dataset", "method"], as_index=False)[
            "disagreement"].mean().to_csv(
            OUT / "stability_by_dataset.csv", index=False)

    # frekuensi seleksi hyperparameter spdt
    spdt = folds[folds["method"] == "spdt"]
    if len(spdt):
        params = pd.json_normalize(spdt["best_params"].map(json.loads))
        freq_rows = []
        for col in params.columns:
            for val, cnt in params[col].value_counts().items():
                freq_rows.append({"parameter": col, "value": str(val),
                                  "frequency": int(cnt),
                                  "share": cnt / len(params)})
        pd.DataFrame(freq_rows).to_csv(OUT / "hyperparameters.csv",
                                       index=False)

    # ablation summary
    abl_path = RAW / "ablation_folds.csv"
    if abl_path.exists():
        abl = pd.read_csv(abl_path)
        abl.groupby("variant", as_index=False).agg(
            f1_mean=("f1_macro", "mean"),
            f1_sd=("f1_macro", "std"),
            fit_mean=("fit_seconds", "mean"),
            nodes_mean=("node_count", "mean"),
        ).to_csv(OUT / "ablation.csv", index=False)
        abl.groupby(["dataset", "variant"], as_index=False)[
            "f1_macro"].mean().to_csv(OUT / "ablation_by_dataset.csv",
                                      index=False)

    print("summary written to", OUT)


if __name__ == "__main__":
    main()
