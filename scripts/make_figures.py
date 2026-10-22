"""Stage 7c: generate semua figure wajib spec ke figures/."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import plots

FIG = ROOT / "figures"
SUM = ROOT / "results" / "summary"


def main():
    agg = pd.read_csv(SUM / "dataset_metrics.csv")
    plots.fig_macro_f1_by_dataset(agg, FIG)
    plots.fig_paired_delta_f1(agg, focus="spdt", reference="cart_custom",
                              out=FIG)
    plots.fig_runtime_tradeoff(agg, FIG)
    stab_path = SUM / "stability.csv"
    if (ROOT / "results" / "raw" / "stability_metrics.csv").exists():
        plots.fig_stability_comparison(
            pd.read_csv(ROOT / "results" / "raw" / "stability_metrics.csv"),
            FIG)
    plots.fig_method_flowchart(FIG)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
