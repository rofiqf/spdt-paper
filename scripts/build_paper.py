"""Rakit paper (Bahasa Indonesia / English) dari artefak hasil.

Aturan integritas:
- Placeholder hasil HARUS tersedia di results/summary/; jika tidak ->
  RuntimeError (tidak boleh ada angka fabrikasi).
- Semua angka di-dump dari artefak; teks narasi statis ditulis hati-hati
  tanpa klaim hasil sebelum ada artefak.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUM = ROOT / "results" / "summary"
DATA = ROOT / "data"
PAPER = ROOT / "paper"


def need(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(
            f"Cannot build empirical paper: required result artifact "
            f"missing: {path}")
    return pd.read_csv(path)


def df_to_md(df: pd.DataFrame, floatfmt=4) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:.{floatfmt}f}")
    header = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.values]
    return "\n".join([header, sep, *rows])


L = {
    "id": {
        "ranks": {"method": "Metode", "mean_rank": "Mean rank"},
        "stats": {"test": "Uji", "statistic": "Statistik", "df": "db",
                  "p_value": "p", "n_datasets": "#dataset",
                  "kendall_w": "Kendall W", "decision": "Keputusan"},
        "runtime": {"method": "Metode", "fit_mean": "Fit rata-rata (s)",
                    "fit_median": "Fit median (s)",
                    "predict_mean": "Prediksi rata-rata (s)",
                    "nodes_mean": "Node rata-rata",
                    "depth_mean": "Kedalaman rata-rata"},
        "stab": {"method": "Metode",
                 "disagreement_mean": "Disagreement rata-rata",
                 "disagreement_sd": "SD"},
        "abl": {"variant": "Varian", "f1_mean": "F1 makro rata-rata",
                "f1_sd": "SD", "fit_mean": "Fit rata-rata (s)",
                "nodes_mean": "Node rata-rata"},
        "inv": {"dataset": "Dataset", "n_samples": "N",
                "n_features": "Fitur", "n_classes": "Kelas",
                "source": "Sumber"},
        "posthoc": {"comparison": "Bandingkan", "median_delta": "Median Delta F1",
                    "raw_p": "p mentah", "holm_p": "p Holm",
                    "reject_h0": "Tolak H0", "rank_biserial": "r_rb"},
        "no_posthoc": ("_Post-hoc tidak dijalankan: uji omnibus Friedman "
                       "tidak memberikan alasan (p >= 0.05), sesuai "
                       "protokol._"),
        "na": "_Tidak tersedia._",
        "decision": {
            "no_omnibus_evidence_posthoc_not_run":
                "tidak ada bukti omnibus; post-hoc tidak dijalankan",
            "omnibus_significant_posthoc_justified":
                "omnibus signifikan; post-hoc dibenarkan",
        },
        "template": "template.md",
        "out": "draft.md",
        "title": ("Stability-Penalized Split Selection for Decision Tree "
                  "Classification: Sebuah Evaluasi Multi-Dataset yang "
                  "Reproducible"),
    },
    "en": {
        "ranks": {"method": "Method", "mean_rank": "Mean rank"},
        "stats": {"test": "Test", "statistic": "Statistic", "df": "df",
                  "p_value": "p", "n_datasets": "#datasets",
                  "kendall_w": "Kendall W", "decision": "Decision"},
        "runtime": {"method": "Method", "fit_mean": "Fit mean (s)",
                    "fit_median": "Fit median (s)",
                    "predict_mean": "Predict mean (s)",
                    "nodes_mean": "Mean nodes",
                    "depth_mean": "Mean depth"},
        "stab": {"method": "Method",
                 "disagreement_mean": "Mean disagreement",
                 "disagreement_sd": "SD"},
        "abl": {"variant": "Variant", "f1_mean": "Mean macro F1",
                "f1_sd": "SD", "fit_mean": "Fit mean (s)",
                "nodes_mean": "Mean nodes"},
        "inv": {"dataset": "Dataset", "n_samples": "N",
                "n_features": "Features", "n_classes": "Classes",
                "source": "Source"},
        "posthoc": {"comparison": "Comparison", "median_delta": "Median Delta F1",
                    "raw_p": "Raw p", "holm_p": "Holm p",
                    "reject_h0": "Reject H0", "rank_biserial": "r_rb"},
        "no_posthoc": ("_Post-hoc not run: the Friedman omnibus test gave "
                       "no grounds (p >= 0.05), per protocol._"),
        "na": "_Not available._",
        "decision": {
            "no_omnibus_evidence_posthoc_not_run":
                "no omnibus evidence; post-hoc not run",
            "omnibus_significant_posthoc_justified":
                "omnibus significant; post-hoc justified",
        },
        "template": "template_en.md",
        "out": "draft_en.md",
        "title": ("Stability-Penalized Split Selection for Decision Tree "
                  "Classification: A Reproducible Multi-Dataset Evaluation"),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["id", "en"], default="id")
    args = ap.parse_args()
    T = L[args.lang]

    perf = need(SUM / "dataset_metrics.csv")
    stats = need(SUM / "statistical_tests.csv")
    runtime = need(SUM / "runtime.csv")
    stab = need(SUM / "stability.csv") if (SUM / "stability.csv").exists() \
        else None
    abl = need(SUM / "ablation.csv") if (SUM / "ablation.csv").exists() \
        else None
    ranks = need(SUM / "mean_ranks.csv")
    hyps = need(SUM / "hyperparameters.csv")
    paired = need(SUM / "paired_differences.csv")

    fr = stats.iloc[0]
    stats = stats.assign(decision=T["decision"].get(
        str(fr["decision"]), str(fr["decision"])))
    posthoc_path = SUM / "posthoc_wilcoxon_holm.csv"
    posthoc = (pd.read_csv(posthoc_path) if posthoc_path.exists() else None)

    # tabel performa: mean +/- sd (deskriptif) per dataset x metode
    folds = pd.read_csv(ROOT / "results" / "raw" / "fold_metrics.csv")
    pm = folds.groupby(["dataset", "method"])["f1_macro"].agg(
        ["mean", "std"]).reset_index()
    pivot_m = pm.pivot(index="dataset", columns="method", values="mean")
    pivot_s = pm.pivot(index="dataset", columns="method", values="std")
    methods = [c for c in ("cart_sklearn", "cart_custom", "spdt",
                           "random_forest") if c in pivot_m.columns]
    perf_rows = []
    for ds in pivot_m.index:
        row = {"Dataset": ds}
        for m in methods:
            row[m] = f"{pivot_m.loc[ds, m]:.3f} +/- {pivot_s.loc[ds, m]:.3f}"
        row["Delta (spdt-cart_custom)"] = (
            f"{pivot_m.loc[ds, 'spdt'] - pivot_m.loc[ds, 'cart_custom']:+.4f}")
        perf_rows.append(row)
    tbl_perf = df_to_md(pd.DataFrame(perf_rows), floatfmt=3)

    tbl_ranks = df_to_md(ranks.rename(columns=T["ranks"]))
    tbl_stat = df_to_md(stats.rename(columns=T["stats"]))
    rt = runtime.rename(columns=T["runtime"])
    tbl_rt = df_to_md(rt[[c for c in rt.columns if c != "fit_median"]])
    tbl_stab = (df_to_md(stab.rename(columns=T["stab"]))
                if stab is not None else T["na"])
    tbl_abl = (df_to_md(abl.rename(columns=T["abl"]))
               if abl is not None else T["na"])

    inv_rows = []
    for meta_p in sorted((DATA / "metadata").glob("*.json")):
        m = json.loads(meta_p.read_text())
        inv_rows.append({"dataset": m["dataset"], "n_samples": m["n_samples"],
                         "n_features": m["n_features"],
                         "n_classes": m["n_classes"], "source": m["source"]})
    tbl_inv = df_to_md(pd.DataFrame(inv_rows).rename(columns=T["inv"]))
    tbl_hyp = df_to_md(hyps)

    pdelta = paired[(paired.method_a == "cart_custom") &
                    (paired.method_b == "spdt")].copy()
    pdelta["delta"] = -pdelta["delta"]
    wins = int((pdelta["delta"] > 0).sum())
    ties = int((pdelta["delta"] == 0).sum())
    losses = int((pdelta["delta"] < 0).sum())
    mean_delta = float(pdelta["delta"].mean())

    figs = {p.name: p for p in (ROOT / "figures").glob("*.png")}
    for req in ("macro_f1_by_dataset.png", "paired_delta_f1.png",
                "runtime_tradeoff.png", "stability_comparison.png",
                "method_flowchart.png"):
        if req not in figs:
            raise RuntimeError(
                f"Cannot build empirical paper: figure missing: {req}")

    posthoc_md = (df_to_md(posthoc.rename(columns=T["posthoc"]))
                  if posthoc is not None else T["no_posthoc"])

    template = (PAPER / T["template"]).read_text()
    out = (template
           .replace("{{TABLE_INVENTORY}}", tbl_inv)
           .replace("{{TABLE_PERFORMANCE}}", tbl_perf)
           .replace("{{TABLE_MEAN_RANK}}", tbl_ranks)
           .replace("{{TABLE_STATISTICS}}", tbl_stat)
           .replace("{{TABLE_POSTHOC}}", posthoc_md)
           .replace("{{TABLE_RUNTIME}}", tbl_rt)
           .replace("{{TABLE_STABILITY}}", tbl_stab)
           .replace("{{TABLE_ABLATION}}", tbl_abl)
           .replace("{{TABLE_HYPERPARAMS}}", tbl_hyp)
           .replace("{{N_DATASETS}}", str(fr["n_datasets"]))
           .replace("{{WINS}}", str(wins))
           .replace("{{TIES}}", str(ties))
           .replace("{{LOSSES}}", str(losses))
           .replace("{{MEAN_DELTA}}", f"{mean_delta:+.4f}")
           .replace("{{FRIEDMAN_STAT}}", f"{fr['statistic']:.4f}")
           .replace("{{FRIEDMAN_P}}", f"{fr['p_value']:.4f}")
           .replace("{{KENDALL_W}}", f"{fr['kendall_w']:.4f}"))
    # sanitasi unicode -> ASCII/latex aman untuk pdflatex (IEEEtran)
    for src, dst in {
        "\u2212": "-",   # minus unicode
        "\u00b7": "*",   # titik tengah
        "\u03bb": "lambda",
        "\u03c7": "chi",
        "\u03a3": "Sum",
        "\u03b5": "eps",
        "\u0394": "Delta",
        "\u2013": "-",
        "\u2014": "--",
        "\u00d7": "x",   # tanda kali
        "\u2265": ">=",
        "\u2264": "<=",
        "\u00b1": "+/-",
        "\u00b2": "^2",
    }.items():
        out = out.replace(src, dst)
    yaml_fm = (
        "---\n"
        f'title: "{T["title"]}"\n'
        'author: "Rofiq Fauzi (p36202600047@mhs.dinus.ac.id), Magister '
        'Artificial Intelligence, Universitas Dian Nuswantoro"\n'
        "---\n\n")
    (PAPER / T["out"]).write_text(yaml_fm + out)
    print("paper written to", PAPER / T["out"])


if __name__ == "__main__":
    main()
