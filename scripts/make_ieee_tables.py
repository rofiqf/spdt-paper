"""Generate paper/tables_generated.tex dari artefak results/ (satu sumber
kebenaran angka). Semua tabel booktabs; caption di atas; label konsisten.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUM = ROOT / "results" / "summary"
OUT = ROOT / "paper" / "tables_generated.tex"

SHORT = {"cart_sklearn": "CART-skl", "cart_custom": "CART-cust",
         "spdt": "SPDT", "random_forest": "RF"}
ABL = {"A0_lam0": "A0 ($\\lambda$=0)", "A1_lam0.1_B10": "A1 ($\\lambda$=.1)",
       "A1_lam1_B10": "A1 ($\\lambda$=1)", "A2_B5": "A2 ($B$=5)",
       "A2_B20": "A2 ($B$=20)", "A3_d3_leaf1": "A3 d3 l1",
       "A3_d3_leaf5": "A3 d3 l5", "A3_d8_leaf1": "A3 d8 l1",
       "A3_d8_leaf5": "A3 d8 l5"}


def tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&")
             .replace("%", r"\%").replace("#", r"\#")
             .replace("_", r"\_"))


def f(v, nd=4):
    return "-" if pd.isna(v) else f"{v:.{nd}f}"


def main():
    folds = pd.read_csv(ROOT / "results/raw/fold_metrics.csv")
    agg = pd.read_csv(SUM / "dataset_metrics.csv")
    stats = pd.read_csv(SUM / "statistical_tests.csv").iloc[0]
    runtime = pd.read_csv(SUM / "runtime.csv")
    stab = pd.read_csv(SUM / "stability.csv")
    ranks = pd.read_csv(SUM / "mean_ranks.csv")
    abl = pd.read_csv(SUM / "ablation.csv")
    hyps = pd.read_csv(SUM / "hyperparameters.csv")

    blocks = []

    # ---- Table I: inventory -------------------------------------------
    inv = []
    for mp in sorted((ROOT / "data/metadata").glob("*.json")):
        m = json.loads(mp.read_text())
        inv.append((m["dataset"], int(m["n_samples"]), int(m["n_features"]),
                    int(m["n_classes"]), m["source"].replace("uci_static_zip",
                                                             "UCI zip")))
    rows = "\n".join(f"{tex_escape(a)} & {b} & {c} & {d} & {tex_escape(e)} \\\\"
                     for a, b, c, d, e in inv)
    blocks.append(
        r"""\begin{table}[!t]
\caption{Dataset Inventory}
\label{tab:inventory}
\centering
\begin{tabular}{@{}lrrrl@{}}
\toprule
Dataset & $N$ & Features & Classes & Source \\
\midrule
""" + rows + "\n" + r"""\bottomrule
\end{tabular}
\end{table}""")

    # ---- Table II: performance (wide) ----------------------------------
    pm = folds.groupby(["dataset", "method"])["f1_macro"].agg(
        ["mean", "std"]).unstack()
    order = ["cart_sklearn", "cart_custom", "spdt", "random_forest"]
    head = " & ".join(SHORT[m] for m in order) + r" & SPDT$-$CART \\"
    lines = []
    for ds in pm.index:
        cells = [f"{pm.loc[ds, ('mean', m)]:.3f} $\\pm$ {pm.loc[ds, ('std', m)]:.3f}"
                 for m in order]
        delta = pm.loc[ds, ('mean', 'spdt')] - pm.loc[ds, ('mean', 'cart_custom')]
        lines.append(f"{tex_escape(ds)} & " + " & ".join(cells) +
                     f" & {delta:+.4f} \\\\")
    blocks.append(
        r"""\begin{table*}[!t]
\caption{Macro-F1 Mean $\pm$ SD (Descriptive Fold-Level Scores) per Dataset and Method}
\label{tab:performance}
\centering
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccc@{}}
\toprule
Dataset & """ + head + r"""
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular*}
\end{table*}""")

    # ---- Table III: mean ranks ----------------------------------------
    rk = ranks.copy()
    rk["method"] = rk["method"].map(lambda m: SHORT.get(m, m))
    lines = "\n".join(f"{tex_escape(r.method)} & {f(r.mean_rank, 3)} \\\\"
                      for r in rk.itertuples())
    blocks.append(
        r"""\begin{table}[!t]
\caption{Mean Macro-F1 Rank Across Datasets}
\label{tab:ranks}
\centering
\begin{tabular}{@{}lr@{}}
\toprule
Method & Mean rank \\
\midrule
""" + lines + "\n" + r"""\bottomrule
\end{tabular}
\end{table}""")

    # ---- Table IV: Friedman -------------------------------------------
    decision = ("post-hoc not run"
                if "no_omnibus" in str(stats["decision"])
                else tex_escape(str(stats["decision"])))
    blocks.append(
        r"""\begin{table}[!t]
\caption{Dataset-Level Statistical Test (Friedman, $D{=}8$ Datasets, 3 Methods)}
\label{tab:friedman}
\centering
\begin{tabular}{@{}lr@{}}
\toprule
Statistic & Value \\
\midrule
$\chi^2$ & """ + f(stats["statistic"]) + r""" \\
Degrees of freedom & """ + str(int(stats["df"])) + r""" \\
$p$ & """ + f(stats["p_value"]) + r""" \\
Kendall's $W$ & """ + f(stats["kendall_w"]) + r""" \\
Decision & """ + decision + r""" \\
\bottomrule
\end{tabular}
\end{table}""")

    # ---- Table V: runtime ---------------------------------------------
    rt = runtime.copy()
    rt["method"] = rt["method"].map(lambda m: SHORT.get(m, m))
    lines = "\n".join(
        f"{tex_escape(r.method)} & {f(r.fit_mean)} & {f(r.fit_median)} & "
        f"{f(r.predict_mean)} & {f(r.nodes_mean)} & {f(r.depth_mean)} \\\\"
        for r in rt.itertuples())
    blocks.append(
        r"""\begin{table}[!t]
\caption{Computational Cost (per Outer Fold, Averaged Over Datasets)}
\label{tab:runtime}
\centering
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Method & Fit (s) & Fit median (s) & Predict (s) & Nodes & Depth \\
\midrule
""" + lines + "\n" + r"""\bottomrule
\end{tabular}
\end{table}""")

    # ---- Table VI: stability ------------------------------------------
    st = stab.copy()
    st["method"] = st["method"].map(lambda m: SHORT.get(m, m))
    lines = "\n".join(f"{tex_escape(r.method)} & {f(r.disagreement_mean)} & "
                      f"{f(r.disagreement_sd)} \\\\" for r in st.itertuples())
    blocks.append(
        r"""\begin{table}[!t]
\caption{Prediction Stability (Mean Pairwise Disagreement; Not an Accuracy Metric)}
\label{tab:stability}
\centering
\begin{tabular}{@{}lrr@{}}
\toprule
Method & Mean disagreement & SD \\
\midrule
""" + lines + "\n" + r"""\bottomrule
\end{tabular}
\end{table}""")

    # ---- Table VII: ablation ------------------------------------------
    ab = abl.copy()
    ab["variant"] = ab["variant"].map(lambda v: ABL.get(v, tex_escape(v)))
    lines = "\n".join(
        f"{r.variant} & {f(r.f1_mean)} & {f(r.f1_sd)} & {f(r.fit_mean)} & "
        f"{f(r.nodes_mean)} \\\\" for r in ab.itertuples())
    blocks.append(
        r"""\begin{table}[!t]
\caption{Ablation on Identical Outer Splits}
\label{tab:ablation}
\centering
\begin{tabular}{@{}lrrrr@{}}
\toprule
Variant & Macro F1 & SD & Fit (s) & Nodes \\
\midrule
""" + lines + "\n" + r"""\bottomrule
\end{tabular}
\end{table}""")

    # ---- Table VIII: hyperparameter frequencies (full) ----------------
    lines = "\n".join(
        f"{tex_escape(r.parameter)} & {tex_escape(str(r.value))} & "
        f"{int(r.frequency)} & {f(r.share)} \\\\" for r in hyps.itertuples())
    blocks.append(
        r"""\begin{table}[!t]
\caption{SPDT Hyperparameter Selection Frequency (45 Outer Folds per Dataset)}
\label{tab:hyper}
\centering
\begin{tabular}{@{}llrr@{}}
\toprule
Parameter & Value & Frequency & Share \\
\midrule
""" + lines + "\n" + r"""\bottomrule
\end{tabular}
\end{table}""")

    OUT.write_text("% AUTO-GENERATED from results/ by make_ieee_tables.py\n"
                   "% Do not hand-edit numbers; regenerate instead.\n\n"
                   + "\n\n".join(blocks) + "\n")
    print("written:", OUT, "| tables:", len(blocks))


if __name__ == "__main__":
    main()
