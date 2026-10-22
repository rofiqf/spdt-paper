"""Generate paper/tables_extension.tex dari results/extension/*.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "results" / "extension"
OUT = ROOT / "paper" / "tables_extension.tex"

SHORT = {"cart_sklearn": "CART-skl", "cart_custom": "CART-cust",
         "spdt": "SPDT", "spdt_norm": "SPDT-norm", "random_forest": "RF",
         "anls": "ANLS"}


def f(v, nd=4):
    return "-" if pd.isna(v) else f"{v:.{nd}f}"


def main():
    a = pd.read_csv(EXT / "normalized_penalty.csv")
    b = pd.read_csv(EXT / "noise_robustness.csv")
    c = pd.read_csv(EXT / "split_stability.csv")
    rt = pd.read_csv(EXT / "runtime_tradeoff.csv")
    ab = pd.read_csv(EXT / "ablation.csv")
    blocks = []

    # T2: per-dataset clean performance (wide)
    w = a.groupby(["dataset", "method"])["f1_macro"].mean().unstack()
    order = ["cart_custom", "spdt", "spdt_norm", "anls", "cart_sklearn",
             "random_forest"]
    lines = []
    for ds in w.index:
        cells = " & ".join(f(w.loc[ds, m], 4) for m in order)
        lines.append(f"{ds} & {cells} \\\\")
    blocks.append(
        r"""\begin{table*}[!t]
\caption{Macro-F1 (Mean Over 45 Outer Folds) per Dataset, Extension Experiment A}
\label{tab:extperf}
\centering
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
Dataset & """ + " & ".join(SHORT[m] for m in order) + r""" \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular*}
\end{table*}""")

    # T3: noise levels (4 methods, mean macro F1)
    g = (b.groupby(["noise_level", "method"])["f1_macro"].mean().unstack())
    lines = []
    for lev in sorted(g.index):
        cells = " & ".join(f(g.loc[lev, m], 4) for m in
                           ["cart_custom", "spdt", "spdt_norm",
                            "random_forest"])
        lines.append(f"{int(lev*100)}\\% & {cells} \\\\")
    blocks.append(
        r"""\begin{table}[!t]
\caption{Mean Macro-F1 by Training Label Noise Level}
\label{tab:noise}
\centering
\begin{tabular}{@{}lrrrr@{}}
\toprule
Noise & CART-cust & SPDT & SPDT-norm & RF \\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}""")

    # T4: split stability
    cs = (c.groupby("method")[["root_feature_agreement",
                               "root_threshold_variation",
                               "prediction_disagreement"]].mean()
          .reindex(["cart_custom", "cart_sklearn", "spdt", "spdt_norm",
                  "anls"]))
    lines = "\n".join(
        f"{SHORT.get(r.Index, r.Index)} & {f(r.root_feature_agreement)} & "
        f"{f(r.root_threshold_variation)} & "
        f"{f(r.prediction_disagreement)} \\\\" for r in cs.itertuples())
    blocks.append(
        r"""\begin{table}[!t]
\caption{Split and Prediction Stability (Mean Over Folds and Datasets)}
\label{tab:stability}
\centering
\begin{tabular}{@{}lrrr@{}}
\toprule
Method & Root-feature agr. & Threshold var. & Prediction dis. \\
\midrule
""" + lines + "\n" + r"""\bottomrule
\end{tabular}
\end{table}""")

    # T5: ablation
    ga = ab.groupby("variant")["f1_macro"].agg(["mean", "std"]).sort_values(
        "mean", ascending=False)
    lines = "\n".join(f"{k.replace('_', r'\_')} & {f(r['mean'])} & {f(r['std'])} \\\\"
                      for k, r in ga.iterrows())
    blocks.append(
        r"""\begin{table}[!t]
\caption{Ablation on Identical Splits (Frozen Structure, Mean Macro-F1)}
\label{tab:ablation}
\centering
\begin{tabular}{@{}lrr@{}}
\toprule
Variant & Macro F1 & SD \\
\midrule
""" + lines + "\n" + r"""\bottomrule
\end{tabular}
\end{table}""")

    # T6: runtime + counters
    lines = "\n".join(
        f"{SHORT.get(r.method, r.method)} & {f(r.fit_mean)} & "
        f"{f(r.predict_mean)} & {f(r.candidates_mean, 0)} & "
        f"{f(r.boot_gain_evals_mean, 0)} \\\\" for r in rt.itertuples())
    blocks.append(
        r"""\begin{table}[!t]
\caption{Computational Cost per Outer Fold (Refit of Tuned Configuration)}
\label{tab:runtime}
\centering
\begin{tabular}{@{}lrrrr@{}}
\toprule
Method & Fit (s) & Predict (s) & Candidates & Boot. gain evals \\
\midrule
""" + lines + "\n" + r"""\bottomrule
\end{tabular}
\end{table}""")

    OUT.write_text("% AUTO-GENERATED from results/extension/ -- do not edit\n\n"
                   + "\n\n".join(blocks) + "\n")
    print("written:", OUT, "| tables:", len(blocks))


if __name__ == "__main__":
    main()
