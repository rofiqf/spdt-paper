"""Figure generation (PNG, matplotlib Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def fig_macro_f1_by_dataset(agg: pd.DataFrame, out=FIG_DIR):
    out.mkdir(parents=True, exist_ok=True)
    wide = agg.pivot(index="dataset", columns="method", values="f1_macro")
    ax = wide.plot(kind="bar", figsize=(11, 6))
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Macro F1 (rata-rata outer CV)")
    ax.set_title("Macro F1 per Dataset dan Metode")
    ax.legend(title="Metode")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out / "macro_f1_by_dataset.png", bbox_inches="tight", dpi=150)
    plt.close()


def fig_paired_delta_f1(agg: pd.DataFrame, focus="spdt",
                        reference="cart_custom", out=FIG_DIR):
    out.mkdir(parents=True, exist_ok=True)
    wide = agg.pivot(index="dataset", columns="method", values="f1_macro")
    delta = (wide[focus] - wide[reference]).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    delta.plot(kind="barh", ax=ax)
    ax.axvline(0.0)
    ax.set_xlabel(f"Δ Macro F1: {focus} − {reference}")
    ax.set_ylabel("Dataset")
    ax.set_title(f"Efek Penalti Stabilitas per Dataset (vs {reference})")
    plt.tight_layout()
    plt.savefig(out / "paired_delta_f1.png", bbox_inches="tight", dpi=150)
    plt.close()


def fig_runtime_tradeoff(agg: pd.DataFrame, out=FIG_DIR):
    out.mkdir(parents=True, exist_ok=True)
    sub = agg.dropna(subset=["fit_seconds"])
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=sub, x="method", y="fit_seconds", ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Metode")
    ax.set_ylabel("Waktu fit (detik, log)")
    ax.set_title("Trade-off Waktu Komputasi")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out / "runtime_tradeoff.png", bbox_inches="tight", dpi=150)
    plt.close()


def fig_stability_comparison(stab_df: pd.DataFrame, out=FIG_DIR):
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=stab_df, x="method", y="disagreement", ax=ax)
    ax.set_xlabel("Metode")
    ax.set_ylabel("Prediction disagreement (mean pairwise)")
    ax.set_title("Stabilitas Prediksi antar Model ter-Resample")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out / "stability_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()


def fig_method_flowchart(out=FIG_DIR):
    """Diagram alur metode (tanpa external graphviz)."""
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    steps = [
        "Data node D (hanya training)",
        "Kandidat split (feature, threshold)\nmidpoint nilai unik terurut",
        "Gain Gini asli Gain(s, D)",
        f"B bootstrap resample D^(b)\nGain(s, D^(b)) per kandidat",
        "V(s) = Var_b[Gain(s, D^(b))]",
        "Score(s) = Gain(s, D) − λ·V(s)",
        "Pilih split valid ber-Score tertinggi\ndengan Gain asli > 0",
        "Rekursi / daun (majority, tie deterministik)",
    ]
    for i, s in enumerate(steps):
        yy = 1 - i / (len(steps) - 1)
        ax.text(0.5, yy, s, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4",
                          fc="#eef3fb" if i not in (4, 5) else "#fdeeca",
                          ec="#557"),
                fontsize=10, transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, 1 - (i + 1) / (len(steps) - 1) + 0.045),
                        xytext=(0.5, yy - 0.045),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#557"))
    plt.tight_layout()
    plt.savefig(out / "method_flowchart.png", bbox_inches="tight", dpi=150)
    plt.close()
