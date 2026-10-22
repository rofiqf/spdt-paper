"""Stage 8 figures for the extension experiments (Agg backend)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "results" / "extension"
FIG = ROOT / "figures"

LABELS = {"cart_custom": "CART-cust", "cart_sklearn": "CART-skl",
          "spdt": "SPDT", "spdt_norm": "SPDT-norm", "random_forest": "RF",
          "anls": "ANLS"}
CORE = ["cart_custom", "spdt", "spdt_norm"]


def fig_noise(b: pd.DataFrame, out=FIG):
    out.mkdir(exist_ok=True)
    g = (b.groupby(["noise_level", "method"])["f1_macro"]
         .mean().reset_index())
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ["cart_custom", "spdt", "spdt_norm", "cart_sklearn"]:
        d = g[g.method == m].sort_values("noise_level")
        ax.plot(d.noise_level * 100, d.f1_macro, marker="o",
                label=LABELS[m])
    ax.set_xlabel("Training label noise (%)")
    ax.set_ylabel("Macro F1 (mean over datasets)")
    ax.set_title("Macro F1 vs Label Noise Level")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "noise_robustness_macro_f1.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def fig_root_agreement(c: pd.DataFrame, out=FIG):
    g = (c.groupby("method")["root_feature_agreement"]
         .mean().reindex(["cart_custom", "cart_sklearn", "spdt",
                          "spdt_norm", "anls"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    g.plot(kind="bar", ax=ax, color="#4878cf")
    ax.set_ylabel("Mean pairwise root-feature agreement")
    ax.set_title("Root Split Stability Across Methods")
    ax.set_xticklabels([LABELS.get(i, i) for i in g.index], rotation=30,
                       ha="right")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out / "root_split_agreement.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def fig_pred_disagreement(c: pd.DataFrame, a: pd.DataFrame, out=FIG):
    ext = (c.groupby("method")["prediction_disagreement"].mean()
           .reindex(["cart_custom", "cart_sklearn", "spdt", "spdt_norm",
                     "anls"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ext.plot(kind="bar", ax=ax, color="#6acc64")
    ax.set_ylabel("Mean prediction disagreement")
    ax.set_title("Prediction Stability (Extension, Subsampled Models)")
    ax.set_xticklabels([LABELS.get(i, i) for i in ext.index], rotation=30,
                       ha="right")
    plt.tight_layout()
    plt.savefig(out / "prediction_disagreement.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def fig_runtime_tradeoff(rt: pd.DataFrame, out=FIG):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rt.itertuples():
        ax.scatter(r.fit_mean, r.f1_mean, s=90,
                   label=LABELS.get(r.method, r.method))
    ax.set_xscale("log")
    ax.set_xlabel("Mean fit time per outer fold (s, log scale)")
    ax.set_ylabel("Mean Macro F1")
    ax.set_title("Runtime vs Performance Trade-off")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out / "runtime_vs_macro_f1.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def main():
    b = pd.read_csv(EXT / "noise_robustness.csv")
    c = pd.read_csv(EXT / "split_stability.csv")
    rt = pd.read_csv(EXT / "runtime_tradeoff.csv")
    a = pd.read_csv(EXT / "normalized_penalty.csv")
    fig_noise(b)
    fig_root_agreement(c)
    fig_pred_disagreement(c, a)
    fig_runtime_tradeoff(rt)
    print("extension figures written")


if __name__ == "__main__":
    main()
