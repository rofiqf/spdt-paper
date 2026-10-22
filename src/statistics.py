"""Dataset-level statistical analysis.

Unit of inference is the DATASET (one aggregate per dataset per method),
never the individual CV fold. Fold rows are descriptive only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


def aggregate_dataset_level(fold_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, method): mean of outer-fold metrics."""
    metric_cols = [c for c in fold_df.columns if c not in
                   ("dataset", "method", "seed", "repeat", "outer_fold",
                    "n_train", "n_test", "inner_seed", "best_params")]
    agg = (fold_df.groupby(["dataset", "method"], as_index=False)
           [metric_cols].mean())
    return agg


def mean_ranks(agg: pd.DataFrame, metric="f1_macro") -> pd.DataFrame:
    wide = agg.pivot(index="dataset", columns="method", values=metric)
    ranks = wide.rank(ascending=False, axis=1)  # best -> rank 1
    out = ranks.mean().rename("mean_rank").reset_index().rename(
        columns={"index": "method"})
    return out.sort_values("mean_rank"), wide


def friedman_test(wide: pd.DataFrame, metric="f1_macro"):
    methods = list(wide.columns)
    stat, p = friedmanchisquare(*[wide[m].values for m in methods])
    D, M = wide.shape
    kendall_w = stat / (D * (M - 1)) if D * (M - 1) > 0 else np.nan
    return {"statistic": stat, "p_value": p, "df": M - 1,
            "n_datasets": D, "n_methods": M, "kendall_w": kendall_w,
            "methods": methods}


def _rank_biserial(diffs: np.ndarray) -> float:
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0
    ranks = np.abs(nz).argsort().argsort().astype(float) + 1.0
    w_pos = ranks[nz > 0].sum()
    w_neg = ranks[nz < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def pairwise_wilcoxon_holm(wide: pd.DataFrame, focus: str) -> pd.DataFrame:
    """Paired Wilcoxon of focus vs every other method + Holm correction."""
    others = [m for m in wide.columns if m != focus]
    records = []
    for m in others:
        d = (wide[focus] - wide[m]).values
        if np.all(d == 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(wide[focus].values, wide[m].values,
                               alternative="two-sided",
                               zero_method="wilcox", method="auto")
        records.append({"comparison": f"{focus} vs {m}",
                        "median_delta": float(np.median(d)),
                        "raw_p": float(p)})
    res = pd.DataFrame(records)
    reject, p_holm, _, _ = multipletests(res["raw_p"].values, alpha=0.05,
                                         method="holm")
    res["holm_p"] = p_holm
    res["reject_h0"] = reject
    res["rank_biserial"] = [
        _rank_biserial((wide[focus] - wide[m]).values) for m in others]
    return res


def paired_differences(agg: pd.DataFrame, metric="f1_macro") -> pd.DataFrame:
    wide = agg.pivot(index="dataset", columns="method", values=metric)
    rows = []
    methods = list(wide.columns)
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            d = (wide[a] - wide[b])
            rows.append({"method_a": a, "method_b": b,
                         "dataset": wide.index,
                         "delta": d.values})
    out = []
    for r in rows:
        for ds, dv in zip(r["dataset"], r["delta"]):
            out.append({"method_a": r["method_a"], "method_b": r["method_b"],
                        "dataset": ds, "delta": dv})
    return pd.DataFrame(out)
