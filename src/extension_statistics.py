"""Statistics for the extension experiments. Unit of inference = dataset.

Exploratory vs confirmatory: the omnibus Friedman across the four primary
methods is the sole confirmatory path; everything else (per-level noise
comparisons, stability deltas) is descriptive/exploratory and labeled so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

PRIMARY = ["cart_sklearn", "cart_custom", "spdt", "spdt_norm"]


def dataset_level(df: pd.DataFrame, metric="f1_macro") -> pd.DataFrame:
    return (df.groupby(["dataset", "method"], as_index=False)[metric]
            .mean().pivot(index="dataset", columns="method", values=metric))


def rank_biserial(diffs: np.ndarray) -> float:
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0
    r = np.abs(nz).argsort().argsort().astype(float) + 1
    wp, wn = r[nz > 0].sum(), r[nz < 0].sum()
    return float((wp - wn) / (wp + wn))


def boot_ci(diffs: np.ndarray, n=10000, seed=7):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    if len(diffs) < 2:
        return (np.nan, np.nan)
    bs = rng.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def friedman_with_posthoc(wide: pd.DataFrame, focus="spdt_norm") -> dict:
    cols = [c for c in PRIMARY if c in wide.columns]
    res = {}
    if len(cols) >= 3 and len(wide) >= 3:
        stat, p = friedmanchisquare(*[wide[c].values for c in cols])
        res["friedman"] = {"statistic": stat, "p": p, "df": len(cols) - 1,
                           "n_datasets": len(wide),
                           "kendall_w": stat / (len(wide) * (len(cols) - 1))}
        if p < 0.05:
            others = [c for c in cols if c != focus]
            recs = []
            for o in others:
                d = (wide[focus] - wide[o]).values
                if np.all(d == 0):
                    st, pv = 0.0, 1.0
                else:
                    st, pv = wilcoxon(wide[focus].values, wide[o].values,
                                      alternative="two-sided",
                                      zero_method="wilcox", method="auto")
                recs.append({"comparison": f"{focus} vs {o}",
                             "raw_p": pv, "median_delta": float(np.median(d)),
                             "rank_biserial": rank_biserial(d)})
            pr = pd.DataFrame(recs)
            rej, holm, _, _ = multipletests(pr["raw_p"], alpha=0.05,
                                            method="holm")
            pr["holm_p"], pr["reject"] = holm, rej
            res["posthoc"] = pr
    return res


def summarize_extension(a: pd.DataFrame, b: pd.DataFrame,
                        c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wide = dataset_level(a)
    fr = friedman_with_posthoc(wide)
    if "friedman" in fr:
        f = fr["friedman"]
        rows.append({"analysis": "A_omnibus_friedman", "subset": "clean",
                     "statistic": f["statistic"], "df": f["df"],
                     "p": f["p"], "holm_p": np.nan, "effect": f["kendall_w"],
                     "ci_low": np.nan, "ci_high": np.nan,
                     "n_datasets": f["n_datasets"],
                     "decision": ("omnibus_significant" if f["p"] < 0.05
                                  else "no_omnibus_evidence")})
    for cmp_name, ref in [("norm_vs_cart_custom", "cart_custom"),
                          ("norm_vs_spdt", "spdt")]:
        d = (wide["spdt_norm"] - wide[ref]).values
        lo, hi = boot_ci(d)
        rows.append({"analysis": "A_paired_descriptive", "subset": cmp_name,
                     "statistic": np.median(d), "df": np.nan,
                     "p": np.nan, "holm_p": np.nan,
                     "effect": rank_biserial(d), "ci_low": lo, "ci_high": hi,
                     "n_datasets": len(d), "decision": "exploratory"})
    if "posthoc" in fr:
        for r in fr["posthoc"].itertuples():
            rows.append({"analysis": "A_posthoc_wilcoxon",
                         "subset": r.comparison, "statistic": np.nan,
                         "df": np.nan, "p": r.raw_p, "holm_p": r.holm_p,
                         "effect": r.rank_biserial, "ci_low": np.nan,
                         "ci_high": np.nan,
                         "n_datasets": len(wide),
                         "decision": ("reject" if r.reject
                                      else "not_rejected")})
    # B: per noise level
    for lev in sorted(b["noise_level"].unique()):
        sub = b[b.noise_level == lev]
        wl = dataset_level(sub)
        frl = friedman_with_posthoc(wl)
        if "friedman" in frl:
            ff = frl["friedman"]
            rows.append({"analysis": "B_friedman_per_level",
                         "subset": f"noise={lev}", "statistic": ff["statistic"],
                         "df": ff["df"], "p": ff["p"], "holm_p": np.nan,
                         "effect": ff["kendall_w"], "ci_low": np.nan,
                         "ci_high": np.nan, "n_datasets": ff["n_datasets"],
                         "decision": ("omnibus_significant" if ff["p"] < 0.05
                                      else "no_omnibus_evidence")})
        for ref in ("cart_custom", "spdt"):
            if ref in wl.columns and "spdt_norm" in wl.columns:
                d = (wl["spdt_norm"] - wl[ref]).values
                lo, hi = boot_ci(d)
                rows.append({"analysis": "B_paired_descriptive",
                             "subset": f"noise={lev} norm_vs_{ref}",
                             "statistic": np.median(d), "df": np.nan,
                             "p": np.nan, "holm_p": np.nan,
                             "effect": rank_biserial(d), "ci_low": lo,
                             "ci_high": hi, "n_datasets": len(d),
                             "decision": "exploratory"})
    # B: Holm across noise levels for the per-level Friedman family
    lvl_rows = [r for r in rows if r["analysis"] == "B_friedman_per_level"]
    if len(lvl_rows) >= 2:
        rej, holm, _, _ = multipletests([r["p"] for r in lvl_rows],
                                        alpha=0.05, method="holm")
        for r, hp, rj in zip(lvl_rows, holm, rej):
            r["holm_p"] = hp
            r["decision"] = ("omnibus_significant_holm" if rj
                             else "no_omnibus_evidence")
            rows.append({"analysis": "B_friedman_per_level_holm",
                         "subset": r["subset"] + " (family-corrected)",
                         "statistic": r["statistic"], "df": r["df"],
                         "p": r["p"], "holm_p": hp,
                         "effect": r["effect"], "ci_low": np.nan,
                         "ci_high": np.nan,
                         "n_datasets": r["n_datasets"],
                         "decision": ("significant_after_holm" if rj
                                      else "not_significant_after_holm")})
    # degradation slopes (descriptive)
    base = b[b.noise_level == 0.0]
    if len(base):
        wb = dataset_level(base)
        for lev in sorted(b["noise_level"].unique()):
            if lev == 0.0:
                continue
            wl = dataset_level(b[b.noise_level == lev])
            for m in PRIMARY:
                if m in wb.columns and m in wl.columns:
                    d = (wl[m] - wb[m]).values
                    rows.append({"analysis": "B_degradation",
                                 "subset": f"noise={lev} {m}",
                                 "statistic": np.median(d), "df": np.nan,
                                 "p": np.nan, "holm_p": np.nan,
                                 "effect": rank_biserial(d), "ci_low": np.nan,
                                 "ci_high": np.nan, "n_datasets": len(d),
                                 "decision": "exploratory"})
    # C: stability descriptives vs legacy reference
    legacy = {"cart_custom": 0.0928, "spdt": 0.1014}
    for m, ref in legacy.items():
        if m in c.method.unique():
            d = c[c.method == m].prediction_disagreement.mean()
            rows.append({"analysis": "C_vs_legacy", "subset": m,
                         "statistic": d, "df": np.nan, "p": np.nan,
                         "holm_p": np.nan, "effect": d - ref, "ci_low": np.nan,
                         "ci_high": np.nan, "n_datasets": c.dataset.nunique(),
                         "decision": f"legacy_ref={ref}"})
    return pd.DataFrame(rows)
