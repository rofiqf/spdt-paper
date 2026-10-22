"""Extension evaluation engine (Experiments A/B/C/D).

Staged tuning (documented, training-only, reproducible):
  S1: structure (max_depth, min_samples_leaf, min_samples_split) via inner
      3-fold CV on plain custom CART (lambda=0; fast).
  S2: lambda in {0.1, 1, 10} for original SPDT at fixed structure (B=10).
  S3: lambda in {0.1, 1, 10} for normalized SPDT at fixed structure
      (B=10, eps=1e-4).
  sklearn CART: its own structural GridSearchCV. RF: fixed contextual.

Per outer fold, one shared pipeline run produces rows for:
  A normalized_penalty (clean eval, root split, counters)
  B noise_robustness (frozen params, corrupted train, clean test)
  C split_stability (S subsample fits, frozen params)
  D runtime tradeoff counters (from A fits)
Tuned params are dumped to logs/extension_tuned_params.json for reuse by
the ablation runner (Experiment E).
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import data as data_mod
from .evaluate import inner_seed, outer_splits
from .noise import corrupt_labels, corruption_seed
from .split_stability import (prediction_disagreement,
                              root_feature_agreement,
                              root_threshold_variation)
from .tree import StabilityTreeClassifier
from .anls import ANLSTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "results" / "extension"

STRUCT_GRID = {"max_depth": [3, 5, 8, None],
               "min_samples_leaf": [1, 2, 5],
               "min_samples_split": [2, 5, 10]}
LAMBDA_GRID = [0.1, 1.0, 10.0]


def _pipe(custom: bool, **kw):
    clf = (StabilityTreeClassifier(random_state=0, **kw)
           if custom else DecisionTreeClassifier(criterion="gini",
                                                random_state=0, **kw))
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("clf", clf)])


def _grid(params: dict) -> dict:
    return {f"clf__{k}": v for k, v in params.items()}


def tune_structure(X_tr, y_tr, cv_seed) -> dict:
    cv = StratifiedKFold(3, shuffle=True, random_state=cv_seed)
    gs = GridSearchCV(_pipe(custom=True), _grid(STRUCT_GRID),
                      scoring="f1_macro", cv=cv, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}


def tune_lambda(X_tr, y_tr, struct, normalized, cv_seed, n_boot=10,
                eps=1e-4) -> float:
    cv = StratifiedKFold(3, shuffle=True, random_state=cv_seed)
    gs = GridSearchCV(
        _pipe(custom=True, n_bootstrap=n_boot,
              normalized_penalty=normalized, penalty_epsilon=eps, **struct),
        {"clf__penalty_lambda": LAMBDA_GRID},
        scoring="f1_macro", cv=cv, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return float(gs.best_params_["clf__penalty_lambda"])


def tune_sklearn(X_tr, y_tr, cv_seed) -> dict:
    cv = StratifiedKFold(3, shuffle=True, random_state=cv_seed)
    gs = GridSearchCV(_pipe(custom=False), _grid(STRUCT_GRID),
                      scoring="f1_macro", cv=cv, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}


def make_method_pipes(struct: dict, lam2: float, lam3: float,
                      skl_struct: dict, cfg) -> dict:
    return {
        "cart_custom": _pipe(custom=True, penalty_lambda=0.0, **struct),
        "spdt": _pipe(custom=True, penalty_lambda=lam2,
                      n_bootstrap=cfg["tune_B"], **struct),
        "spdt_norm": _pipe(custom=True, penalty_lambda=lam3,
                           n_bootstrap=cfg["tune_B"], normalized_penalty=True,
                           penalty_epsilon=cfg["epsilon"], **struct),
        "cart_sklearn": _pipe(custom=False, **skl_struct),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, random_state=20261021, n_jobs=-1))]),
        # M5: ANLS dengan parameter default paper (tidak ditune)
        "anls": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", ANLSTreeClassifier(random_state=0))]),
    }


def fit_predict(pipe, X_tr, y_tr, X_te):
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = pipe.predict(X_te)
    pred_s = time.perf_counter() - t0
    return pred, fit_s, pred_s


def metrics_row(dataset, method, seed, repeat, fold, y_te, pred, fit_s,
                pred_s, pipe, extra=None):
    clf = pipe.named_steps["clf"]
    if isinstance(clf, StabilityTreeClassifier):
        nodes, depth = clf.node_count_(), clf.tree_depth_()
        stats = clf.stats_()
    elif isinstance(clf, DecisionTreeClassifier):
        nodes, depth = int(clf.tree_.node_count), int(clf.tree_.max_depth)
        stats = {"candidates_evaluated": np.nan,
                 "bootstrap_gain_evaluations": 0.0}
    else:
        # ANLS / RandomForest: struktur internal tidak setara
        nodes = depth = np.nan
        stats = {"candidates_evaluated": np.nan,
                 "bootstrap_gain_evaluations": np.nan}
    row = {"dataset": dataset, "method": method, "seed": seed,
           "repeat": repeat, "outer_fold": fold,
           "f1_macro": f1_score(y_te, pred, average="macro"),
           "balanced_accuracy": balanced_accuracy_score(y_te, pred),
           "accuracy": accuracy_score(y_te, pred),
           "fit_seconds": fit_s, "predict_seconds": pred_s,
           "node_count": nodes, "tree_depth": depth,
           "candidates_evaluated": stats["candidates_evaluated"],
           "bootstrap_gain_evaluations": stats["bootstrap_gain_evaluations"]}
    if extra:
        row.update(extra)
    return row


def root_split(pipe):
    clf = pipe.named_steps["clf"]
    if isinstance(clf, StabilityTreeClassifier):
        r = clf.tree_
        return (r.feature if not r.is_leaf else -1,
                r.threshold if not r.is_leaf else np.nan)
    if isinstance(clf, ANLSTreeClassifier):
        return clf.root_split_()
    if isinstance(clf, DecisionTreeClassifier):
        return int(clf.tree_.feature[0]), float(clf.tree_.threshold[0])
    return np.nan, np.nan


def run_fold(cfg, name, dset_idx, X, y, seed, repeat, fold, tr, te,
             rows_a, rows_b, rows_c, tuned_log):
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr, y_te = np.asarray(y)[tr], np.asarray(y)[te]
    cv_seed = inner_seed(seed, repeat, fold)

    struct = tune_structure(X_tr, y_tr, cv_seed)
    lam2 = tune_lambda(X_tr, y_tr, struct, False, cv_seed,
                       cfg["tune_B"], cfg["epsilon"])
    lam3 = tune_lambda(X_tr, y_tr, struct, True, cv_seed,
                       cfg["tune_B"], cfg["epsilon"])
    skl_struct = tune_sklearn(X_tr, y_tr, cv_seed)
    pipes = make_method_pipes(struct, lam2, lam3, skl_struct, cfg)
    tuned_log.append({"dataset": name, "seed": seed, "repeat": repeat,
                      "outer_fold": fold, "struct": struct,
                      "lambda_spdt": lam2, "lambda_spdt_norm": lam3,
                      "struct_sklearn": skl_struct})

    methods_a = ["cart_custom", "spdt", "spdt_norm", "cart_sklearn",
                 "random_forest", "anls"]
    for m in methods_a:
        p = clone(pipes[m])
        pred, fit_s, pred_s = fit_predict(p, X_tr, y_tr, X_te)
        rf, rt = root_split(p)
        rows_a.append(metrics_row(name, m, seed, repeat, fold, y_te, pred,
                                  fit_s, pred_s, p,
                                  {"root_feature": rf,
                                   "root_threshold": rt,
                                   "lambda": (lam2 if m == "spdt" else
                                              lam3 if m == "spdt_norm" else 0)}))

    # ---- B: label noise (frozen params, corrupted train, clean test)
    # ANLS sengaja di luar B: biaya loop bootstrap per fit (~detik) x
    # 75 realisasi/fold tidak proporsional untuk comparator stabilitas;
    # fokus robustness pada M0-M3 (dokumentasikan di experiment_protocol.md)
    methods_b = [m for m in methods_a if m != "anls"]
    for level in cfg["noise_levels"]:
        for r in range(cfg["noise_realizations"]):
            cseed = corruption_seed(cfg["seeds"][0], repeat, fold, level, r,
                                    dset_idx)
            rng = np.random.default_rng(cseed)
            y_noisy = corrupt_labels(y_tr, level, rng)
            for m in methods_b:
                p = clone(pipes[m])
                pred, fit_s, pred_s = fit_predict(p, X_tr, y_noisy, X_te)
                rows_b.append(metrics_row(
                    name, m, seed, repeat, fold, y_te, pred, fit_s, pred_s,
                    p,
                    {"noise_level": level, "realization": r,
                     "corruption_seed": cseed}))

    # ---- C: split stability (S subsample fits, frozen params)
    S = cfg["stability_subsamples"]
    for m in methods_a:
        root_feats, root_thrs, preds = [], [], np.empty((S, len(te)), dtype=object)
        for s_i in range(S):
            ss = np.random.SeedSequence([int(seed), int(repeat), int(fold),
                                         555, dset_idx, s_i])
            rs = np.random.default_rng(ss)
            sub = rs.choice(len(tr), size=int(0.9 * len(tr)), replace=False)
            p = clone(pipes[m]).fit(X_tr.iloc[sub], y_tr[sub])
            rf, rt = root_split(p)
            root_feats.append(rf)
            root_thrs.append(rt)
            preds[s_i] = p.predict(X_te)
        rows_c.append({"dataset": name, "method": m, "seed": seed,
                       "repeat": repeat, "outer_fold": fold,
                       "root_feature_agreement":
                           root_feature_agreement(root_feats),
                       "root_threshold_variation":
                           root_threshold_variation(root_feats, root_thrs),
                       "prediction_disagreement":
                           prediction_disagreement(preds)})


def run_datasets(cfg, datasets, manifest):
    EXT.mkdir(parents=True, exist_ok=True)
    rows_a, rows_b, rows_c, tuned_log = [], [], [], []
    y_full = {}
    for name in datasets:
        drop = None
        X, y = data_mod.load_dataset(name, drop_columns=drop)
        y_full[name] = np.asarray(y)
    for dset_idx, name in enumerate(datasets):
        X, y = data_mod.load_dataset(name)
        y_arr = np.asarray(y)
        for seed, repeat, fold, tr, te in outer_splits(y_arr, cfg):
            run_fold(cfg, name, dset_idx, X, y_arr, seed, repeat, fold,
                     tr, te, rows_a, rows_b, rows_c, tuned_log)
            print(f"[{name}] seed={seed} r={repeat} f={fold} ext done",
                  flush=True)
        # checkpoint per dataset
        _atomic_write(EXT / "normalized_penalty.csv",
                      pd.DataFrame(rows_a))
        _atomic_write(EXT / "noise_robustness.csv",
                      pd.DataFrame(rows_b))
        _atomic_write(EXT / "split_stability.csv",
                      pd.DataFrame(rows_c))
    (ROOT / "logs" / "extension_tuned_params.json").write_text(
        json.dumps(tuned_log, indent=2))
    manifest["rows"] = {"A": len(rows_a), "B": len(rows_b),
                        "C": len(rows_c)}
    return rows_a, rows_b, rows_c


def _atomic_write(path: Path, df: pd.DataFrame):
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(path)


def memory_probe(cfg) -> float:
    """Single-process tracemalloc peak around one penalized fit
    (wdbc-like scale). Documented estimate, not a controlled benchmark."""
    X, y = data_mod.load_dataset("wdbc")
    yv = np.asarray(y)
    ss = np.random.SeedSequence([42, 0, 0])
    rs = np.random.default_rng(ss)
    tr = rs.choice(len(yv), size=int(0.8 * len(yv)), replace=False)
    mask = np.zeros(len(yv), bool)
    mask[tr] = True
    Xi = SimpleImputer(strategy="median").fit_transform(
        np.asarray(X.values, dtype=float))
    tracemalloc.start()
    StabilityTreeClassifier(penalty_lambda=1.0, n_bootstrap=cfg["tune_B"],
                            random_state=0).fit(Xi[mask], yv[mask])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def default_cfg() -> dict:
    return {
        "cv": {"outer": {"seeds": [42, 123, 2026], "splits": 5,
                         "repeats": 3},
               "inner": {"splits": 3}},
        "seeds": [42],
        "tune_B": 10,
        "epsilon": 1e-4,
        "noise_levels": [0.0, 0.05, 0.10, 0.20, 0.30],
        "noise_realizations": 3,
        "stability_subsamples": 10,
    }
