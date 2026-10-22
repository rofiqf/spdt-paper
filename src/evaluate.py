"""Nested repeated stratified CV evaluation engine.

Guarantees:
- Identical outer splits across all methods (splits computed once per
  dataset from the config: one RepeatedStratifiedKFold per seed).
- Preprocessing (median imputation) fitted inside a sklearn Pipeline, so
  every inner fold and the outer test fold only ever see train-fitted
  transformers.
- Inner 3-fold tuning per outer fold with deterministically derived seeds.
- Each outer test fold is evaluated exactly once per trained model.
- Per-dataset checkpoints for resumability (results/raw/checkpoints/).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from . import baselines

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"


def build_model_specs(config: dict) -> dict:
    """Map config.models to {name: {factory, grid, fixed}}."""
    specs = {}
    for name, mc in config["models"].items():
        if mc.get("optional") and not mc.get("enabled", False):
            continue
        if name == "cart_sklearn":
            factory = baselines.make_sklearn_cart
        elif name == "cart_custom":
            factory = baselines.make_custom_cart
        elif name == "spdt":
            factory = baselines.make_spdt
        elif name == "random_forest":
            factory = baselines.make_random_forest
        else:
            raise ValueError(f"unknown model {name}")
        specs[name] = {
            "factory": factory,
            "grid": {k: (list(v) if isinstance(v, list) else v)
                     for k, v in mc.get("grid", {}).items()},
            "fixed": mc.get("fixed", {}),
        }
    return specs


def outer_splits(y, config):
    """Yield (seed, repeat, fold, train_idx, test_idx) deterministically."""
    cv = config["cv"]["outer"]
    X_dummy = np.zeros((len(y), 1))
    for seed in cv["seeds"]:
        from sklearn.model_selection import RepeatedStratifiedKFold
        rskf = RepeatedStratifiedKFold(
            n_splits=cv["splits"], n_repeats=cv["repeats"],
            random_state=seed)
        for i, (tr, te) in enumerate(rskf.split(X_dummy, y)):
            yield seed, i // cv["splits"], i % cv["splits"], tr, te


def inner_seed(seed, repeat, fold):
    ss = np.random.SeedSequence([int(seed), int(repeat), int(fold)])
    return int(ss.generate_state(1)[0])


def _tree_stats(fitted_pipe):
    clf = fitted_pipe.named_steps["clf"]
    if hasattr(clf, "tree_"):  # sklearn CART
        return int(clf.tree_.node_count), int(clf.tree_.max_depth)
    if hasattr(clf, "node_count_"):  # custom tree
        return int(clf.node_count_()), int(clf.tree_depth_())
    return np.nan, np.nan  # random forest etc.


def _disagreement(preds: np.ndarray) -> float:
    """preds: (S, n_test). Mean pairwise disagreement rate."""
    S = preds.shape[0]
    if S < 2:
        return 0.0
    agree = 0.0
    for i in range(S):
        agree += (preds[i][None, :] == preds).sum()
    agree -= S * preds.shape[1]  # remove self-pairs
    total = S * (S - 1) * preds.shape[1]
    return float(1.0 - agree / total)


def run_dataset(dataset_name, X, y, config, specs, checkpoint_dir,
                stability_resamples=0, verbose=True):
    rows = []
    stab_rows = []
    y_arr = np.asarray(y)
    n = len(y_arr)
    for seed, repeat, fold, tr_idx, te_idx in outer_splits(y_arr, config):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]
        for mname, spec in specs.items():
            est = spec["factory"](**spec["fixed"])
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("clf", est)])
            grid = {f"clf__{k}": v for k, v in spec["grid"].items()}
            i_seed = inner_seed(seed, repeat, fold)
            inner_cv = StratifiedKFold(n_splits=config["cv"]["inner"]["splits"],
                                       shuffle=True, random_state=i_seed)
            t0 = time.perf_counter()
            search = GridSearchCV(pipe, grid, scoring="f1_macro",
                                  cv=inner_cv, n_jobs=-1, refit=True)
            search.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0
            best = search.best_params_
            t0 = time.perf_counter()
            pred = search.predict(X_te)
            pred_s = time.perf_counter() - t0
            nodes, depth = _tree_stats(search.best_estimator_)
            rows.append({
                "dataset": dataset_name, "method": mname,
                "seed": seed, "repeat": repeat, "outer_fold": fold,
                "n_train": len(tr_idx), "n_test": len(te_idx),
                "inner_seed": i_seed,
                "best_params": json.dumps(
                    {k.replace("clf__", ""): v for k, v in best.items()}),
                "f1_macro": f1_score(y_te, pred, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(y_te, pred),
                "accuracy": accuracy_score(y_te, pred),
                "fit_seconds": fit_s, "predict_seconds": pred_s,
                "node_count": nodes, "tree_depth": depth,
            })
            if stability_resamples > 0:
                preds = np.empty((stability_resamples, len(te_idx)),
                                 dtype=object)
                for s in range(stability_resamples):
                    ss = np.random.SeedSequence(
                        [int(seed), int(repeat), int(fold), 777, s])
                    rs = np.random.default_rng(ss)
                    sub = rs.choice(len(tr_idx),
                                    size=int(0.9 * len(tr_idx)),
                                    replace=False)
                    est_s = clone(search.best_estimator_)
                    est_s.fit(X_tr.iloc[sub], y_tr[sub])
                    preds[s] = est_s.predict(X_te)
                stab_rows.append({
                    "dataset": dataset_name, "method": mname,
                    "seed": seed, "repeat": repeat, "outer_fold": fold,
                    "disagreement": _disagreement(preds),
                })
        if verbose:
            print(f"[{dataset_name}] seed={seed} repeat={repeat} "
                  f"fold={fold} done", flush=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            checkpoint_dir / f"{dataset_name}.csv", index=False)
    return rows, stab_rows
