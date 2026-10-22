"""Split-selection stability measures.

Root feature agreement: fraction of model pairs selecting the same root
feature. Root threshold variation: SD of root thresholds ONLY among models
that selected the same root feature (never compare across features).
Prediction disagreement is kept from the original protocol.
"""

from __future__ import annotations

import numpy as np


def root_feature_agreement(root_features) -> float:
    """Mean pairwise agreement that two models share the root feature."""
    rf = np.asarray(root_features)
    S = len(rf)
    if S < 2:
        return 1.0
    agree = 0.0
    for i in range(S):
        agree += (rf[i] == rf).sum() - 1
    return float(agree / (S * (S - 1)))


def root_threshold_variation(root_features, root_thresholds) -> float:
    """SD of thresholds among models sharing the modal root feature;
    NaN if the modal feature is selected by fewer than 2 models."""
    rf = np.asarray(root_features)
    rt = np.asarray(root_thresholds, dtype=float)
    if len(rf) == 0:
        return float("nan")
    vals, counts = np.unique(rf, return_counts=True)
    modal = vals[np.argmax(counts)]
    thr = rt[rf == modal]
    thr = thr[np.isfinite(thr)]
    if len(thr) < 2:
        return float("nan")
    return float(np.std(thr, ddof=1))


def prediction_disagreement(preds: np.ndarray) -> float:
    """Mean pairwise disagreement on a fixed evaluation set."""
    preds = np.asarray(preds)
    S, n = preds.shape
    if S < 2:
        return 0.0
    agree = 0.0
    for i in range(S):
        agree += (preds[i][None, :] == preds).sum() - n
    return float(1.0 - agree / (S * (S - 1) * n))
