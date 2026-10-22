"""Deterministic label corruption (training labels only).

Protocol guarantees:
- test labels are never touched (caller passes train indices only);
- identical (seed, level, realization, fold) -> identical corruption mask;
- a corrupted label is always replaced by a DIFFERENT valid class;
- same protocol for binary and multiclass.
"""

from __future__ import annotations

import numpy as np


def corrupt_labels(y: np.ndarray, rate: float, rng: np.random.Generator,
                   forbidden: np.ndarray | None = None) -> np.ndarray:
    """Return a copy of y with `rate` fraction of labels flipped to a
    different class. `forbidden` (optional bool mask) marks indices that
    must NOT be corrupted (e.g., already fixed rows)."""
    y = np.asarray(y)
    y_new = y.copy()
    classes = np.unique(y)
    n = len(y)
    n_corrupt = int(round(rate * n))
    if n_corrupt == 0:
        return y_new
    eligible = np.ones(n, dtype=bool)
    if forbidden is not None:
        eligible &= ~forbidden
    idx = np.where(eligible)[0]
    if len(idx) == 0:
        return y_new
    chosen = rng.choice(idx, size=min(n_corrupt, len(idx)), replace=False)
    for i in chosen:
        others = classes[classes != y[i]]
        y_new[i] = others[rng.integers(0, len(others))]
    return y_new


def corruption_seed(base_seed: int, repeat: int, fold: int, level: float,
                    realization: int, dataset_idx: int) -> int:
    ss = np.random.SeedSequence([int(base_seed), int(repeat), int(fold),
                                 int(level * 1000), int(realization),
                                 int(dataset_idx)])
    return int(ss.generate_state(1)[0])
