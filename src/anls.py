"""ANLS (Lee, Sim, Hong 2024, IEEE Access, DOI 10.1109/ACCESS.2024.3419228)
faithful re-implementation for comparator M5.

Algorithm (paper Sec. III-A pseudocode): at each node, repeatedly draw a
bootstrap resample of the node data, find the plain CART-optimal split
(predictor + cutoff), and accumulate predictor frequencies; stop when the
binomial Z-statistic of the top-two predictors exceeds Z_alpha (min
`min_iterations` replicates first). The stabilized split uses the modal
predictor; for continuous predictors the cutoff is the MEDIAN of the
winning cutoffs. Pre-pruning: nodes smaller than `min_leaf_frac` of the
training set are not split.

Documented deviations (runtime control): a `max_iterations` cap is applied
(the paper used no cap and reports up to 47,587 iterations per node); our
datasets are entirely numeric so the categorical branch is omitted.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class _ANLSNode:
    __slots__ = ("is_leaf", "prediction", "proba", "feature", "threshold",
                 "left", "right")

    def __init__(self):
        self.is_leaf = True
        self.prediction = 0
        self.proba = None
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None


def _best_split(X, y_enc, n_classes):
    """Plain CART-optimal (feature, threshold) on given rows; vectorized.
    Returns (feature, threshold) or None."""
    n = X.shape[0]
    best = None
    node_totals = np.bincount(y_enc, minlength=n_classes).astype(float)
    gini_node = 1.0 - np.sum((node_totals / n) ** 2)
    for j in range(X.shape[1]):
        v = X[:, j]
        order = np.argsort(v, kind="mergesort")
        v_sorted = v[order]
        if n < 2 or not (v_sorted[1:] > v_sorted[:-1]).any():
            continue
        oh = np.zeros((n, n_classes))
        oh[np.arange(n), y_enc[order]] = 1.0
        left = np.cumsum(oh, axis=0)[:-1]
        ln = left.sum(axis=1)
        rn = n - ln
        mask = (v_sorted[1:] > v_sorted[:-1]) & (ln >= 1) & (rn >= 1)
        if not mask.any():
            continue
        right = node_totals - left
        gl = 1.0 - np.sum((left / ln[:, None]) ** 2, axis=1)
        gr = 1.0 - np.sum((right / rn[:, None]) ** 2, axis=1)
        gain = gini_node - (ln / n) * gl - (rn / n) * gr
        gain = np.where(mask, gain, -np.inf)
        t = int(np.argmax(gain))
        if best is None or gain[t] > best[0]:
            thr = 0.5 * (v_sorted[t] + v_sorted[t + 1])
            best = (gain[t], j, thr)
    if best is None or best[0] <= 0:
        return None
    return best[1], best[2]


class ANLSTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, z_alpha: float = 1.645, min_iterations: int = 10,
                 min_leaf_frac: float = 0.01, max_iterations: int = 1500,
                 random_state: int = 0):
        self.z_alpha = z_alpha
        self.min_iterations = min_iterations
        self.min_leaf_frac = min_leaf_frac
        self.max_iterations = max_iterations
        self.random_state = random_state

    def _leaf(self, y_enc):
        node = _ANLSNode()
        counts = np.bincount(y_enc, minlength=self._n_classes).astype(float)
        node.proba = counts / counts.sum()
        node.prediction = int(np.argmax(counts))
        return node

    def _stabilized_split(self, X, y_enc, node_id):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(self.random_state), 997, node_id]))
        n = X.shape[0]
        feats, thrs = [], []
        z = -np.inf
        it = 0
        while z <= self.z_alpha and it < self.max_iterations:
            it += 1
            idx = rng.integers(0, n, size=n)
            bs = _best_split(X[idx], y_enc[idx], self._n_classes)
            if bs is None:
                continue
            feats.append(bs[0])
            thrs.append(bs[1])
            if it >= self.min_iterations and len(feats) >= 2:
                f = np.asarray(feats)
                vals, counts = np.unique(f, return_counts=True)
                order = np.argsort(-counts)
                x1 = counts[order[0]]
                x2 = counts[order[1]] if len(order) > 1 else 0
                denom = x1 + x2
                if x2 == 0:
                    z = np.inf  # konvergen sempurna
                else:
                    z = (x1 / denom - 0.5) / np.sqrt(0.25 / denom)
        if not feats:
            return None
        f = np.asarray(feats)
        t = np.asarray(thrs)
        vals, counts = np.unique(f, return_counts=True)
        winner = vals[np.argmax(counts)]
        return int(winner), float(np.median(t[f == winner])), it

    def _build(self, X, y_enc, node_id):
        node = self._leaf(y_enc)
        n = X.shape[0]
        if (n < max(2, int(np.ceil(self.min_leaf_frac * self._n_train)))
                or np.unique(y_enc).size == 1):
            return node
        split = self._stabilized_split(X, y_enc, node_id)
        if split is None:
            return node
        j, thr, _ = split
        left = X[:, j] <= thr
        if left.sum() == 0 or (~left).sum() == 0:
            return node
        node = _ANLSNode()
        node.is_leaf = False
        node.feature = j
        node.threshold = thr
        node.proba = self._leaf(y_enc).proba
        node.prediction = self._leaf(y_enc).prediction
        self._node_counter += 1
        node.left = self._build(X[left], y_enc[left], self._node_counter)
        node.right = self._build(X[~left], y_enc[~left], self._node_counter)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if np.isnan(X).any():
            raise ValueError("NaN in X: impute upstream")
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self._n_classes = len(self.classes_)
        self._n_train = X.shape[0]
        self._node_counter = 0
        self.tree_ = self._build(X, y_enc, 0)
        return self

    def _traverse(self, x):
        node = self.tree_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self.classes_[self._traverse(x).prediction]
                         for x in X])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._traverse(x).proba for x in X])

    def root_split_(self):
        r = self.tree_
        if r.is_leaf:
            return -1, np.nan
        return r.feature, r.threshold
