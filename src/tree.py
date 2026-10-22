"""Stability-penalized decision tree classifier.

Proposed split criterion (candidate formulation, novelty NOT established):

    Score(s) = Gain(s, D) - lambda * V(s)
    V(s)     = sample variance over B bootstrap resamples of the node data
               of Gain(s, D^(b)), with the candidate (feature, threshold)
               fixed.

For penalty_lambda = 0 the estimator reduces to a custom CART reference
(Gini gain, midpoint thresholds between sorted unique adjacent values,
min_samples_split / min_samples_leaf on the original node data, majority
leaf with deterministic tie-breaking).

Documented deterministic policies:
- Candidate thresholds: midpoints of adjacent *distinct* sorted values.
- Validity (min_samples_leaf) is enforced on ORIGINAL node counts.
- A split is chosen only if the ORIGINAL gain of the best candidate is > 0.
- Empty child partition inside a bootstrap resample contributes 0 to the
  resampled gain (weighted-Gini term omitted; Gini(empty) = 0).
- Majority-vote ties resolve to the smallest encoded (sorted) class label.
- Bootstrap resampling uses only the current node's training rows; seeds are
  derived deterministically from (random_state, node id).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class _Node:
    __slots__ = ("is_leaf", "prediction", "proba",
                 "feature", "threshold", "left", "right")

    def __init__(self):
        self.is_leaf = True
        self.prediction = 0
        self.proba = None
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None


def _gini_from_counts(counts: np.ndarray) -> np.ndarray:
    """Gini from class-count rows. Empty rows (sum 0) give 0."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    out = np.zeros(counts.shape[:-1], dtype=np.float64)
    nz = total[..., 0] > 0
    out[nz] = 1.0 - np.sum(p[nz] ** 2, axis=-1)
    return out


class StabilityTreeClassifier(BaseEstimator, ClassifierMixin):
    """Decision tree with optional bootstrap-variance-penalized split gain.

    penalty_lambda = 0  -> custom CART reference (no bootstrap work).
    penalty_lambda > 0  -> Score = Gain - lambda * Var_b[Gain_b].
    normalized_penalty  -> Score = Gain - lambda * V / (Gain^2 + eps)
                           (SEPARATE exploratory variant; never the default).
    """

    def __init__(self, penalty_lambda: float = 0.0,
                 n_bootstrap: int = 10,
                 max_depth: int | None = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 normalized_penalty: bool = False,
                 penalty_epsilon: float = 1e-12,
                 random_state: int = 0):
        self.penalty_lambda = penalty_lambda
        self.n_bootstrap = n_bootstrap
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.normalized_penalty = normalized_penalty
        self.penalty_epsilon = penalty_epsilon
        self.random_state = random_state

    # --------------------------------------------------------------- utils
    def _leaf(self, y_enc: np.ndarray) -> _Node:
        node = _Node()
        counts = np.bincount(y_enc, minlength=self._n_classes).astype(np.float64)
        node.proba = counts / counts.sum()
        node.prediction = int(np.argmax(counts))  # first max -> deterministic
        return node

    def _candidate_mask(self, v_sorted: np.ndarray) -> np.ndarray:
        """Candidate split positions t: value gap between t and t+1."""
        return v_sorted[1:] > v_sorted[:-1]

    def _orig_gains(self, X, y_enc, n, gini_node):
        """Original CART gain for every (feature, threshold) candidate.

        Returns list over features of (gain vector len n-1, candidate bool
        mask honoring min_samples_leaf on original counts)."""
        out = []
        node_totals = np.bincount(y_enc,
                                  minlength=self._n_classes).astype(np.float64)
        for j in range(X.shape[1]):
            v = X[:, j]
            order = np.argsort(v, kind="mergesort")
            v_sorted = v[order]
            y_sorted = y_enc[order]
            oh = np.zeros((n, self._n_classes))
            oh[np.arange(n), y_sorted] = 1.0
            left_counts = np.cumsum(oh, axis=0)[:-1]           # (n-1, C)
            left_n = left_counts.sum(axis=1)                    # (n-1,)
            right_counts = node_totals - left_counts
            right_n = n - left_n
            lc = left_counts / left_n[:, None]
            rc = right_counts / right_n[:, None]
            gini_l = 1.0 - np.sum(lc ** 2, axis=1)
            gini_r = 1.0 - np.sum(rc ** 2, axis=1)
            gini_l[left_n == 0] = 0.0
            gini_r[right_n == 0] = 0.0
            gain = (gini_node
                    - (left_n / n) * gini_l
                    - (right_n / n) * gini_r)
            valid = ((left_n >= self.min_samples_leaf)
                     & (right_n >= self.min_samples_leaf)
                     & self._candidate_mask(v_sorted))
            out.append((gain, valid, order))
        return out

    def _bootstrap_variance(self, X, y_enc, n):
        """V(s) for every (feature, threshold): variance over resamples.

        Empty child in a resample contributes 0 (term omitted)."""
        rng = np.random.default_rng(
            np.random.SeedSequence([int(self.random_state), self._node_counter]))
        idx = rng.integers(0, n, size=(self.n_bootstrap, n))
        y_draw = y_enc[idx]                                    # (B, n)
        total_b = np.apply_along_axis(
            lambda r: np.bincount(r, minlength=self._n_classes),
            1, y_draw).astype(np.float64)                      # (B, C)
        gini_total_b = 1.0 - np.sum((total_b / n) ** 2, axis=1)  # (B,)
        variances = []
        for j in range(X.shape[1]):
            v = X[:, j]
            order = np.argsort(v, kind="mergesort")
            rank = np.empty(n, dtype=np.int64)
            rank[order] = np.arange(n)
            r_draw = rank[idx]                                 # (B, n)
            # cnt[b, p, c] = # draws with rank p and class c
            cnt = np.zeros((self.n_bootstrap, n, self._n_classes))
            bidx = np.broadcast_to(np.arange(self.n_bootstrap)[:, None],
                                   (self.n_bootstrap, n))
            np.add.at(cnt, (bidx, r_draw, y_draw), 1.0)
            left = np.cumsum(cnt, axis=1)[:, :-1, :]           # (B, n-1, C)
            right = total_b[:, None, :] - left                 # (B, n-1, C)
            left_n = left.sum(axis=2)                          # (B, n-1)
            right_n = n - left_n
            with np.errstate(invalid="ignore", divide="ignore"):
                g_l = 1.0 - np.sum((left / left_n[..., None]) ** 2, axis=2)
                g_r = 1.0 - np.sum((right / right_n[..., None]) ** 2, axis=2)
            gini_l = np.where(left_n > 0, g_l, 0.0)
            gini_r = np.where(right_n > 0, g_r, 0.0)
            gain_b = (gini_total_b[:, None]
                      - np.where(left_n > 0, (left_n / n) * gini_l, 0.0)
                      - np.where(right_n > 0, (right_n / n) * gini_r, 0.0))
            v_t = np.var(gain_b, axis=0, ddof=1)               # (n-1,)
            variances.append(v_t)
        return variances

    # ---------------------------------------------------------------- fit
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y length mismatch")
        if X.shape[0] == 0:
            raise ValueError("empty training data")
        if np.isnan(X).any():
            raise ValueError(
                "NaN in X: median imputation must be fitted upstream "
                "(training partition only) before this estimator")
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self._n_classes = len(self.classes_)
        self._node_counter = 0
        self._depth_seen = 0
        self._n_nodes = 0
        self._candidates_evaluated = 0      # total valid candidate splits
        self._bootstrap_gain_evals = 0      # gain evals over resamples
        self.tree_ = self._build(X, y_enc, depth=0)
        return self

    def stats_(self):
        return {"nodes": self._n_nodes, "depth": self._depth_seen,
                "candidates_evaluated": self._candidates_evaluated,
                "bootstrap_gain_evaluations": self._bootstrap_gain_evals}

    def _build(self, X, y_enc, depth):
        self._depth_seen = max(self._depth_seen, depth)
        n = X.shape[0]

        stopping = (
            (self.max_depth is not None and depth >= self.max_depth)
            or n < self.min_samples_split
            or np.unique(y_enc).size == 1)

        best = None  # (score, gain, feature, thr, order)
        if not stopping:
            gini_node = float(_gini_from_counts(
                np.bincount(y_enc, minlength=self._n_classes)[None, :])[0])
            orig = self._orig_gains(X, y_enc, n, gini_node)

            if self.penalty_lambda and self.penalty_lambda > 0.0:
                varis = self._bootstrap_variance(X, y_enc, n)
            else:
                varis = [None] * X.shape[1]

            for j in range(X.shape[1]):
                gain, valid, order = orig[j]
                if not valid.any():
                    continue
                self._candidates_evaluated += int(valid.sum())
                if varis[j] is None:
                    score = gain
                elif self.normalized_penalty:
                    denom = gain ** 2 + self.penalty_epsilon
                    score = gain - self.penalty_lambda * varis[j] / denom
                else:
                    score = gain - self.penalty_lambda * varis[j]
                score = np.where(valid, score, -np.inf)
                t = int(np.argmax(score))
                if not valid[t]:
                    continue
                cand = (float(score[t]), float(gain[t]), j, t, order)
                if best is None or cand[0] > best[0]:
                    best = cand
            if best is not None and self.penalty_lambda and self.penalty_lambda > 0.0:
                n_valid_node = sum(int(o[1].sum()) for o in orig)
                self._bootstrap_gain_evals += n_valid_node * int(self.n_bootstrap)

        if best is None or best[1] <= 0.0:
            self._n_nodes += 1
            return self._leaf(y_enc)  # daun: stopping / tidak ada gain > 0

        _, gain_best, j, t, order = best
        v_sorted = X[order, j]
        thr = 0.5 * (v_sorted[t] + v_sorted[t + 1])
        left_rows = order[:t + 1]
        right_rows = order[t + 1:]

        self._node_counter += 1
        node = _Node()
        self._n_nodes += 1
        node.is_leaf = False
        node.feature = j
        node.threshold = float(thr)
        node.left = self._build(X[left_rows], y_enc[left_rows], depth + 1)
        node.right = self._build(X[right_rows], y_enc[right_rows], depth + 1)
        self._depth_seen = max(self._depth_seen, depth + 1)
        return node

    # ------------------------------------------------------------ predict
    def _traverse(self, x):
        node = self.tree_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if np.isnan(X).any():
            raise ValueError("NaN in X at predict time")
        return np.array([self.classes_[self._traverse(x).prediction]
                         for x in X])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if np.isnan(X).any():
            raise ValueError("NaN in X at predict time")
        return np.array([self._traverse(x).proba for x in X])

    def node_count_(self):
        return self._n_nodes

    def tree_depth_(self):
        return self._depth_seen
