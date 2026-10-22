"""Noise protocol tests: test labels untouched, identical masks across
methods, deterministic, labels always change."""

import numpy as np

from src.noise import corruption_seed, corrupt_labels


def test_test_labels_never_modified():
    y_tr = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
    y_te = y_tr.copy()
    rng = np.random.default_rng(0)
    _ = corrupt_labels(y_tr, 0.3, rng)
    assert np.array_equal(y_te, y_tr)  # input test tidak disentuh caller


def test_corrupted_label_always_different():
    y = np.repeat([0, 1, 2], 40)
    rng = np.random.default_rng(1)
    y2 = corrupt_labels(y, 0.5, rng)
    changed = y2 != y
    assert changed.any()
    assert np.all(y2[changed] != y[changed])
    assert set(np.unique(y2)) <= {0, 1, 2}


def test_identical_mask_across_calls_and_methods():
    """Seed identik -> mask identik (semua metode dapat korupsi yang sama)."""
    y = np.repeat([0, 1], 50)
    s = corruption_seed(42, 0, 0, 0.2, 1, 3)
    r1 = np.random.default_rng(s)
    r2 = np.random.default_rng(s)
    a = corrupt_labels(y, 0.2, r1)
    b = corrupt_labels(y, 0.2, r2)
    assert np.array_equal(a, b)


def test_zero_rate_is_identity():
    y = np.array([0, 1, 2, 1, 0])
    rng = np.random.default_rng(2)
    assert np.array_equal(corrupt_labels(y, 0.0, rng), y)


def test_different_levels_differ():
    y = np.repeat([0, 1], 100)
    s = corruption_seed(42, 1, 2, 0.05, 0, 0)
    y5 = corrupt_labels(y, 0.05, np.random.default_rng(s))
    s2 = corruption_seed(42, 1, 2, 0.30, 0, 0)
    y30 = corrupt_labels(y, 0.30, np.random.default_rng(s2))
    assert (y30 != y).sum() > (y5 != y).sum()
