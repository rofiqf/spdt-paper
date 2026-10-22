"""Split-stability measure tests."""

import numpy as np

from src.split_stability import (prediction_disagreement,
                                 root_feature_agreement,
                                 root_threshold_variation)


def test_root_feature_agreement_bounds():
    assert root_feature_agreement([0, 0, 0]) == 1.0
    assert root_feature_agreement([0, 1]) == 0.0
    a = root_feature_agreement([0, 0, 1, 1])
    assert abs(a - 1 / 3) < 1e-9


def test_threshold_variation_same_feature_only():
    feats = [0, 0, 0]
    thrs = [1.0, 1.1, 0.9]
    v = root_threshold_variation(feats, thrs)
    assert abs(v - 0.1) < 1e-9
    # hanya threshold pada FITUR MODAL yang dihitung: model ke-1 (fitur 0,
    # thr 5.0) harus diabaikan; modal=fitur 1 dengan thr [1.0, 1.2]
    feats2 = [0, 1, 1]
    thrs2 = [5.0, 1.0, 1.2]
    v2 = root_threshold_variation(feats2, thrs2)
    assert abs(v2 - np.std([1.0, 1.2], ddof=1)) < 1e-9
    # semua beda fitur -> tidak ada grup modal >= 2 -> NaN
    assert np.isnan(root_threshold_variation([0, 1, 2], [1.0, 1.0, 1.0]))


def test_threshold_variation_min_count():
    feats = [0, 1, 1]
    thrs = [np.nan, 2.0, np.nan]
    v = root_threshold_variation(feats, thrs)
    assert np.isnan(v)  # modal feature hanya 1 threshold finite


def test_prediction_disagreement_bounds():
    preds = np.array([["a", "b"], ["a", "b"], ["b", "b"]])
    d = prediction_disagreement(preds)
    assert 0.0 <= d <= 1.0
    same = np.tile(np.array(["a", "b"]), (3, 1))
    assert prediction_disagreement(same) == 0.0
