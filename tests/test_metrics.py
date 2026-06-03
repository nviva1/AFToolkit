"""Tests for adapter metrics and model-suffix helpers."""

import numpy as np
import pytest

from aftoolkit import calculate_metrics
from aftoolkit.models.utils import get_model_suffix


def test_calculate_metrics_perfect_prediction():
    targets = np.array([0.0, -1.0, 0.5, -0.9, 1.2])
    preds = targets.copy()
    m = calculate_metrics(preds, targets)
    assert m["SpearmanR"] == pytest.approx(1.0)
    assert m["PearsonR"] == pytest.approx(1.0)
    assert m["RMSE"] == pytest.approx(0.0)
    assert m["R2"] == pytest.approx(1.0)


def test_calculate_metrics_keys_present():
    targets = np.array([0.0, -1.0, 0.5, -0.9])
    preds = np.array([0.1, -0.8, 0.3, -1.2])
    m = calculate_metrics(preds, targets)
    for key in ("PearsonR", "SpearmanR", "RMSE", "ROC AUC", "MCC", "ACC", "R2"):
        assert key in m


class _DummyModel:
    concat_features = True
    protein_aggregation = "mutpos"
    multi_aggregation = "sum"


def test_get_model_suffix():
    suffix = get_model_suffix(_DummyModel(), multi_train=False)
    assert suffix == "_concat_nomultitrain_aggmutpos_multisum"

    _DummyModel.concat_features = False
    suffix = get_model_suffix(_DummyModel(), multi_train=True)
    assert suffix == "_diff_multitrain_aggmutpos_multisum"
    _DummyModel.concat_features = True  # restore
