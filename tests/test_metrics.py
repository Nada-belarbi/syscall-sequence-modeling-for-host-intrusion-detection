"""Unit tests for src/metrics.py"""
from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import compute_binary_metrics


def test_perfect_classifier():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    m = compute_binary_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["fpr"] == pytest.approx(0.0)
    assert m["fnr"] == pytest.approx(0.0)


def test_all_wrong_classifier():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    m = compute_binary_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.0)
    assert m["f1"] == pytest.approx(0.0)


def test_roc_auc_with_scores():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    m = compute_binary_metrics(y_true, y_pred, y_score=y_score)
    assert "roc_auc" in m
    assert m["roc_auc"] == pytest.approx(1.0)
    assert "pr_auc" in m
    assert m["pr_auc"] == pytest.approx(1.0)


def test_confusion_matrix_counts():
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 0]
    m = compute_binary_metrics(y_true, y_pred)
    assert m["tp"] == pytest.approx(2.0)
    assert m["tn"] == pytest.approx(2.0)
    assert m["fp"] == pytest.approx(1.0)
    assert m["fn"] == pytest.approx(1.0)


def test_safe_div_zero_denominator():
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 1]
    m = compute_binary_metrics(y_true, y_pred)
    assert m["fpr"] == pytest.approx(0.0)  # no negatives → denominator=0
