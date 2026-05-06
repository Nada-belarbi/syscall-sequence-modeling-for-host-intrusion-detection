from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .metrics import comparison_table, compute_binary_metrics


@dataclass(frozen=True)
class BaselineRunSpec:
    name: str
    algorithm: str
    feature_set: str


def _build_model(algorithm: str, random_state: int = 42) -> Any:
    if algorithm == "logreg":
        return LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            n_jobs=None,
            random_state=random_state,
        )
    if algorithm == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _get_score(model: Any, x: csr_matrix, y_pred: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-raw))
    return y_pred.astype(np.float64)


def run_baseline_suite(
    features: dict[str, Any],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Train required baselines and return one unified comparison table.

    Required runs:
    - Logistic Regression on BoW
    - Logistic Regression on TF-IDF
    - Random Forest on BoW
    - Random Forest on TF-IDF
    - Baseline on bigrams (Logistic Regression)
    """
    specs = [
        BaselineRunSpec(name="lr_bow", algorithm="logreg", feature_set="bow"),
        BaselineRunSpec(name="lr_tfidf", algorithm="logreg", feature_set="tfidf"),
        BaselineRunSpec(name="rf_bow", algorithm="rf", feature_set="bow"),
        BaselineRunSpec(name="rf_tfidf", algorithm="rf", feature_set="tfidf"),
        BaselineRunSpec(name="lr_bigram", algorithm="logreg", feature_set="bigram"),
    ]

    rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    preds: dict[str, dict[str, np.ndarray]] = {}

    for spec in specs:
        feat = features[spec.feature_set]
        x_train = feat["train"]
        x_val = feat["val"]
        x_test = feat["test"]

        model = _build_model(spec.algorithm, random_state=random_state)
        model.fit(x_train, y_train)

        y_val_pred = model.predict(x_val)
        y_test_pred = model.predict(x_test)
        y_val_score = _get_score(model, x_val, y_val_pred)
        y_test_score = _get_score(model, x_test, y_test_pred)

        val_metrics = compute_binary_metrics(y_val, y_val_pred, y_val_score)
        test_metrics = compute_binary_metrics(y_test, y_test_pred, y_test_score)

        rows.append(
            {
                "model": spec.name,
                "algorithm": spec.algorithm,
                "feature_set": spec.feature_set,
                "split": "val",
                **val_metrics,
            }
        )
        rows.append(
            {
                "model": spec.name,
                "algorithm": spec.algorithm,
                "feature_set": spec.feature_set,
                "split": "test",
                **test_metrics,
            }
        )

        models[spec.name] = model
        preds[spec.name] = {
            "val_pred": np.asarray(y_val_pred),
            "val_score": np.asarray(y_val_score),
            "test_pred": np.asarray(y_test_pred),
            "test_score": np.asarray(y_test_score),
        }

    result_df = comparison_table(rows, sort_by="f1")
    return result_df, models, preds
