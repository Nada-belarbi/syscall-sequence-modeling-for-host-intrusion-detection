from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)


def _safe_div(numerator: float, denominator: float) -> float:
	if denominator == 0.0:
		return 0.0
	return float(numerator / denominator)


def compute_binary_metrics(
	y_true: np.ndarray | list[int],
	y_pred: np.ndarray | list[int],
	y_score: np.ndarray | list[float] | None = None,
) -> dict[str, float]:
	"""Compute core binary classification metrics with robust fallbacks."""
	y_true_arr = np.asarray(y_true, dtype=np.int64)
	y_pred_arr = np.asarray(y_pred, dtype=np.int64)

	tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

	metrics: dict[str, float] = {
		"accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
		"precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
		"recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
		"f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
		"fpr": _safe_div(float(fp), float(fp + tn)),
		"fnr": _safe_div(float(fn), float(fn + tp)),
		"tp": float(tp),
		"tn": float(tn),
		"fp": float(fp),
		"fn": float(fn),
	}

	if y_score is not None:
		y_score_arr = np.asarray(y_score, dtype=np.float64)
		try:
			metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
		except ValueError:
			metrics["roc_auc"] = float("nan")
		try:
			metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_score_arr))
		except ValueError:
			metrics["pr_auc"] = float("nan")
	else:
		metrics["roc_auc"] = float("nan")
		metrics["pr_auc"] = float("nan")

	return metrics


def confusion_matrix_df(y_true: np.ndarray | list[int], y_pred: np.ndarray | list[int]) -> pd.DataFrame:
	"""Return a labeled confusion matrix as a DataFrame."""
	cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=[0, 1])
	return pd.DataFrame(cm, index=["true_normal", "true_attack"], columns=["pred_normal", "pred_attack"])


def save_confusion_matrix_plot(
	y_true: np.ndarray | list[int],
	y_pred: np.ndarray | list[int],
	output_path: Path,
	title: str,
) -> None:
	"""Save a confusion matrix heatmap to disk."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	cm_df = confusion_matrix_df(y_true, y_pred)
	fig, ax = plt.subplots(figsize=(5, 4))
	sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
	ax.set_title(title)
	ax.set_xlabel("Predicted")
	ax.set_ylabel("True")
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def comparison_table(rows: list[dict[str, Any]], sort_by: str = "f1") -> pd.DataFrame:
	"""Build a unified model comparison table sorted by the target metric."""
	df = pd.DataFrame(rows)
	if df.empty:
		return df
	if sort_by in df.columns:
		df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
	return df
