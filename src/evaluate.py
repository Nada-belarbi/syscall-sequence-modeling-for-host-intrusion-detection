from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from .metrics import compute_binary_metrics


def evaluate_sklearn_classifier(model: Any, x: Any, y_true: np.ndarray | list[int]) -> dict[str, Any]:
	"""Evaluate sklearn-like classifier and return metrics + predictions."""
	y_true_arr = np.asarray(y_true, dtype=np.int64)
	y_pred = np.asarray(model.predict(x), dtype=np.int64)

	if hasattr(model, "predict_proba"):
		y_score = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
	elif hasattr(model, "decision_function"):
		raw = np.asarray(model.decision_function(x), dtype=np.float64)
		y_score = 1.0 / (1.0 + np.exp(-raw))
	else:
		y_score = y_pred.astype(np.float64)

	metrics = compute_binary_metrics(y_true_arr, y_pred, y_score)
	return {
		"metrics": metrics,
		"y_true": y_true_arr,
		"y_pred": y_pred,
		"y_score": y_score,
	}


@torch.no_grad()
def evaluate_lstm_classifier(
	model: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	device: torch.device,
	threshold: float = 0.5,
) -> dict[str, Any]:
	"""Evaluate an LSTM model on a PyTorch dataloader."""
	model.eval()
	model = model.to(device)

	y_true: list[int] = []
	y_pred: list[int] = []
	y_score: list[float] = []
	trace_ids: list[str] = []

	for batch in loader:
		input_ids = batch["input_ids"].to(device)
		lengths = batch["lengths"].to(device)
		labels = batch["labels"].long().to(device)

		logits = model(input_ids=input_ids, lengths=lengths)
		probs = torch.sigmoid(logits)
		preds = (probs >= threshold).long()

		y_true.extend(labels.detach().cpu().tolist())
		y_pred.extend(preds.detach().cpu().tolist())
		y_score.extend(probs.detach().cpu().tolist())
		trace_ids.extend([str(x) for x in batch.get("sample_ids", [])])

	y_true_arr = np.asarray(y_true, dtype=np.int64)
	y_pred_arr = np.asarray(y_pred, dtype=np.int64)
	y_score_arr = np.asarray(y_score, dtype=np.float64)

	metrics = compute_binary_metrics(y_true_arr, y_pred_arr, y_score_arr)
	return {
		"metrics": metrics,
		"y_true": y_true_arr,
		"y_pred": y_pred_arr,
		"y_score": y_score_arr,
		"trace_ids": trace_ids,
	}


def find_best_threshold(
	y_true: np.ndarray | list[int],
	y_score: np.ndarray | list[float],
	metric: str = "f1",
	max_fpr: float | None = None,
	thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
	"""Find the best decision threshold on validation predictions."""
	y_true_arr = np.asarray(y_true, dtype=np.int64)
	y_score_arr = np.asarray(y_score, dtype=np.float64)
	if thresholds is None:
		thresholds = np.linspace(0.05, 0.95, 37)

	best_thr = 0.5
	best_metrics: dict[str, float] | None = None
	best_value = float("-inf")

	for thr in thresholds:
		y_pred = (y_score_arr >= float(thr)).astype(np.int64)
		m = compute_binary_metrics(y_true_arr, y_pred, y_score_arr)
		if max_fpr is not None and float(m.get("fpr", 1.0)) > max_fpr:
			continue
		value = float(m.get(metric, float("-inf")))
		if value > best_value:
			best_value = value
			best_thr = float(thr)
			best_metrics = m

	if best_metrics is None:
		fallback_pred = (y_score_arr >= 0.5).astype(np.int64)
		best_metrics = compute_binary_metrics(y_true_arr, fallback_pred, y_score_arr)
		best_thr = 0.5

	return {
		"best_threshold": best_thr,
		"metric": metric,
		"metrics": best_metrics,
	}


def build_prediction_frame(
	trace_ids: list[str],
	y_true: np.ndarray,
	y_pred: np.ndarray,
	y_score: np.ndarray,
) -> pd.DataFrame:
	"""Build a normalized per-sample prediction table."""
	df = pd.DataFrame(
		{
			"trace_id": trace_ids,
			"y_true": y_true.astype(int),
			"y_pred": y_pred.astype(int),
			"y_score": y_score.astype(float),
		}
	)
	df["error_type"] = "correct"
	df.loc[(df["y_true"] == 0) & (df["y_pred"] == 1), "error_type"] = "false_positive"
	df.loc[(df["y_true"] == 1) & (df["y_pred"] == 0), "error_type"] = "false_negative"
	return df


def error_rates_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
	"""Compute FP/FN rates by group over a prediction frame with y_true/y_pred."""
	if group_col not in df.columns:
		raise ValueError(f"Missing group column: {group_col}")

	rows: list[dict[str, Any]] = []
	for key, group in df.groupby(group_col):
		tn = int(((group["y_true"] == 0) & (group["y_pred"] == 0)).sum())
		fp = int(((group["y_true"] == 0) & (group["y_pred"] == 1)).sum())
		fn = int(((group["y_true"] == 1) & (group["y_pred"] == 0)).sum())
		tp = int(((group["y_true"] == 1) & (group["y_pred"] == 1)).sum())
		rows.append(
			{
				group_col: key,
				"count": int(len(group)),
				"fpr": fp / max(1, fp + tn),
				"fnr": fn / max(1, fn + tp),
				"error_rate": (fp + fn) / max(1, len(group)),
			}
		)
	return pd.DataFrame(rows).sort_values("error_rate", ascending=False).reset_index(drop=True)
