from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequencePolicy:
	max_len: int
	pad_value: int = 0
	truncation: str = "post"
	padding: str = "post"


def normalize_whitespace(raw_text: str) -> str:
	"""Normalize multiple whitespace and newline noise."""
	return " ".join(raw_text.strip().split())


def safe_parse_int_tokens(raw_text: str, strict: bool = True) -> list[int]:
	"""Secure token parsing with optional strict failure on corrupted tokens."""
	normalized = normalize_whitespace(raw_text)
	if not normalized:
		return []
	out: list[int] = []
	for tok in normalized.split(" "):
		try:
			out.append(int(tok))
		except ValueError:
			if strict:
				raise ValueError(f"Invalid integer token: {tok}")
	return out


def sanitize_sequence(seq: Iterable[int]) -> list[int]:
	"""Convert iterable to integer sequence, dropping negative syscall IDs."""
	out: list[int] = []
	for value in seq:
		ivalue = int(value)
		if ivalue >= 0:
			out.append(ivalue)
	return out


def choose_max_len(lengths: Iterable[int], quantile: float = 0.95, min_cap: int = 32) -> int:
	"""Choose max sequence length from empirical length quantile."""
	arr = np.array([int(v) for v in lengths if int(v) > 0], dtype=np.int64)
	if arr.size == 0:
		return min_cap
	q_value = int(np.quantile(arr, quantile))
	return max(min_cap, q_value)


def compute_length_statistics(lengths: Iterable[int]) -> dict[str, float]:
	"""Compute descriptive length statistics for scientific justification."""
	arr = np.array([int(v) for v in lengths if int(v) >= 0], dtype=np.int64)
	if arr.size == 0:
		return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
	return {
		"count": float(arr.size),
		"min": float(arr.min()),
		"max": float(arr.max()),
		"mean": float(arr.mean()),
		"median": float(np.quantile(arr, 0.50)),
		"p90": float(np.quantile(arr, 0.90)),
		"p95": float(np.quantile(arr, 0.95)),
		"p99": float(np.quantile(arr, 0.99)),
	}


def suggest_truncation_lengths(lengths: Iterable[int], candidates: list[float] | None = None) -> dict[str, int]:
	"""Suggest max lengths from quantiles for controlled truncation experiments."""
	if candidates is None:
		candidates = [0.90, 0.95, 0.99]
	arr = np.array([int(v) for v in lengths if int(v) > 0], dtype=np.int64)
	if arr.size == 0:
		return {"p90": 64, "p95": 64, "p99": 64}
	out: dict[str, int] = {}
	for q in candidates:
		out[f"p{int(q * 100)}"] = int(np.quantile(arr, q))
	return out


def pad_or_truncate(
	sequence: list[int],
	max_len: int,
	pad_value: int = 0,
	truncation: str = "post",
	padding: str = "post",
) -> list[int]:
	"""Apply deterministic pad/truncate operation to a tokenized sequence."""
	if max_len <= 0:
		raise ValueError("max_len must be > 0")
	if truncation not in {"pre", "post"}:
		raise ValueError("truncation must be 'pre' or 'post'")
	if padding not in {"pre", "post"}:
		raise ValueError("padding must be 'pre' or 'post'")

	seq = list(sequence)
	if len(seq) > max_len:
		seq = seq[-max_len:] if truncation == "pre" else seq[:max_len]

	if len(seq) < max_len:
		pad_len = max_len - len(seq)
		pad_chunk = [pad_value] * pad_len
		seq = pad_chunk + seq if padding == "pre" else seq + pad_chunk

	return seq


def truncate_sequence(sequence: list[int], max_len: int, strategy: str = "post") -> list[int]:
	"""Truncate sequence without padding for controlled ablations."""
	if len(sequence) <= max_len:
		return list(sequence)
	if strategy == "pre":
		return list(sequence[-max_len:])
	if strategy == "post":
		return list(sequence[:max_len])
	raise ValueError("strategy must be 'pre' or 'post'")


def pad_sequence(sequence: list[int], max_len: int, pad_value: int = 0, strategy: str = "post") -> list[int]:
	"""Pad sequence to fixed length without truncation."""
	if len(sequence) >= max_len:
		return list(sequence)
	pad_chunk = [pad_value] * (max_len - len(sequence))
	if strategy == "pre":
		return pad_chunk + list(sequence)
	if strategy == "post":
		return list(sequence) + pad_chunk
	raise ValueError("strategy must be 'pre' or 'post'")


def filter_valid_samples(df: pd.DataFrame) -> pd.DataFrame:
	"""Keep only parseable non-empty traces for modeling."""
	out = df.copy()
	if "has_parse_error" in out.columns:
		out = out[~out["has_parse_error"]]
	if "length" in out.columns:
		out = out[out["length"] > 0]
	return out.reset_index(drop=True)


def filter_invalid_sequences(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Return valid dataframe and invalid dataframe for explicit audit trails."""
	out = df.copy()
	invalid_mask = pd.Series(False, index=out.index)
	if "has_parse_error" in out.columns:
		invalid_mask = invalid_mask | out["has_parse_error"]
	if "length" in out.columns:
		invalid_mask = invalid_mask | (out["length"] <= 0)
	invalid = out[invalid_mask].copy().reset_index(drop=True)
	valid = out[~invalid_mask].copy().reset_index(drop=True)
	return valid, invalid

