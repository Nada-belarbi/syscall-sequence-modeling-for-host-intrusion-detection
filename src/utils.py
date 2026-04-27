from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
METADATA_ROOT = DATA_ROOT / "metadata"
RESULTS_ROOT = PROJECT_ROOT / "results"
LOGS_ROOT = RESULTS_ROOT / "logs"


def get_logger(name: str = "syscall_ids", level: int = logging.INFO) -> logging.Logger:
	"""Create a console logger with a consistent format."""
	logger = logging.getLogger(name)
	if not logger.handlers:
		handler = logging.StreamHandler()
		formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
		handler.setFormatter(formatter)
		logger.addHandler(handler)
	logger.setLevel(level)
	return logger


def ensure_dir(path: Path) -> Path:
	"""Create the target directory if missing and return it."""
	path.mkdir(parents=True, exist_ok=True)
	return path


def set_seed(seed: int = 42) -> None:
	"""Set random seeds for reproducibility across Python and NumPy."""
	random.seed(seed)
	np.random.seed(seed)


def save_json(data: dict[str, Any], output_path: Path, indent: int = 2) -> None:
	"""Persist a dictionary as JSON with UTF-8 encoding."""
	ensure_dir(output_path.parent)
	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=indent, ensure_ascii=True)


def load_yaml(config_path: Path) -> dict[str, Any]:
	"""Load a YAML configuration file into a dictionary."""
	with config_path.open("r", encoding="utf-8") as handle:
		return yaml.safe_load(handle)


def save_yaml(data: dict[str, Any], output_path: Path) -> None:
	"""Persist a dictionary as YAML."""
	ensure_dir(output_path.parent)
	with output_path.open("w", encoding="utf-8") as handle:
		yaml.safe_dump(data, handle, sort_keys=False)


def sha1_text(text: str) -> str:
	"""Compute SHA1 fingerprint from text content."""
	return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def sha1_file(file_path: Path) -> str:
	"""Compute SHA1 fingerprint from a file."""
	hasher = hashlib.sha1()
	with file_path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(8192), b""):
			hasher.update(chunk)
	return hasher.hexdigest()


@contextmanager
def timer(label: str):
	"""Simple timing context manager for notebook/script instrumentation."""
	start = time.perf_counter()
	yield
	elapsed = time.perf_counter() - start
	print(f"[timer] {label}: {elapsed:.3f}s")


def ensure_experiment_log(log_path: Path | None = None) -> Path:
	"""Create a run log CSV with header if it does not exist."""
	if log_path is None:
		log_path = LOGS_ROOT / "experiments.csv"
	ensure_dir(log_path.parent)
	if not log_path.exists():
		with log_path.open("w", encoding="utf-8", newline="") as handle:
			writer = csv.writer(handle)
			writer.writerow(["timestamp", "run_name", "protocol", "split", "model", "params", "metrics"])
	return log_path


def append_experiment_log(
	run_name: str,
	protocol: str,
	split: str,
	model: str,
	params: dict[str, Any],
	metrics: dict[str, Any],
	log_path: Path | None = None,
) -> Path:
	"""Append one experiment row to results/logs/experiments.csv."""
	if log_path is None:
		log_path = ensure_experiment_log()
	else:
		ensure_experiment_log(log_path)
	with log_path.open("a", encoding="utf-8", newline="") as handle:
		writer = csv.writer(handle)
		writer.writerow(
			[
				time.strftime("%Y-%m-%d %H:%M:%S"),
				run_name,
				protocol,
				split,
				model,
				json.dumps(params, ensure_ascii=True),
				json.dumps(metrics, ensure_ascii=True),
			]
		)
	return log_path

