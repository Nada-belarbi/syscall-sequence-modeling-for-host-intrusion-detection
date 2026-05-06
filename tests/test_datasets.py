"""Unit tests for src/datasets.py"""
from __future__ import annotations

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import SyscallSequenceDataset, make_dataloader


_SEQS   = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10]]
_LABELS = [0, 1, 1, 0]
_IDS    = ["t1", "t2", "t3", "t4"]


def test_dataset_length():
    ds = SyscallSequenceDataset(_SEQS, _LABELS, _IDS)
    assert len(ds) == 4


def test_dataset_item():
    ds = SyscallSequenceDataset(_SEQS, _LABELS, _IDS)
    item = ds[0]
    assert "input_ids" in item
    assert "label" in item
    assert item["label"].item() == 0


def test_dataloader_batch_shape():
    ds = SyscallSequenceDataset(_SEQS, _LABELS, _IDS)
    loader = make_dataloader(ds, batch_size=4, shuffle=False, pad_value=0)
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 4        # batch size
    assert batch["input_ids"].shape[1] == max(len(s) for s in _SEQS)  # padded to max


def test_dataloader_padding_value():
    ds = SyscallSequenceDataset(_SEQS, _LABELS, _IDS)
    loader = make_dataloader(ds, batch_size=4, shuffle=False, pad_value=0)
    batch = next(iter(loader))
    # The shortest sequence (length 1) should have trailing zeros
    lengths = batch["lengths"]
    min_len = lengths.min().item()
    shortest_idx = lengths.argmin().item()
    padded_row = batch["input_ids"][shortest_idx]
    assert padded_row[min_len:].sum().item() == 0


def test_label_tensor_dtype():
    ds = SyscallSequenceDataset(_SEQS, _LABELS, _IDS)
    item = ds[0]
    assert item["label"].dtype == torch.float32 or item["label"].dtype == torch.long
