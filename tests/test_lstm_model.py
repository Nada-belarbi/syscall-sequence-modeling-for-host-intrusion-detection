"""Unit tests for src/lstm_model.py"""
from __future__ import annotations

import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lstm_model import LSTMClassifier


def _make_batch(batch_size: int = 4, seq_len: int = 10, vocab_size: int = 20):
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    lengths   = torch.randint(3, seq_len + 1, (batch_size,))
    lengths[0] = seq_len  # ensure at least one full-length sequence
    return input_ids, lengths


def test_forward_unidirectional():
    model = LSTMClassifier(vocab_size=20, embedding_dim=8, hidden_size=16,
                           num_layers=1, bidirectional=False, use_attention=False)
    model.eval()
    ids, lengths = _make_batch()
    with torch.no_grad():
        out = model(ids, lengths)
    assert out.shape == torch.Size([4])


def test_forward_bidirectional_attention():
    model = LSTMClassifier(vocab_size=20, embedding_dim=8, hidden_size=16,
                           num_layers=2, bidirectional=True, use_attention=True)
    model.eval()
    ids, lengths = _make_batch()
    with torch.no_grad():
        out = model(ids, lengths)
    assert out.shape == torch.Size([4])


def test_invalid_vocab_size():
    with pytest.raises(ValueError, match="vocab_size"):
        LSTMClassifier(vocab_size=0, embedding_dim=8, hidden_size=16)


def test_output_dtype():
    model = LSTMClassifier(vocab_size=20, embedding_dim=8, hidden_size=16)
    ids, lengths = _make_batch()
    with torch.no_grad():
        out = model(ids, lengths)
    assert out.dtype in (torch.float32, torch.float16)


def test_batch_size_one():
    model = LSTMClassifier(vocab_size=20, embedding_dim=8, hidden_size=16,
                           bidirectional=True, use_attention=True)
    model.eval()
    ids, lengths = _make_batch(batch_size=1)
    with torch.no_grad():
        out = model(ids, lengths)
    assert out.shape == torch.Size([1])
