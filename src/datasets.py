from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class SyscallSequenceDataset(Dataset):
	"""PyTorch dataset for encoded syscall sequences."""

	def __init__(
		self,
		encoded_sequences: list[list[int]],
		labels: list[int],
		sample_ids: list[str] | None = None,
		metadata: list[dict[str, object]] | None = None,
		return_metadata: bool = True,
	) -> None:
		if len(encoded_sequences) != len(labels):
			raise ValueError("encoded_sequences and labels must have the same length")
		self.encoded_sequences = encoded_sequences
		self.labels = labels
		self.sample_ids = sample_ids if sample_ids is not None else [str(i) for i in range(len(labels))]
		self.metadata = metadata if metadata is not None else [{} for _ in range(len(labels))]
		self.return_metadata = return_metadata
		if len(self.metadata) != len(self.labels):
			raise ValueError("metadata must have the same length as labels")

	def __len__(self) -> int:
		return len(self.labels)

	def __getitem__(self, idx: int) -> dict[str, object]:
		seq = self.encoded_sequences[idx]
		item = {
			"input_ids": torch.tensor(seq, dtype=torch.long),
			"label": torch.tensor(self.labels[idx], dtype=torch.long),
			"length": torch.tensor(len(seq), dtype=torch.long),
			"sample_id": self.sample_ids[idx],
		}
		if self.return_metadata:
			item["metadata"] = self.metadata[idx]
		return item


@dataclass(frozen=True)
class CollateConfig:
	pad_value: int = 0
	sort_by_length: bool = False


def collate_padded_batch(
	batch: list[dict[str, object]],
	pad_value: int = 0,
	sort_by_length: bool = False,
) -> dict[str, object]:
	"""Pad variable-length sequences in a batch while preserving lengths."""
	if sort_by_length:
		batch = sorted(batch, key=lambda x: int(x["length"]), reverse=True)

	input_ids = [item["input_ids"] for item in batch]
	labels = torch.stack([item["label"] for item in batch])
	lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
	sample_ids = [str(item["sample_id"]) for item in batch]
	metadata = [item.get("metadata", {}) for item in batch]

	padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_value)
	padding_mask = padded.eq(pad_value)

	return {
		"input_ids": padded,
		"labels": labels,
		"lengths": lengths,
		"sample_ids": sample_ids,
		"padding_mask": padding_mask,
		"metadata": metadata,
	}


def make_dataloader(
	dataset: SyscallSequenceDataset,
	batch_size: int,
	shuffle: bool,
	pad_value: int = 0,
	sort_by_length: bool = False,
	num_workers: int = 0,
	sampler: Any | None = None,
) -> DataLoader:
	"""Create a DataLoader with dynamic batch padding."""
	if sampler is not None and shuffle:
		raise ValueError("Cannot set shuffle=True when sampler is provided")
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle if sampler is None else False,
		sampler=sampler,
		num_workers=num_workers,
		collate_fn=lambda b: collate_padded_batch(b, pad_value=pad_value, sort_by_length=sort_by_length),
	)

