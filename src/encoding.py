from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(
	sequences: Iterable[list[int]],
	min_freq: int = 1,
	max_vocab_size: int | None = None,
) -> dict[str, int]:
	"""Build syscall token vocabulary from training sequences only."""
	counter: Counter[str] = Counter()
	for seq in sequences:
		counter.update(str(tok) for tok in seq)

	items = [(tok, cnt) for tok, cnt in counter.items() if cnt >= min_freq]
	items.sort(key=lambda x: (-x[1], x[0]))
	if max_vocab_size is not None:
		items = items[:max(0, max_vocab_size - 2)]

	vocab: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
	for tok, _ in items:
		vocab[tok] = len(vocab)
	return vocab


def encode_sequence(seq: list[int], vocab: dict[str, int]) -> list[int]:
	"""Encode integer syscall sequence using token vocabulary."""
	unk = vocab[UNK_TOKEN]
	return [vocab.get(str(tok), unk) for tok in seq]


def decode_sequence(encoded: list[int], inv_vocab: dict[int, str]) -> list[str]:
	"""Decode encoded sequence to token strings for debugging."""
	return [inv_vocab.get(idx, UNK_TOKEN) for idx in encoded]


def make_bow_matrix(
	sequences: Iterable[list[int]],
	vocabulary: dict[int, int] | None = None,
) -> tuple[csr_matrix, dict[int, int]]:
	"""Create sparse bag-of-syscalls frequency matrix.

	vocabulary maps raw syscall integer -> column index.
	"""
	seq_list = list(sequences)
	if vocabulary is None:
		unique_tokens = sorted({tok for seq in seq_list for tok in seq})
		vocabulary = {tok: idx for idx, tok in enumerate(unique_tokens)}

	rows: list[int] = []
	cols: list[int] = []
	data: list[float] = []

	for ridx, seq in enumerate(seq_list):
		counter = Counter(seq)
		for tok, count in counter.items():
			if tok in vocabulary:
				rows.append(ridx)
				cols.append(vocabulary[tok])
				data.append(float(count))

	matrix = csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(len(seq_list), len(vocabulary)))
	return matrix, vocabulary


def sequences_to_documents(sequences: Iterable[list[int]]) -> list[str]:
	"""Convert integer sequences to space-separated token documents."""
	return [" ".join(str(tok) for tok in seq) for seq in sequences]


def make_ngram_count_matrix(
	sequences: Iterable[list[int]],
	ngram_range: tuple[int, int] = (2, 2),
	vocabulary: dict[str, int] | None = None,
) -> tuple[csr_matrix, dict[str, int]]:
	"""Build sparse n-gram count matrix over syscall tokens."""
	docs = sequences_to_documents(sequences)
	min_n, max_n = ngram_range
	if min_n <= 0 or max_n < min_n:
		raise ValueError("invalid ngram_range")

	if vocabulary is None:
		vocabulary = {}
		allow_new_vocab = True
	else:
		allow_new_vocab = False

	rows: list[int] = []
	cols: list[int] = []
	data: list[float] = []

	for ridx, doc in enumerate(docs):
		tokens = doc.split()
		local_counter: Counter[str] = Counter()
		for n in range(min_n, max_n + 1):
			for i in range(0, max(0, len(tokens) - n + 1)):
				ng = " ".join(tokens[i : i + n])
				local_counter[ng] += 1

		for ng, cnt in local_counter.items():
			if ng not in vocabulary:
				if allow_new_vocab:
					vocabulary[ng] = len(vocabulary)
			if ng in vocabulary:
				rows.append(ridx)
				cols.append(vocabulary[ng])
				data.append(float(cnt))

	matrix = csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(len(docs), len(vocabulary)))
	return matrix, vocabulary


def fit_tfidf_transformer(count_matrix: csr_matrix) -> TfidfTransformer:
	"""Fit TF-IDF transformer on training count matrix only."""
	transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
	transformer.fit(count_matrix)
	return transformer


def apply_tfidf(transformer: TfidfTransformer, count_matrix: csr_matrix) -> csr_matrix:
	"""Transform count matrix into TF-IDF feature space."""
	return transformer.transform(count_matrix)


def build_baseline_feature_sets(
	train_sequences: list[list[int]],
	val_sequences: list[list[int]],
	test_sequences: list[list[int]],
) -> dict[str, object]:
	"""Create robust baseline features: BoW, TF-IDF, and bigrams."""
	b_train, b_vocab = make_bow_matrix(train_sequences, vocabulary=None)
	b_val, _ = make_bow_matrix(val_sequences, vocabulary=b_vocab)
	b_test, _ = make_bow_matrix(test_sequences, vocabulary=b_vocab)

	tfidf = fit_tfidf_transformer(b_train)
	t_train = apply_tfidf(tfidf, b_train)
	t_val = apply_tfidf(tfidf, b_val)
	t_test = apply_tfidf(tfidf, b_test)

	bg_train, bg_vocab = make_ngram_count_matrix(train_sequences, ngram_range=(2, 2), vocabulary=None)
	bg_val, _ = make_ngram_count_matrix(val_sequences, ngram_range=(2, 2), vocabulary=bg_vocab)
	bg_test, _ = make_ngram_count_matrix(test_sequences, ngram_range=(2, 2), vocabulary=bg_vocab)

	return {
		"bow": {
			"train": b_train,
			"val": b_val,
			"test": b_test,
			"vocab": b_vocab,
			"fit_on": "train",
		},
		"tfidf": {
			"train": t_train,
			"val": t_val,
			"test": t_test,
			"transformer": tfidf,
			"fit_on": "train",
		},
		"bigram": {
			"train": bg_train,
			"val": bg_val,
			"test": bg_test,
			"vocab": bg_vocab,
			"fit_on": "train",
		},
		"meta": {
			"train_size": len(train_sequences),
			"val_size": len(val_sequences),
			"test_size": len(test_sequences),
			"fit_scope": "train_only",
		},
	}

