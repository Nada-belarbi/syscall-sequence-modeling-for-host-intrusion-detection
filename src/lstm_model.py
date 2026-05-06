from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """LSTM sequence classifier with optional bidirectionality and attention pooling."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        padding_idx: int = 0,
        bidirectional: bool = False,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        proj_size = hidden_size * self.num_directions
        if use_attention:
            self.attention_proj = nn.Linear(proj_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(proj_size, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths=lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, (h_n, _) = self.lstm(packed)

        if self.use_attention:
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            time_steps = out.size(1)
            mask = torch.arange(time_steps, device=lengths.device).unsqueeze(
                0
            ) < lengths.unsqueeze(1)
            scores = self.attention_proj(torch.tanh(out)).squeeze(-1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)
            representation = (out * weights.unsqueeze(-1)).sum(dim=1)
        else:
            if self.bidirectional:
                representation = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                representation = h_n[-1]

        logits = self.classifier(self.dropout(representation)).squeeze(-1)
        return logits
