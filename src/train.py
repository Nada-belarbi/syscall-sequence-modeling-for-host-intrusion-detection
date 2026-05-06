from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from .metrics import compute_binary_metrics
from .utils import ensure_dir


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float | None = None,
    decision_threshold: float = 0.5,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []

    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].float().to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, lengths=lengths)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= decision_threshold).long()

        y_true.extend(labels.long().detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_score.extend(probs.detach().cpu().tolist())

    epoch_loss = running_loss / max(1, len(loader.dataset))
    metrics = compute_binary_metrics(
        np.array(y_true), np.array(y_pred), np.array(y_score)
    )
    metrics["loss"] = float(epoch_loss)
    return metrics


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    decision_threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].float().to(device)

        logits = model(input_ids=input_ids, lengths=lengths)
        loss = criterion(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= decision_threshold).long()

        y_true.extend(labels.long().detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_score.extend(probs.detach().cpu().tolist())

    epoch_loss = running_loss / max(1, len(loader.dataset))
    metrics = compute_binary_metrics(
        np.array(y_true), np.array(y_pred), np.array(y_score)
    )
    metrics["loss"] = float(epoch_loss)
    return metrics


def fit_lstm(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    checkpoint_path: Path,
    grad_clip: float | None = 1.0,
    early_stopping_patience: int | None = 4,
    pos_weight: float | None = None,
    decision_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train LSTM and save the best model based on validation F1."""
    ensure_dir(checkpoint_path.parent)
    model = model.to(device)
    if pos_weight is not None and pos_weight > 0:
        pw = torch.tensor(float(pos_weight), dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            grad_clip=grad_clip,
            decision_threshold=decision_threshold,
        )
        val_metrics = evaluate_one_epoch(
            model,
            val_loader,
            criterion,
            device=device,
            decision_threshold=decision_threshold,
        )

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        val_f1 = float(val_metrics["f1"])
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        if (
            early_stopping_patience is not None
            and patience_counter >= early_stopping_patience
        ):
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    best_summary = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "checkpoint_path": str(checkpoint_path),
        "decision_threshold": float(decision_threshold),
        "pos_weight": float(pos_weight) if pos_weight is not None else None,
    }
    return history, best_summary
