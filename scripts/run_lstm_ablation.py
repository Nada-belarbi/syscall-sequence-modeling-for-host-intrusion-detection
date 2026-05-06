from __future__ import annotations

# ruff: noqa: E402

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.datasets import SyscallSequenceDataset, make_dataloader  # noqa: E402
from src.evaluate import build_prediction_frame, evaluate_lstm_classifier  # noqa: E402
from src.lstm_model import LSTMClassifier  # noqa: E402
from src.metrics import save_confusion_matrix_plot  # noqa: E402
from src.train import fit_lstm  # noqa: E402
from src.utils import (
    RESULTS_ROOT,
    PROCESSED_ROOT,
    append_experiment_log,
    ensure_dir,
    set_seed,
)  # noqa: E402


def build_split_pack(
    index_df: pd.DataFrame, seq_post: np.ndarray, seq_pre: np.ndarray, view_name: str
) -> dict[str, dict[str, object]]:
    if view_name == "post":
        seq_array = seq_post
    elif view_name == "pre":
        seq_array = seq_pre
    else:
        raise ValueError(view_name)

    df = index_df.reset_index(drop=True).copy()
    if view_name == "post":
        effective_len_col = "len_after_post"
    else:
        effective_len_col = "len_after_pre"
    if effective_len_col not in df.columns:
        raise ValueError(
            f"Missing required column for variable-length reconstruction: {effective_len_col}"
        )

    lengths = df[effective_len_col].fillna(0).astype(int).tolist()
    encoded = []
    for arr, eff_len in zip(seq_array, lengths):
        k = max(1, int(eff_len))
        encoded.append(arr[:k].tolist())
    df["encoded_view"] = encoded

    split_pack: dict[str, dict[str, object]] = {}
    for split in ["train", "val", "test"]:
        sdf = df[df["split"] == split].reset_index(drop=True)
        split_pack[split] = {
            "x": sdf["encoded_view"].tolist(),
            "y": sdf["label_binary"].astype(int).tolist(),
            "trace_ids": sdf["trace_id"].astype(str).tolist(),
        }
    return split_pack


def make_loaders(
    split_pack: dict[str, dict[str, object]],
    pad_value: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, torch.utils.data.DataLoader]:
    loaders: dict[str, torch.utils.data.DataLoader] = {}
    for split in ["train", "val", "test"]:
        ds = SyscallSequenceDataset(
            encoded_sequences=split_pack[split]["x"],
            labels=split_pack[split]["y"],
            sample_ids=split_pack[split]["trace_ids"],
            return_metadata=False,
        )
        loaders[split] = make_dataloader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            pad_value=pad_value,
            sort_by_length=False,
            num_workers=num_workers,
        )
    return loaders


def _subset_split_block(
    block: dict[str, object], cap: int, rng: np.random.Generator
) -> dict[str, object]:
    n = len(block["y"])
    if n <= cap:
        return block
    idx = np.sort(rng.choice(n, size=cap, replace=False))
    return {
        "x": [block["x"][int(i)] for i in idx],
        "y": [block["y"][int(i)] for i in idx],
        "trace_ids": [block["trace_ids"][int(i)] for i in idx],
    }


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")

    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("project", {}).get("seed", 42))
    protocol_name = str(
        cfg.get("protocols", {}).get("main", {}).get("name", "protocol_a_main")
    )
    set_seed(seed)
    torch.manual_seed(seed)

    version_dir = PROCESSED_ROOT / "protocol_a_main_ml1231_dualview_v1"
    index_path = version_dir / "adfa_processed_index.parquet"
    seq_post_path = version_dir / "adfa_sequences_post.npy"
    seq_pre_path = version_dir / "adfa_sequences_pre.npy"

    if (
        not index_path.exists()
        or not seq_post_path.exists()
        or not seq_pre_path.exists()
    ):
        raise FileNotFoundError("Missing processed inputs for LSTM ablation")

    index_df = pd.read_parquet(index_path)
    seq_post = np.load(seq_post_path)
    seq_pre = np.load(seq_pre_path)

    vocab_size = int(max(int(seq_post.max()), int(seq_pre.max())) + 1)
    pad_value = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_cfg = cfg.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 64))
    num_workers = int(training_cfg.get("num_workers", 0))

    lstm_cfg = cfg.get("lstm", {})
    views = lstm_cfg.get("views", ["post", "pre"])
    ablation_cfg = lstm_cfg.get("ablation", {})
    emb_dims = ablation_cfg.get("embedding_dims", [64, 128])
    hid_sizes = ablation_cfg.get("hidden_sizes", [128, 256])
    dropouts = ablation_cfg.get("dropouts", [0.1, 0.3, 0.5])
    ablation_configs = [
        {"embedding_dim": e, "hidden_size": h, "dropout": d}
        for e in emb_dims
        for h in hid_sizes
        for d in dropouts
    ]

    model_cfg = lstm_cfg.get("model", {})
    bidirectional = bool(model_cfg.get("bidirectional", True))
    use_attention = bool(model_cfg.get("use_attention", True))
    num_layers_cfg = int(model_cfg.get("num_layers", 2))

    num_epochs = int(lstm_cfg.get("num_epochs", 20))
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))

    ckpt_dir = ensure_dir(RESULTS_ROOT / "logs" / "checkpoints")
    tables_dir = ensure_dir(RESULTS_ROOT / "tables")
    cm_dir = ensure_dir(RESULTS_ROOT / "confusion_matrices")
    fig_dir = ensure_dir(PROJECT_ROOT / "reports" / "figures")

    rows: list[dict[str, object]] = []
    history_rows: list[pd.DataFrame] = []
    best_run: dict[str, object] | None = None
    best_val_f1 = -1.0

    subset_rng = np.random.default_rng(seed)
    search_train_cap = 512
    search_val_cap = 512
    search_test_cap = 512

    for view_name in views:
        for cfg_run in ablation_configs:
            emb_dim = int(cfg_run["embedding_dim"])
            hid_size = int(cfg_run["hidden_size"])
            dropout = float(cfg_run["dropout"])
            split_pack = build_split_pack(
                index_df=index_df,
                seq_post=seq_post,
                seq_pre=seq_pre,
                view_name=view_name,
            )

            search_pack = {
                "train": _subset_split_block(
                    split_pack["train"], cap=search_train_cap, rng=subset_rng
                ),
                "val": _subset_split_block(
                    split_pack["val"], cap=search_val_cap, rng=subset_rng
                ),
                "test": _subset_split_block(
                    split_pack["test"], cap=search_test_cap, rng=subset_rng
                ),
            }
            loaders = make_loaders(
                split_pack=search_pack,
                pad_value=pad_value,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            run_name = f"lstm_{view_name}_e{emb_dim}_h{hid_size}_d{dropout}"
            ckpt_path = ckpt_dir / f"{run_name}.pt"

            model = LSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=emb_dim,
                hidden_size=hid_size,
                num_layers=num_layers_cfg,
                dropout=dropout,
                padding_idx=pad_value,
                bidirectional=bidirectional,
                use_attention=use_attention,
            )

            history, _ = fit_lstm(
                model=model,
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                num_epochs=num_epochs,
                lr=lr,
                weight_decay=weight_decay,
                device=device,
                checkpoint_path=ckpt_path,
                grad_clip=float(training_cfg.get("grad_clip", 1.0)),
                early_stopping_patience=int(
                    training_cfg.get("early_stopping_patience", 5)
                ),
            )
            if history:
                history_rows.append(pd.DataFrame(history).assign(run_name=run_name))

            val_eval = evaluate_lstm_classifier(model, loaders["val"], device=device)
            test_eval = evaluate_lstm_classifier(model, loaders["test"], device=device)

            rows.append(
                {
                    "model": run_name,
                    "algorithm": "lstm",
                    "feature_set": f"sequence_{view_name}",
                    "split": "val",
                    "view": view_name,
                    "embedding_dim": emb_dim,
                    "hidden_size": hid_size,
                    "dropout": dropout,
                    **val_eval["metrics"],
                }
            )
            rows.append(
                {
                    "model": run_name,
                    "algorithm": "lstm",
                    "feature_set": f"sequence_{view_name}",
                    "split": "test",
                    "view": view_name,
                    "embedding_dim": emb_dim,
                    "hidden_size": hid_size,
                    "dropout": dropout,
                    **test_eval["metrics"],
                }
            )

            run_params = {
                "view": view_name,
                "embedding_dim": emb_dim,
                "hidden_size": hid_size,
                "dropout": dropout,
                "num_layers": num_layers_cfg,
                "bidirectional": bidirectional,
                "use_attention": use_attention,
                "num_epochs": num_epochs,
                "lr": lr,
                "weight_decay": weight_decay,
            }
            append_experiment_log(
                run_name=run_name,
                protocol=protocol_name,
                split="val",
                model="lstm",
                params=run_params,
                metrics=val_eval["metrics"],
            )
            append_experiment_log(
                run_name=run_name,
                protocol=protocol_name,
                split="test",
                model="lstm",
                params=run_params,
                metrics=test_eval["metrics"],
            )

            val_f1 = float(val_eval["metrics"]["f1"])
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_run = {
                    "run_name": run_name,
                    "view": view_name,
                    "embedding_dim": emb_dim,
                    "hidden_size": hid_size,
                    "dropout": dropout,
                    "checkpoint_path": str(ckpt_path),
                    "val_eval": val_eval,
                    "test_eval_search": test_eval,
                    "view_for_best": view_name,
                }

            print(
                f"[done] {run_name} val_f1={val_f1:.4f} test_f1={float(test_eval['metrics']['f1']):.4f}"
            )

    if best_run is None:
        raise RuntimeError("No LSTM run completed")

    best_split_pack = build_split_pack(
        index_df=index_df,
        seq_post=seq_post,
        seq_pre=seq_pre,
        view_name=str(best_run["view_for_best"]),
    )
    best_full_loaders = make_loaders(
        split_pack=best_split_pack,
        pad_value=pad_value,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    best_model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=int(best_run["embedding_dim"]),
        hidden_size=int(best_run["hidden_size"]),
        num_layers=num_layers_cfg,
        dropout=float(best_run["dropout"]),
        padding_idx=pad_value,
        bidirectional=bidirectional,
        use_attention=use_attention,
    )
    best_ckpt = torch.load(best_run["checkpoint_path"], map_location=device)
    best_model.load_state_dict(best_ckpt["model_state_dict"])
    best_model = best_model.to(device)
    best_model.eval()

    best_val_eval_full = evaluate_lstm_classifier(
        best_model, best_full_loaders["val"], device=device
    )
    best_test_eval_full = evaluate_lstm_classifier(
        best_model, best_full_loaders["test"], device=device
    )
    best_run["val_eval"] = best_val_eval_full
    best_run["test_eval"] = best_test_eval_full
    best_run["test_trace_ids"] = best_full_loaders["test"].dataset.sample_ids

    ablation_df = (
        pd.DataFrame(rows)
        .sort_values(["split", "f1"], ascending=[True, False])
        .reset_index(drop=True)
    )
    ablation_path = tables_dir / "lstm_ablation_results.csv"
    ablation_df.to_csv(ablation_path, index=False)

    history_df = (
        pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    )
    history_path = tables_dir / "lstm_training_history.csv"
    history_df.to_csv(history_path, index=False)

    pred_df = build_prediction_frame(
        trace_ids=[str(x) for x in best_run["test_trace_ids"]],
        y_true=best_run["test_eval"]["y_true"],
        y_pred=best_run["test_eval"]["y_pred"],
        y_score=best_run["test_eval"]["y_score"],
    )
    pred_df["model"] = str(best_run["run_name"])
    pred_path = tables_dir / "predictions_lstm_test.csv"
    pred_df.to_csv(pred_path, index=False)

    cm_path = cm_dir / f"lstm_{best_run['run_name']}_test.png"
    save_confusion_matrix_plot(
        y_true=best_run["test_eval"]["y_true"],
        y_pred=best_run["test_eval"]["y_pred"],
        output_path=cm_path,
        title=f"LSTM {best_run['run_name']} - Test Confusion Matrix",
    )

    best_summary_row = {
        "model": str(best_run["run_name"]),
        "algorithm": "lstm",
        "feature_set": f"sequence_{best_run['view']}",
        "view": str(best_run["view"]),
        "embedding_dim": int(best_run["embedding_dim"]),
        "hidden_size": int(best_run["hidden_size"]),
        "dropout": float(best_run["dropout"]),
        "checkpoint_path": str(
            Path(best_run["checkpoint_path"]).relative_to(PROJECT_ROOT)
        ),
    }
    best_summary_row.update(
        {f"val_{k}": v for k, v in best_run["val_eval"]["metrics"].items()}
    )
    best_summary_row.update(
        {f"test_{k}": v for k, v in best_run["test_eval"]["metrics"].items()}
    )
    best_summary_path = tables_dir / "lstm_best_run_summary.csv"
    pd.DataFrame([best_summary_row]).to_csv(best_summary_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_val = (
        ablation_df[ablation_df["split"] == "val"]
        .sort_values("f1", ascending=False)
        .head(8)
    )
    sns.barplot(data=plot_val, x="model", y="f1", hue="view", ax=axes[0])
    axes[0].set_title("Top LSTM runs on validation (F1)")
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].set_ylim(0, 1)

    plot_test = (
        ablation_df[ablation_df["split"] == "test"]
        .sort_values("f1", ascending=False)
        .head(8)
    )
    sns.barplot(data=plot_test, x="model", y="f1", hue="view", ax=axes[1])
    axes[1].set_title("Top LSTM runs on test (F1)")
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    lstm_fig_path = fig_dir / "lstm_ablation_f1_top8.png"
    fig.savefig(lstm_fig_path, dpi=150)
    plt.close(fig)

    print("[saved]", ablation_path)
    print("[saved]", history_path)
    print("[saved]", pred_path)
    print("[saved]", cm_path)
    print("[saved]", best_summary_path)
    print("[saved]", lstm_fig_path)


if __name__ == "__main__":
    main()
