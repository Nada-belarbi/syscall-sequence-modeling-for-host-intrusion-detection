from __future__ import annotations

# ruff: noqa: E402

from datetime import datetime
from pathlib import Path
import json
import pickle
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loading import (
    enforce_sha1_disjoint_splits,
    split_overlap_counts,
)  # noqa: E402
from src.encoding import (
    PAD_TOKEN,
    build_baseline_feature_sets,
    build_vocab,
    encode_sequence,
)  # noqa: E402
from src.preprocessing import (
    filter_invalid_sequences,
    pad_or_truncate,
    sanitize_sequence,
)  # noqa: E402


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    metadata_root = project_root / "data" / "metadata"
    processed_root = project_root / "data" / "processed"

    # 1) Load existing protocol index files and enforce sha1-disjoint splits.
    protocol_paths = {
        "a": metadata_root / "adfa_sample_index_protocol_a.csv",
        "b": metadata_root / "adfa_sample_index_protocol_b.csv",
        "c": metadata_root / "adfa_sample_index_protocol_c.csv",
    }

    indices: dict[str, pd.DataFrame] = {}
    for key, path in protocol_paths.items():
        df = pd.read_csv(path)
        before_sha = split_overlap_counts(df, key_col="sha1")
        before_tr = split_overlap_counts(df, key_col="trace_id")
        df_fixed = enforce_sha1_disjoint_splits(df, priority=("train", "val", "test"))
        after_sha = split_overlap_counts(df_fixed, key_col="sha1")
        after_tr = split_overlap_counts(df_fixed, key_col="trace_id")

        df_fixed.to_csv(path, index=False)
        before_sha.to_csv(
            metadata_root / f"adfa_leakage_sha1_checks_protocol_{key}_before_fix.csv",
            index=False,
        )
        before_tr.to_csv(
            metadata_root
            / f"adfa_leakage_traceid_checks_protocol_{key}_before_fix.csv",
            index=False,
        )
        after_sha.to_csv(
            metadata_root / f"adfa_leakage_sha1_checks_protocol_{key}_after_fix.csv",
            index=False,
        )
        after_tr.to_csv(
            metadata_root / f"adfa_leakage_traceid_checks_protocol_{key}_after_fix.csv",
            index=False,
        )
        indices[key] = df_fixed

    # Keep compatibility names used by notebooks.
    split_overlap_counts(indices["a"], key_col="sha1").to_csv(
        metadata_root / "adfa_leakage_sha1_checks.csv", index=False
    )
    split_overlap_counts(indices["a"], key_col="trace_id").to_csv(
        metadata_root / "adfa_leakage_traceid_checks.csv", index=False
    )

    ratio_tbl = indices["a"].groupby(["split", "label"]).size().unstack(fill_value=0)
    ratio_tbl["attack_to_normal_ratio"] = ratio_tbl["attack"] / ratio_tbl[
        "normal"
    ].clip(lower=1)
    ratio_tbl["attack_pct"] = (
        ratio_tbl["attack"] / (ratio_tbl["attack"] + ratio_tbl["normal"]).clip(lower=1)
    ) * 100.0
    ratio_tbl.reset_index().to_csv(
        metadata_root / "adfa_split_ratio_protocol_a.csv", index=False
    )

    scenario_split_tbl = (
        indices["a"][indices["a"]["label"] == "attack"]
        .groupby(["attack_scenario", "split"])
        .size()
        .rename("count")
        .reset_index()
        .pivot(index="attack_scenario", columns="split", values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    scenario_split_tbl.to_csv(
        metadata_root / "adfa_attack_scenario_split_protocol_a.csv", index=False
    )

    # 2) Rebuild processed artifacts from fixed protocol A without rerunning notebooks.
    valid_df, invalid_df = filter_invalid_sequences(indices["a"])
    if valid_df.empty:
        raise RuntimeError("No valid rows after sha1 remediation")

    # Parse and sanitize sequences.
    from src.data_loading import (
        parse_syscall_file,
    )  # local import to avoid side effects at module import time

    valid_df = valid_df.copy()
    valid_df["raw_sequence"] = [
        sanitize_sequence(parse_syscall_file(Path(p), strict=True))
        for p in valid_df["filepath"]
    ]

    # Use previously selected max_len if available, else infer from data.
    max_len = 1257
    frozen_cfg_path = metadata_root / "frozen_preprocessing_config.json"
    if frozen_cfg_path.exists():
        frozen = json.loads(frozen_cfg_path.read_text(encoding="utf-8"))
        max_len = int(
            frozen.get("max_len_choice_summary", {}).get("selected_max_len", max_len)
        )

    train_seq = valid_df.loc[valid_df["split"].eq("train"), "raw_sequence"].tolist()
    val_seq = valid_df.loc[valid_df["split"].eq("val"), "raw_sequence"].tolist()
    test_seq = valid_df.loc[valid_df["split"].eq("test"), "raw_sequence"].tolist()

    vocab = build_vocab(train_seq, min_freq=1, max_vocab_size=None)
    pad_value = vocab[PAD_TOKEN]

    valid_df["encoded"] = valid_df["raw_sequence"].map(
        lambda s: encode_sequence(s, vocab)
    )
    valid_df["encoded_post"] = valid_df["encoded"].map(
        lambda s: pad_or_truncate(
            s, max_len=max_len, pad_value=pad_value, truncation="post", padding="post"
        )
    )
    valid_df["encoded_pre"] = valid_df["encoded"].map(
        lambda s: pad_or_truncate(
            s, max_len=max_len, pad_value=pad_value, truncation="pre", padding="post"
        )
    )

    valid_df["len_before_trunc"] = valid_df["encoded"].map(len)
    valid_df["len_after_post"] = valid_df["encoded_post"].map(
        lambda s: sum(1 for x in s if x != pad_value)
    )
    valid_df["len_after_pre"] = valid_df["encoded_pre"].map(
        lambda s: sum(1 for x in s if x != pad_value)
    )
    valid_df["was_truncated"] = valid_df["len_before_trunc"] > max_len
    valid_df["padding_ratio_post"] = valid_df["encoded_post"].map(
        lambda s: s.count(pad_value) / len(s) if len(s) > 0 else 0.0
    )
    valid_df["padding_ratio_pre"] = valid_df["encoded_pre"].map(
        lambda s: s.count(pad_value) / len(s) if len(s) > 0 else 0.0
    )

    features = build_baseline_feature_sets(train_seq, val_seq, test_seq)

    protocol_used = (
        str(valid_df["protocol"].mode().iloc[0])
        if "protocol" in valid_df.columns
        else "protocol_a_main"
    )
    version_tag = f"{protocol_used}_ml{max_len}_dualview_v2_sha1safe"
    version_dir = processed_root / version_tag
    version_dir.mkdir(parents=True, exist_ok=True)

    processed_index_path = version_dir / "adfa_processed_index.parquet"
    seq_post_path = version_dir / "adfa_sequences_post.npy"
    seq_pre_path = version_dir / "adfa_sequences_pre.npy"
    vocab_path = version_dir / "token_vocab.json"
    baseline_pack_path = version_dir / "baseline_features.pkl"
    invalid_path = version_dir / "invalid_traces.csv"
    feature_dims_path = version_dir / "feature_dimensions.csv"

    index_to_save = valid_df.drop(columns=["raw_sequence", "encoded"])
    index_to_save.to_parquet(processed_index_path, index=False)
    np.save(seq_post_path, np.array(valid_df["encoded_post"].tolist(), dtype=np.int32))
    np.save(seq_pre_path, np.array(valid_df["encoded_pre"].tolist(), dtype=np.int32))
    invalid_df.to_csv(invalid_path, index=False)

    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=True)
    with baseline_pack_path.open("wb") as f:
        pickle.dump(features, f)

    feature_dims_df = pd.DataFrame(
        [
            {
                "feature_set": "BoW",
                "train_shape": tuple(features["bow"]["train"].shape),
                "val_shape": tuple(features["bow"]["val"].shape),
                "test_shape": tuple(features["bow"]["test"].shape),
            },
            {
                "feature_set": "TF-IDF",
                "train_shape": tuple(features["tfidf"]["train"].shape),
                "val_shape": tuple(features["tfidf"]["val"].shape),
                "test_shape": tuple(features["tfidf"]["test"].shape),
            },
            {
                "feature_set": "Bigram",
                "train_shape": tuple(features["bigram"]["train"].shape),
                "val_shape": tuple(features["bigram"]["val"].shape),
                "test_shape": tuple(features["bigram"]["test"].shape),
            },
        ]
    )
    feature_dims_df.to_csv(feature_dims_path, index=False)

    frozen = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "protocol_used": protocol_used,
        "max_len_choice_summary": {"selected_max_len": int(max_len)},
        "truncation_choice": "dual_view_post_pre",
        "encoding_version": version_tag,
        "sha1_split_fix_applied": True,
        "sha1_overlap_after_fix": split_overlap_counts(
            indices["a"], key_col="sha1"
        ).to_dict(orient="records"),
        "traceid_overlap_after_fix": split_overlap_counts(
            indices["a"], key_col="trace_id"
        ).to_dict(orient="records"),
    }
    (metadata_root / "frozen_preprocessing_config.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    # Keep compatibility root files updated to fixed version too.
    index_to_save.to_parquet(
        processed_root / "adfa_processed_index.parquet", index=False
    )
    np.save(
        processed_root / "adfa_sequences_post.npy",
        np.array(valid_df["encoded_post"].tolist(), dtype=np.int32),
    )
    np.save(
        processed_root / "adfa_sequences_pre.npy",
        np.array(valid_df["encoded_pre"].tolist(), dtype=np.int32),
    )
    with (processed_root / "baseline_features.pkl").open("wb") as f:
        pickle.dump(features, f)

    print("Remediation completed.")
    print("Fixed protocol A/B/C indexes saved in metadata.")
    print(f"New processed version: {version_tag}")


if __name__ == "__main__":
    main()
