from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .utils import sha1_file


@dataclass(frozen=True)
class AdfaPaths:
    dataset_root: Path
    attack_root: Path
    normal_train_root: Path
    normal_validation_root: Path


def resolve_adfa_paths(raw_adfa_root: Path) -> AdfaPaths:
    """Resolve canonical ADFA-LD folder paths under data/raw/adfa-ld."""
    dataset_root = raw_adfa_root / "ADFA-LD"
    attack_root = dataset_root / "Attack_Data_Master"
    normal_train_root = dataset_root / "Training_Data_Master"
    normal_validation_root = dataset_root / "Validation_Data_Master"

    required = [dataset_root, attack_root, normal_train_root, normal_validation_root]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ADFA-LD paths: {missing}")

    return AdfaPaths(
        dataset_root=dataset_root,
        attack_root=attack_root,
        normal_train_root=normal_train_root,
        normal_validation_root=normal_validation_root,
    )


def _iter_trace_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.txt")):
        if path.is_file():
            yield path


def parse_syscall_file(file_path: Path, strict: bool = True) -> list[int]:
    """Read one syscall trace file and parse it as a list of integer syscall IDs."""
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []

    tokens = text.split()
    seq: list[int] = []
    for token in tokens:
        try:
            seq.append(int(token))
        except ValueError:
            if strict:
                raise ValueError(f"Invalid token '{token}' in {file_path}")
    return seq


def _extract_attack_metadata(attack_dir_name: str) -> tuple[str | None, int | None]:
    parts = attack_dir_name.rsplit("_", 1)
    if len(parts) != 2:
        return None, None
    attack_family, scenario_str = parts[0], parts[1]
    try:
        return attack_family, int(scenario_str)
    except ValueError:
        return attack_family, None


def _make_trace_id(rel_path: str) -> str:
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()
    return f"T_{digest[:12]}"


def discover_adfa_samples(
    raw_adfa_root: Path,
    load_sequences: bool = False,
    strict: bool = True,
) -> pd.DataFrame:
    """Create a robust metadata index for ADFA-LD.

    Returned columns include:
    - trace_id
    - filepath
    - rel_path
    - source_split (train_normal, val_normal, attack)
    - split_origin (same as source split, explicit for reporting)
    - attack_family
    - attack_scenario
    - label / label_binary
    - length
    - sha1
    - is_empty / has_parse_error / parse_error
    - raw_sequence (optional)
    """
    paths = resolve_adfa_paths(raw_adfa_root)
    rows: list[dict[str, object]] = []

    def add_row(
        file_path: Path,
        source_split: str,
        attack_family: str | None,
        attack_scenario: int | None,
    ) -> None:
        rel_path = str(file_path.relative_to(paths.dataset_root))
        trace_id = _make_trace_id(rel_path)
        label = "attack" if source_split == "attack" else "normal"
        label_binary = int(label == "attack")

        parse_error = ""
        seq: list[int] = []
        length = -1
        is_empty = False
        try:
            seq = parse_syscall_file(file_path, strict=strict)
            length = len(seq)
            is_empty = length == 0
        except Exception as exc:  # noqa: BLE001
            parse_error = str(exc)

        row: dict[str, object] = {
            "trace_id": trace_id,
            "filepath": str(file_path),
            "rel_path": rel_path,
            "source_split": source_split,
            "split_origin": source_split,
            "attack_family": attack_family,
            "attack_scenario": attack_scenario,
            "label": label,
            "label_binary": label_binary,
            "length": length,
            "sha1": sha1_file(file_path),
            "is_empty": is_empty,
            "has_parse_error": bool(parse_error),
            "parse_error": parse_error,
        }
        if load_sequences:
            row["raw_sequence"] = seq
        rows.append(row)

    for fpath in _iter_trace_files(paths.normal_train_root):
        add_row(fpath, "train_normal", None, None)

    for fpath in _iter_trace_files(paths.normal_validation_root):
        add_row(fpath, "val_normal", None, None)

    for attack_dir in sorted(p for p in paths.attack_root.iterdir() if p.is_dir()):
        family, scenario = _extract_attack_metadata(attack_dir.name)
        for fpath in _iter_trace_files(attack_dir):
            add_row(fpath, "attack", family, scenario)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No ADFA-LD samples were discovered.")

    df = df.sort_values(["source_split", "rel_path"]).reset_index(drop=True)
    return df


def _stable_hash_to_unit_interval(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    value = int(digest[:16], 16)
    return value / float(16**16 - 1)


def assign_protocol_main(
    samples: pd.DataFrame, validation_normal_ratio: float = 0.5
) -> pd.DataFrame:
    """Protocol A (principal):
    - Apprentissage sur une partie des attaques (scenarios 1-7)
    - Validation sur une attaque intermediaire (scenario 8)
    - Generalisation sur attaques tenues a l ecart (scenarios 9-10)
    """
    if not 0.0 < validation_normal_ratio < 1.0:
        raise ValueError("validation_normal_ratio must be in (0, 1)")

    df = samples.copy()
    df["protocol"] = "protocol_a_main"
    df["split"] = ""

    is_normal_train = df["source_split"].eq("train_normal")
    df.loc[is_normal_train, "split"] = "train"

    is_normal_validation = df["source_split"].eq("val_normal")
    hash_values = df.loc[is_normal_validation, "rel_path"].map(
        _stable_hash_to_unit_interval
    )
    val_mask = hash_values < validation_normal_ratio
    df.loc[hash_values.index[val_mask], "split"] = "val"
    df.loc[hash_values.index[~val_mask], "split"] = "test"

    is_attack = df["label"].eq("attack")
    scenario = df["attack_scenario"].fillna(-1).astype(int)
    df.loc[is_attack & scenario.between(1, 7), "split"] = "train"
    df.loc[is_attack & scenario.eq(8), "split"] = "val"
    df.loc[is_attack & scenario.between(9, 10), "split"] = "test"

    if (df["split"] == "").any():
        raise RuntimeError("Some traces have unresolved split in protocol A.")
    return df


def assign_protocol_baseline_binary(
    samples: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Protocol B: classical binary stratified split train/val/test."""
    if not 0.0 < train_size < 1.0:
        raise ValueError("train_size must be in (0, 1)")
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be in (0, 1)")
    test_size = 1.0 - train_size - val_size
    if test_size <= 0:
        raise ValueError("train_size + val_size must be < 1")

    df = samples.copy().reset_index(drop=True)
    df["protocol"] = "protocol_b_baseline_binary"
    df["split"] = ""

    idx = np.arange(len(df))
    labels = df["label_binary"].to_numpy()

    sss_train = StratifiedShuffleSplit(
        n_splits=1, train_size=train_size, random_state=random_state
    )
    train_idx, temp_idx = next(sss_train.split(idx.reshape(-1, 1), labels))

    temp_labels = labels[temp_idx]
    val_ratio_within_temp = val_size / (val_size + test_size)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, train_size=val_ratio_within_temp, random_state=random_state
    )
    val_pos, test_pos = next(sss_val.split(temp_idx.reshape(-1, 1), temp_labels))
    val_idx = temp_idx[val_pos]
    test_idx = temp_idx[test_pos]

    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def assign_protocol_heldout_families(
    samples: pd.DataFrame, heldout_families: list[str]
) -> pd.DataFrame:
    """Protocol C: hold out full attack families from training."""
    df = samples.copy()
    df["protocol"] = "protocol_c_heldout_family"
    df["split"] = ""

    is_normal_train = df["source_split"].eq("train_normal")
    is_normal_val = df["source_split"].eq("val_normal")
    df.loc[is_normal_train, "split"] = "train"
    df.loc[is_normal_val, "split"] = "test"

    is_attack = df["label"].eq("attack")
    is_heldout_family = df["attack_family"].isin(heldout_families)
    df.loc[is_attack & ~is_heldout_family, "split"] = "train"
    df.loc[is_attack & is_heldout_family, "split"] = "test"

    # Build a validation subset from training split deterministically.
    train_mask = df["split"].eq("train")
    train_hash = df.loc[train_mask, "trace_id"].map(_stable_hash_to_unit_interval)
    val_idx = train_hash.index[train_hash < 0.15]
    df.loc[val_idx, "split"] = "val"
    return df


def build_scenario_cv_folds(samples: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Option B selected: cross-validation by held-out attack scenario.

    For fold s in [1..10]:
    - train on attack scenarios != s plus train_normal
    - validate on attack scenario s plus a deterministic subset of val_normal
    - test on remaining val_normal and optionally scenarios > s (strict mode can be added later)
    """
    folds: dict[int, pd.DataFrame] = {}
    for heldout in range(1, 11):
        df = samples.copy()
        df["protocol"] = "protocol_cv_by_scenario"
        df["fold"] = heldout
        df["split"] = ""

        is_train_normal = df["source_split"].eq("train_normal")
        is_val_normal = df["source_split"].eq("val_normal")
        is_attack = df["label"].eq("attack")
        scen = df["attack_scenario"].fillna(-1).astype(int)

        df.loc[is_train_normal, "split"] = "train"
        df.loc[is_attack & scen.ne(heldout), "split"] = "train"
        df.loc[is_attack & scen.eq(heldout), "split"] = "val"

        norm_hash = df.loc[is_val_normal, "trace_id"].map(_stable_hash_to_unit_interval)
        df.loc[norm_hash.index[norm_hash < 0.5], "split"] = "val"
        df.loc[norm_hash.index[norm_hash >= 0.5], "split"] = "test"
        folds[heldout] = df
    return folds


def compute_sequence_lengths(
    samples: pd.DataFrame, strict: bool = True
) -> pd.DataFrame:
    """Recompute lengths and parser health (if required after external edits)."""
    df = samples.copy()
    records: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        fpath = Path(row.filepath)
        try:
            seq = parse_syscall_file(fpath, strict=strict)
            records.append(
                {
                    "trace_id": row.trace_id,
                    "length": len(seq),
                    "is_empty": len(seq) == 0,
                    "has_parse_error": False,
                    "parse_error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "trace_id": row.trace_id,
                    "length": -1,
                    "is_empty": False,
                    "has_parse_error": True,
                    "parse_error": str(exc),
                }
            )
    metrics = pd.DataFrame(records)
    base_cols = [
        c
        for c in df.columns
        if c not in {"length", "is_empty", "has_parse_error", "parse_error"}
    ]
    return df[base_cols].merge(metrics, on="trace_id", how="left")


def leakage_report(index_df: pd.DataFrame) -> dict[str, object]:
    """Detect common leakage sources: duplicate traces and hash overlaps across splits."""
    report: dict[str, object] = {}
    report["duplicate_trace_id"] = int(index_df["trace_id"].duplicated().sum())
    report["duplicate_sha1"] = int(index_df["sha1"].duplicated().sum())

    if "split" in index_df.columns:
        hash_split = index_df.groupby("sha1")["split"].nunique(dropna=False)
        report["sha1_shared_across_splits"] = int((hash_split > 1).sum())

    if "attack_family" in index_df.columns:
        family_split = (
            index_df.groupby(["split", "attack_family"])
            .size()
            .reset_index(name="count")
            if "split" in index_df.columns
            else pd.DataFrame()
        )
        report["family_split_table"] = (
            family_split.to_dict(orient="records") if not family_split.empty else []
        )

    return report


def split_overlap_counts(index_df: pd.DataFrame, key_col: str = "sha1") -> pd.DataFrame:
    """Compute pairwise overlap counts for a key across train/val/test splits."""
    if "split" not in index_df.columns:
        raise ValueError("index_df must contain a 'split' column")
    splits = ["train", "val", "test"]
    rows: list[dict[str, object]] = []
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            set_a = set(
                index_df.loc[index_df["split"].eq(a), key_col].dropna().tolist()
            )
            set_b = set(
                index_df.loc[index_df["split"].eq(b), key_col].dropna().tolist()
            )
            rows.append(
                {
                    "pair": f"{a}-{b}",
                    f"{key_col}_overlap_count": len(set_a.intersection(set_b)),
                }
            )
    return pd.DataFrame(rows)


def enforce_sha1_disjoint_splits(
    index_df: pd.DataFrame,
    priority: tuple[str, str, str] = ("train", "val", "test"),
) -> pd.DataFrame:
    """Force all identical-content traces (same sha1) to a single split.

    This post-hoc safety pass prevents cross-split leakage when identical traces exist
    under multiple filenames in ADFA-LD.
    """
    required = {"sha1", "split"}
    missing = [c for c in required if c not in index_df.columns]
    if missing:
        raise ValueError(f"index_df missing required columns: {missing}")

    priority_rank = {name: i for i, name in enumerate(priority)}
    df = index_df.copy()
    original_split = df["split"].astype(str).copy()

    chosen_split_by_sha1: dict[str, str] = {}
    for sha1, group in df.groupby("sha1", sort=False):
        group_splits = [
            s
            for s in group["split"].dropna().astype(str).tolist()
            if s in priority_rank
        ]
        if not group_splits:
            continue
        chosen = sorted(group_splits, key=lambda s: priority_rank[s])[0]
        chosen_split_by_sha1[str(sha1)] = chosen

    df["split"] = df["sha1"].astype(str).map(chosen_split_by_sha1).fillna(df["split"])
    df["split_sha1_adjusted"] = original_split.ne(df["split"])
    df["split_sha1_adjustment_policy"] = f"priority:{'>' .join(priority)}"
    return df


def dataset_audit_report(index_df: pd.DataFrame) -> dict[str, object]:
    """Build a compact audit report dictionary for metadata export."""
    result: dict[str, object] = {}
    result["num_samples"] = int(len(index_df))
    result["num_by_label"] = index_df["label"].value_counts(dropna=False).to_dict()
    result["num_by_source_split"] = (
        index_df["source_split"].value_counts(dropna=False).to_dict()
    )
    if "split" in index_df.columns:
        result["num_by_split"] = index_df["split"].value_counts(dropna=False).to_dict()
    if "attack_family" in index_df.columns:
        fam = index_df[index_df["label"] == "attack"]["attack_family"].value_counts(
            dropna=False
        )
        result["num_by_attack_family"] = fam.to_dict()

    if "length" in index_df.columns:
        valid_lengths = index_df.loc[index_df["length"] >= 0, "length"]
        if not valid_lengths.empty:
            quantiles = np.quantile(valid_lengths, [0.5, 0.9, 0.95, 0.99]).tolist()
            result["length_stats"] = {
                "min": int(valid_lengths.min()),
                "max": int(valid_lengths.max()),
                "mean": float(valid_lengths.mean()),
                "median": float(quantiles[0]),
                "p90": float(quantiles[1]),
                "p95": float(quantiles[2]),
                "p99": float(quantiles[3]),
            }

    result["num_empty_sequences"] = (
        int(index_df["is_empty"].sum()) if "is_empty" in index_df.columns else 0
    )
    result["num_parse_errors"] = (
        int(index_df["has_parse_error"].sum())
        if "has_parse_error" in index_df.columns
        else 0
    )
    result["leakage_checks"] = leakage_report(index_df)
    return result


def assign_research_split(
    samples: pd.DataFrame, validation_normal_ratio: float = 0.5
) -> pd.DataFrame:
    """Backward-compatible alias for the principal protocol."""
    return assign_protocol_main(
        samples=samples, validation_normal_ratio=validation_normal_ratio
    )
