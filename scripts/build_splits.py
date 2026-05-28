from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


DEFAULT_MANIFEST = Path("data/paired_manifest.csv")
DEFAULT_OUTPUT = Path("data/splits/split_v1.csv")


def assign_splits(group_ids: list[str], train_ratio: float, val_ratio: float, seed: int) -> pd.DataFrame:
    random_generator = random.Random(seed)
    group_ids = sorted(set(group_ids))
    random_generator.shuffle(group_ids)

    n_total = len(group_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_groups = group_ids[:n_train]
    val_groups = group_ids[n_train : n_train + n_val]
    test_groups = group_ids[n_train + n_val :]

    rows = []
    for group_id in train_groups:
        rows.append({"split_group_id": group_id, "split": "train"})
    for group_id in val_groups:
        rows.append({"split_group_id": group_id, "split": "val"})
    for group_id in test_groups:
        rows.append({"split_group_id": group_id, "split": "test"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build grouped train/val/test splits")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--match-status", type=str, default="ok")
    args = parser.parse_args()

    manifest_df = pd.read_csv(args.manifest)
    eligible = manifest_df[manifest_df["match_status"] == args.match_status].copy()
    split_df = assign_splits(
        group_ids=eligible["split_group_id"].tolist(),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.output, index=False)

    print(f"Saved split -> {args.output}")
    print(split_df["split"].value_counts().to_string())
    print(f"groups={len(split_df)}")


if __name__ == "__main__":
    main()
