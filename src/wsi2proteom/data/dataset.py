from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PairedSlideDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        proteome_pickle_path: str | Path,
        split_path: str | Path | None = None,
        split_name: str | None = None,
        feature_model: str | None = None,
        match_status: str = "ok",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.proteome_pickle_path = Path(proteome_pickle_path)
        self.split_path = Path(split_path) if split_path is not None else None
        self.split_name = split_name
        self.feature_model = feature_model
        self.match_status = match_status

        self.manifest = pd.read_csv(self.manifest_path)
        if self.match_status is not None:
            self.manifest = self.manifest[self.manifest["match_status"] == self.match_status].copy()

        if self.feature_model is not None:
            self.manifest = self.manifest[self.manifest["feature_model"] == self.feature_model].copy()

        if self.split_path is not None and self.split_name is not None:
            split_df = pd.read_csv(self.split_path)
            allowed = set(split_df.loc[split_df["split"] == self.split_name, "split_group_id"])
            self.manifest = self.manifest[self.manifest["split_group_id"].isin(allowed)].copy()

        self.manifest = self.manifest.reset_index(drop=True)

        with open(self.proteome_pickle_path, "rb") as f:
            self.targets: dict[str, np.ndarray] = pickle.load(f)

    def __len__(self) -> int:
        return len(self.manifest)

    @property
    def out_dim(self) -> int:
        sample_key = next(iter(self.targets))
        return int(self.targets[sample_key].shape[0])

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[index]
        feature_path = Path(row["feature_path"])
        feature_data = torch.load(feature_path, map_location="cpu")
        features = feature_data["features"].float()

        aliquot_id = row["aliquot_submitter_id"]
        target_np = self.targets[aliquot_id].astype(np.float32, copy=False)
        target = torch.from_numpy(target_np)

        return {
            "slide_id": row["slide_id"],
            "case_submitter_id": row["case_submitter_id"],
            "sample_submitter_id": row["sample_submitter_id"],
            "aliquot_submitter_id": aliquot_id,
            "feature_model": row["feature_model"],
            "features": features,
            "target": target,
        }


def collate_paired_samples(batch: list[dict]) -> dict:
    return {
        "slide_id": [item["slide_id"] for item in batch],
        "case_submitter_id": [item["case_submitter_id"] for item in batch],
        "sample_submitter_id": [item["sample_submitter_id"] for item in batch],
        "aliquot_submitter_id": [item["aliquot_submitter_id"] for item in batch],
        "feature_model": [item["feature_model"] for item in batch],
        "features": [item["features"] for item in batch],
        "target": torch.stack([item["target"] for item in batch], dim=0),
    }
