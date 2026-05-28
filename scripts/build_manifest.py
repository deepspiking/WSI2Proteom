from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch


DEFAULT_FEATURES_ROOT = Path("/data/workspace/ai2bio/data/CPTAC-BRCA_v1_feat/BRCA")
DEFAULT_BIOSPECIMEN_MANIFEST = Path("/data/workspace/ai2bio/data/CPTAC-BRCA_v1/PDC_biospecimen_manifest_05242026_175318.tsv")
DEFAULT_PROTEOME_SELECTION = Path("/data/workspace/ai2bio/data/CPTAC-BRCA_v1/CPTAC2_Breast_Prospective_Collection_BI_Proteome_unshared_vectors.selection.csv")
DEFAULT_OUTPUT = Path("data/paired_manifest.csv")


def parse_slide_id(slide_id: str) -> tuple[str, str]:
    case_id, sample_id = slide_id.split("-", 1)
    return case_id, sample_id


def infer_feature_dim(feature_path: Path) -> int:
    data = torch.load(feature_path, map_location="cpu")
    features = data["features"]
    return int(features.shape[1]) if features.ndim > 1 else 1


def extract_slide_id_from_feature_path(feature_path: Path, model_name: str) -> str:
    suffix = f".{model_name}"
    stem = feature_path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def resolve_feature_path(model_dir: Path, slide_id: str, model_name: str) -> Path | None:
    candidates = [
        model_dir / f"{slide_id}.pt",
        model_dir / f"{slide_id}.{model_name}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_rows(features_root: Path, biospecimen_df: pd.DataFrame, selection_df: pd.DataFrame, models: list[str]) -> list[dict]:
    selected_aliquots = set(selection_df["aliquot_id"])
    rows: list[dict] = []

    for model_name in models:
        model_dir = features_root / model_name
        if not model_dir.exists():
            continue

        feature_files = sorted(model_dir.glob("*.pt"))
        for feature_file in feature_files:
            slide_id = extract_slide_id_from_feature_path(feature_file, model_name)
            case_id, sample_id = parse_slide_id(slide_id)

            matched = biospecimen_df[
                (biospecimen_df["Case Submitter ID"] == case_id)
                & (biospecimen_df["Sample Submitter ID"] == sample_id)
            ].copy()

            matched_selected = matched[matched["Aliquot Submitter ID"].isin(selected_aliquots)].copy()

            if matched.empty:
                rows.append(
                    {
                        "slide_id": slide_id,
                        "case_submitter_id": case_id,
                        "sample_submitter_id": sample_id,
                        "aliquot_submitter_id": "",
                        "feature_model": model_name,
                        "feature_path": str(feature_file.resolve()),
                        "feature_dim": infer_feature_dim(feature_file),
                        "sample_type": "",
                        "target_key": "",
                        "target_available": 0,
                        "split_group_id": case_id,
                        "match_status": "missing_manifest_match",
                    }
                )
                continue

            if matched_selected.empty:
                first = matched.iloc[0]
                rows.append(
                    {
                        "slide_id": slide_id,
                        "case_submitter_id": case_id,
                        "sample_submitter_id": sample_id,
                        "aliquot_submitter_id": first["Aliquot Submitter ID"],
                        "feature_model": model_name,
                        "feature_path": str(feature_file.resolve()),
                        "feature_dim": infer_feature_dim(feature_file),
                        "sample_type": first["Sample Type"],
                        "target_key": first["Aliquot Submitter ID"],
                        "target_available": 0,
                        "split_group_id": case_id,
                        "match_status": "missing_target",
                    }
                )
                continue

            if len(matched_selected) > 1:
                rows.append(
                    {
                        "slide_id": slide_id,
                        "case_submitter_id": case_id,
                        "sample_submitter_id": sample_id,
                        "aliquot_submitter_id": "|".join(matched_selected["Aliquot Submitter ID"].tolist()),
                        "feature_model": model_name,
                        "feature_path": str(feature_file.resolve()),
                        "feature_dim": infer_feature_dim(feature_file),
                        "sample_type": "|".join(matched_selected["Sample Type"].astype(str).tolist()),
                        "target_key": "",
                        "target_available": 0,
                        "split_group_id": case_id,
                        "match_status": "ambiguous_manifest_match",
                    }
                )
                continue

            row = matched_selected.iloc[0]
            rows.append(
                {
                    "slide_id": slide_id,
                    "case_submitter_id": case_id,
                    "sample_submitter_id": sample_id,
                    "aliquot_submitter_id": row["Aliquot Submitter ID"],
                    "feature_model": model_name,
                    "feature_path": str(feature_file.resolve()),
                    "feature_dim": infer_feature_dim(feature_file),
                    "sample_type": row["Sample Type"],
                    "target_key": row["Aliquot Submitter ID"],
                    "target_available": 1,
                    "split_group_id": case_id,
                    "match_status": "ok",
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired WSI/proteome manifest")
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--biospecimen-manifest", type=Path, default=DEFAULT_BIOSPECIMEN_MANIFEST)
    parser.add_argument("--proteome-selection", type=Path, default=DEFAULT_PROTEOME_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=["ctranspath", "retccl"])
    args = parser.parse_args()

    biospecimen_df = pd.read_csv(args.biospecimen_manifest, sep="\t")
    selection_df = pd.read_csv(args.proteome_selection)

    rows = build_rows(
        features_root=args.features_root,
        biospecimen_df=biospecimen_df,
        selection_df=selection_df,
        models=args.models,
    )
    manifest_df = pd.DataFrame(rows).sort_values(["feature_model", "slide_id"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(args.output, index=False)

    print(f"Saved manifest -> {args.output}")
    print(manifest_df["match_status"].value_counts(dropna=False).to_string())
    print(f"rows={len(manifest_df)}")


if __name__ == "__main__":
    main()
