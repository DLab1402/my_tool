from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import nrrd
import numpy as np
import pandas as pd
import torch
import torchio as tio
from sklearn.model_selection import StratifiedKFold


SEGMENT_TAG_PATTERN = re.compile(r"^Segment(\d+)_Tags$")
SEGMENT_SIGNAL_STRING = "Segmentation.Status:inprogress"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def get_acl_indices(seg_header: dict[str, Any]) -> list[str]:
    indices: list[str] = []
    for key in seg_header.keys():
        match = SEGMENT_TAG_PATTERN.match(key)
        if match:
            indices.append(match.group(1))
    return indices


def get_acl_type(seg_header: dict[str, Any]) -> str | None:
    for idx in get_acl_indices(seg_header):
        tag_key = f"Segment{idx}_Tags"
        name_key = f"Segment{idx}_Name"
        if tag_key in seg_header and name_key in seg_header:
            tag_value = str(seg_header[tag_key])
            if tag_value.startswith(SEGMENT_SIGNAL_STRING):
                return str(seg_header[name_key])
    return None


def map_acl_label(raw_label: str | None) -> tuple[int, str] | None:
    label = normalize_text(raw_label)
    if not label:
        return None

    healthy_aliases = {
        "healthy",
        "normal",
        "intact",
        "binh_thuong",
        "no_acl_injury",
        "no_injury",
    }
    partial_markers = ("partial", "dut_ban_phan")
    complete_markers = ("complete", "full_thickness", "dut_hoan_toan")

    if label in healthy_aliases:
        return 0, "healthy"
    if any(marker in label for marker in partial_markers):
        return 1, "acl_injury"
    if any(marker in label for marker in complete_markers):
        return 1, "acl_injury"
    return None


def build_affine_from_nrrd_header(header: dict[str, Any]) -> np.ndarray:
    affine = np.eye(4, dtype=np.float32)

    space_directions = header.get("space directions")
    if space_directions is not None:
        directions = np.asarray(space_directions, dtype=object)
        if directions.ndim == 2 and directions.shape[0] >= 3:
            matrix = np.zeros((3, 3), dtype=np.float32)
            for axis in range(3):
                vector = directions[axis]
                if isinstance(vector, str) and vector.lower() == "none":
                    continue
                vector_np = np.asarray(vector, dtype=np.float32).reshape(-1)
                if vector_np.size == 3:
                    matrix[:, axis] = vector_np
            if np.any(matrix):
                affine[:3, :3] = matrix

    space_origin = header.get("space origin")
    if space_origin is not None:
        origin = np.asarray(space_origin, dtype=np.float32).reshape(-1)
        if origin.size == 3:
            affine[:3, 3] = origin

    return affine


def find_studies(input_root: Path) -> list[dict[str, Any]]:
    studies: list[dict[str, Any]] = []
    for path_dir in sorted(input_root.rglob("*")):
        if not path_dir.is_dir():
            continue

        seg_files = sorted(path_dir.glob("Segmentation*.seg.nrrd"))
        t2_files = sorted(
            path
            for path in path_dir.glob("*.nrrd")
            if "t2_tse_sag" in path.stem.lower() and not path.name.lower().endswith(".seg.nrrd")
        )
        if not seg_files or not t2_files:
            continue

        _, seg_header = nrrd.read(str(seg_files[0]))
        raw_label = get_acl_type(seg_header)
        mapped = map_acl_label(raw_label)
        if mapped is None:
            continue

        acl_label, acl_label_name = mapped
        studies.append(
            {
                "patient_key": str(path_dir.relative_to(input_root)).replace("\\", "/"),
                "nrrd_path": t2_files[0],
                "seg_path": seg_files[0],
                "raw_label": raw_label,
                "acl_label_name": acl_label_name,
                "acl": acl_label,
                "abnormal": acl_label,
                "meniscus": 0,
                "plane": "sagittal",
            }
        )

    return studies


def export_images(records: list[dict[str, Any]], output_root: Path) -> pd.DataFrame:
    preprocessed_root = output_root / "preprocessed"
    rows: list[dict[str, Any]] = []

    for uid, record in enumerate(records):
        volume, header = nrrd.read(str(record["nrrd_path"]))
        volume = np.asarray(volume, dtype=np.float32)
        affine = build_affine_from_nrrd_header(header)

        image = tio.ScalarImage(
            tensor=torch.from_numpy(volume[None]),
            affine=affine,
        )
        out_path_train = preprocessed_root / "data" / "train" / record["plane"] / f"{uid:04d}.nii.gz"
        out_path_train.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path_train)

        rows.append(
            {
                "ID": uid,
                "PatientKey": record["patient_key"],
                "RawLabel": record["raw_label"],
                "AclLabelName": record["acl_label_name"],
                "abnormal": record["abnormal"],
                "acl": record["acl"],
                "meniscus": record["meniscus"],
                "Plane": record["plane"],
                "Folder": "train/",
                "BaseSplit": "cross_validation_pool",
                "NrrdPath": str(record["nrrd_path"]),
                "SegPath": str(record["seg_path"]),
                "NiftiPath": str(out_path_train),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(preprocessed_root / "manifest.csv", index=False)
    manifest.to_csv(preprocessed_root / "train.csv", index=False)
    manifest.iloc[0:0].to_csv(preprocessed_root / "valid.csv", index=False)
    return manifest


def build_cv_split(manifest: pd.DataFrame, num_folds: int, random_state: int) -> pd.DataFrame:
    train_df = manifest.reset_index(drop=True)

    if train_df.empty:
        raise ValueError("No studies were found to split.")

    class_counts = train_df["acl"].value_counts()
    min_class_count = int(class_counts.min())
    if num_folds < 2:
        raise ValueError("--num_folds must be at least 2.")
    if min_class_count < num_folds:
        raise ValueError(
            "Not enough training samples per class for the requested number of folds. "
            f"Smallest class in train split has {min_class_count} samples, but --num_folds={num_folds}."
        )

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    split_df = train_df.copy()
    split_df["Fold"] = -1
    split_df["Split"] = "train"

    for fold_i, (_, val_idx) in enumerate(skf.split(train_df["ID"], train_df["acl"])):
        split_df.loc[val_idx, "Fold"] = fold_i

    if (split_df["Fold"] < 0).any():
        raise RuntimeError("Some studies were not assigned to any fold.")

    return split_df[
        [
            "ID",
            "PatientKey",
            "RawLabel",
            "AclLabelName",
            "abnormal",
            "acl",
            "meniscus",
            "Plane",
            "Folder",
            "BaseSplit",
            "Split",
            "Fold",
            "NrrdPath",
            "SegPath",
            "NiftiPath",
        ]
    ]


def build_summary(manifest: pd.DataFrame, split_df: pd.DataFrame, num_folds: int) -> dict[str, Any]:
    fold_summaries: list[dict[str, Any]] = []
    for fold_i in range(num_folds):
        fold_val = split_df[split_df["Fold"] == fold_i]
        fold_train = split_df[split_df["Fold"] != fold_i]
        fold_summaries.append(
            {
                "fold": fold_i,
                "num_train": int(len(fold_train)),
                "num_val": int(len(fold_val)),
                "class_distribution_train": dict(sorted(Counter(fold_train["acl"]).items())),
                "class_distribution_val": dict(sorted(Counter(fold_val["acl"]).items())),
            }
        )

    return {
        "num_total": int(len(manifest)),
        "class_distribution_total": dict(sorted(Counter(manifest["acl"]).items())),
        "num_folds": num_folds,
        "folds": fold_summaries,
    }


def export_dataset(
    records: list[dict[str, Any]],
    output_root: Path,
    num_folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = export_images(records, output_root)

    split_root = output_root / "preprocessed" / "splits"
    split_root.mkdir(parents=True, exist_ok=True)

    split_df = build_cv_split(manifest, num_folds=num_folds, random_state=random_state)
    split_df.to_csv(split_root / "split.csv", index=False)

    summary = build_summary(manifest, split_df, num_folds=num_folds)
    with open(split_root / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return manifest, split_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert sagittal ACL .nrrd studies into an MST MRNet-style dataset with "
            "train/valid export and K-fold cross-validation metadata."
        )
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        required=True,
        help="Root directory containing the raw study folders with .nrrd and .seg.nrrd files.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Output root. The script will create <output_root>/preprocessed/...",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds created from the full dataset. Default: 5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for K-fold shuffle. Default: 0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    studies = find_studies(args.input_root)
    if not studies:
        raise FileNotFoundError(
            f"No matching studies were found under {args.input_root}. "
            "Expected folders containing both '*t2_tse_sag*.nrrd' and 'Segmentation*.seg.nrrd'."
        )

    manifest, split_df, summary = export_dataset(
        studies,
        args.output_root,
        num_folds=args.num_folds,
        random_state=args.seed,
    )

    preprocessed_root = args.output_root / "preprocessed"
    print(f"Exported {len(manifest)} studies to {preprocessed_root}")
    print(f"Manifest:   {preprocessed_root / 'manifest.csv'}")
    print(f"Train CSV:  {preprocessed_root / 'train.csv'}")
    print(f"Valid CSV:  {preprocessed_root / 'valid.csv'}")
    print(f"Split CSV:  {preprocessed_root / 'splits' / 'split.csv'}")
    print(f"Rows in split.csv: {len(split_df)}")
    print("Class distribution total:", summary["class_distribution_total"])


if __name__ == "__main__":
    main()
