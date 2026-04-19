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
            path for path in path_dir.glob("*.nrrd")
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


def export_as_test_set(records: list[dict[str, Any]], output_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    preprocessed_root = output_root / "preprocessed"
    path_test = preprocessed_root / "data" / "valid" / "sagittal"
    split_root = preprocessed_root / "splits"
    path_test.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for uid, record in enumerate(records):
        volume, header = nrrd.read(str(record["nrrd_path"]))
        volume = np.asarray(volume, dtype=np.float32)
        affine = build_affine_from_nrrd_header(header)

        image = tio.ScalarImage(
            tensor=torch.from_numpy(volume[None]),
            affine=affine,
        )
        out_path = path_test / f"{uid:04d}.nii.gz"
        image.save(out_path)

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
                "Folder": "valid/",
                "Split": "test",
                "Fold": 0,
                "NrrdPath": str(record["nrrd_path"]),
                "SegPath": str(record["seg_path"]),
                "NiftiPath": str(out_path),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(preprocessed_root / "manifest.csv", index=False)
    manifest.to_csv(preprocessed_root / "valid.csv", index=False)
    manifest.to_csv(split_root / "split.csv", index=False)

    summary = {
        "num_total": int(len(manifest)),
        "num_test": int(len(manifest)),
        "class_distribution_total": dict(sorted(Counter(manifest["acl"]).items())),
        "export_mode": "all_test",
        "fold": 0,
    }
    with open(split_root / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return manifest, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert sagittal ACL .nrrd studies into an MST MRNet-style test-only dataset."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    studies = find_studies(args.input_root)
    if not studies:
        raise FileNotFoundError(
            f"No matching studies were found under {args.input_root}. "
            "Expected folders containing both '*t2_tse_sag*.nrrd' and 'Segmentation*.seg.nrrd'."
        )

    manifest, summary = export_as_test_set(studies, args.output_root)

    print(f"Exported {len(manifest)} studies to {args.output_root / 'preprocessed'}")
    print(f"Manifest: {args.output_root / 'preprocessed' / 'manifest.csv'}")
    print(f"Split:    {args.output_root / 'preprocessed' / 'splits' / 'split.csv'}")
    print("Class distribution:", summary["class_distribution_total"])


if __name__ == "__main__":
    main()
