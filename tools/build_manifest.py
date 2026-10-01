from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egohand3d.io_utils import write_csv, write_json


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
PRED_SUFFIXES = [
    "_kpts2d.json",
    "_joints3d.json",
    "_mano.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an EgoHand3D processing manifest.")
    parser.add_argument("--image_dir", type=Path, default=None)
    parser.add_argument("--video_dir", type=Path, default=None)
    parser.add_argument("--pred_dir", type=Path, default=None)
    parser.add_argument("--label_dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/manifest.json"))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--require_predictions", action="store_true")
    parser.add_argument("--require_labels", action="store_true")
    return parser.parse_args()


def iter_media(root: Path | None, suffixes: set[str], recursive: bool) -> list[Path]:
    if root is None or not root.exists():
        return []
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in suffixes)


def find_prediction_files(pred_dir: Path | None, stem: str) -> dict:
    result = {}
    if pred_dir is None or not pred_dir.exists():
        return result
    for suffix in PRED_SUFFIXES:
        path = pred_dir / f"{stem}{suffix}"
        if path.exists():
            result[suffix.lstrip("_").replace(".json", "")] = str(path)
    per_hand = sorted(pred_dir.glob(f"{stem}_*_joints3d.json")) + sorted(pred_dir.glob(f"{stem}_*_mano.json"))
    if per_hand:
        result["per_hand"] = [str(p) for p in per_hand]
    return result


def find_label_file(label_dir: Path | None, stem: str) -> str | None:
    if label_dir is None or not label_dir.exists():
        return None
    candidates = [
        label_dir / f"{stem}.json",
        label_dir / f"{stem}.pkl",
        label_dir / f"{stem}.pickle",
        label_dir / f"{stem}_label.json",
        label_dir / f"{stem}_kps2d.pkl",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def media_record(path: Path, media_type: str, args: argparse.Namespace) -> dict:
    preds = find_prediction_files(args.pred_dir, path.stem)
    label = find_label_file(args.label_dir, path.stem)
    return {
        "id": path.stem,
        "media_type": media_type,
        "path": str(path),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "predictions": preds,
        "has_predictions": bool(preds),
        "label": label,
        "has_label": label is not None,
    }


def filter_records(records: list[dict], require_predictions: bool, require_labels: bool) -> list[dict]:
    out = []
    for record in records:
        if require_predictions and not record["has_predictions"]:
            continue
        if require_labels and not record["has_label"]:
            continue
        out.append(record)
    return out


def summarize(records: list[dict]) -> dict:
    media_counts = Counter(record["media_type"] for record in records)
    suffix_counts = Counter(record["suffix"] for record in records)
    pred_counts = Counter()
    for record in records:
        for key in record["predictions"]:
            pred_counts[key] += 1
    return {
        "num_records": len(records),
        "media_counts": dict(media_counts),
        "suffix_counts": dict(suffix_counts),
        "prediction_counts": dict(pred_counts),
        "num_with_predictions": sum(1 for r in records if r["has_predictions"]),
        "num_with_labels": sum(1 for r in records if r["has_label"]),
        "num_missing_predictions": sum(1 for r in records if not r["has_predictions"]),
        "num_missing_labels": sum(1 for r in records if not r["has_label"]),
    }


def csv_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        rows.append(
            {
                "id": record["id"],
                "media_type": record["media_type"],
                "path": record["path"],
                "has_predictions": record["has_predictions"],
                "has_label": record["has_label"],
                "label": record.get("label") or "",
                "kpts2d": record["predictions"].get("kpts2d", ""),
                "joints3d": record["predictions"].get("joints3d", ""),
                "mano": record["predictions"].get("mano", ""),
            }
        )
    return rows


def write_markdown(path: Path, manifest: dict) -> None:
    summary = manifest["summary"]
    lines = [
        "# EgoHand3D Manifest",
        "",
        "## Summary",
        "",
        f"- Records: `{summary['num_records']}`",
        f"- With predictions: `{summary['num_with_predictions']}`",
        f"- With labels: `{summary['num_with_labels']}`",
        f"- Missing predictions: `{summary['num_missing_predictions']}`",
        f"- Missing labels: `{summary['num_missing_labels']}`",
        "",
        "## Media Counts",
        "",
        "| Type | Count |",
        "| --- | ---: |",
    ]
    for key, value in summary["media_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## First Records", "", "| ID | Type | Predictions | Label |", "| --- | --- | --- | --- |"])
    for record in manifest["records"][:50]:
        pred_keys = ", ".join(record["predictions"].keys())
        lines.append(f"| {record['id']} | {record['media_type']} | {pred_keys} | {record.get('label') or ''} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    image_paths = iter_media(args.image_dir, IMAGE_SUFFIXES, args.recursive)
    video_paths = iter_media(args.video_dir, VIDEO_SUFFIXES, args.recursive)
    records = [media_record(path, "image", args) for path in image_paths]
    records.extend(media_record(path, "video", args) for path in video_paths)
    records = filter_records(records, args.require_predictions, args.require_labels)
    manifest = {
        "image_dir": str(args.image_dir) if args.image_dir else None,
        "video_dir": str(args.video_dir) if args.video_dir else None,
        "pred_dir": str(args.pred_dir) if args.pred_dir else None,
        "label_dir": str(args.label_dir) if args.label_dir else None,
        "recursive": bool(args.recursive),
        "summary": summarize(records),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, manifest)
    write_csv(
        args.out.with_suffix(".csv"),
        ["id", "media_type", "path", "has_predictions", "has_label", "label", "kpts2d", "joints3d", "mano"],
        csv_rows(records),
    )
    write_markdown(args.out.with_suffix(".md"), manifest)
    print(f"[manifest] {args.out} records={len(records)}")


if __name__ == "__main__":
    main()
