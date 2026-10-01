from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egohand3d.io_utils import read_json, write_csv, write_json


JSON_SUFFIXES = {
    "_kpts2d.json": "joints2d_json",
    "_joints3d.json": "joints3d_json",
    "_mano.json": "mano_json",
}

ARRAY_SUFFIXES = {
    "_joints2d.npy": "joints2d_npy",
    "_joints3d_root.npy": "joints3d_root_npy",
    "_joints3d_camera.npy": "joints3d_camera_npy",
    "_mano.npz": "mano_npz",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKIP_JSON_NAMES = {"summary.json", "report.json", "manifest.json", "validate_setup.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge EgoHand3D exported files into one sample-level index.")
    parser.add_argument("--input_dir", dest="input_dirs", action="append", type=Path, default=[])
    parser.add_argument("--joints2d_dir", type=Path, default=None)
    parser.add_argument("--joints3d_dir", type=Path, default=None)
    parser.add_argument("--mano_dir", type=Path, default=None)
    parser.add_argument("--mesh_dir", type=Path, default=None)
    parser.add_argument("--overlay_dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/merged_exports.json"))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--load_json", action="store_true", help="Read JSON exports and include compact metadata summaries.")
    parser.add_argument("--include_unknown_json", action="store_true")
    return parser.parse_args()


def configured_dirs(args: argparse.Namespace) -> list[tuple[Path, str | None]]:
    pairs: list[tuple[Path, str | None]] = []
    for path in args.input_dirs:
        pairs.append((path, None))
    named = [
        (args.joints2d_dir, "joints2d_json"),
        (args.joints3d_dir, "joints3d_json"),
        (args.mano_dir, "mano_json"),
        (args.mesh_dir, "mesh_obj"),
        (args.overlay_dir, "render_overlay"),
    ]
    for path, forced_kind in named:
        if path is not None:
            pairs.append((path, forced_kind))
    return pairs


def iter_files(root: Path, recursive: bool) -> list[Path]:
    if not root.exists():
        return []
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in iterator if path.is_file())


def split_sample_and_hand(stem: str) -> tuple[str, int | None]:
    match = re.match(r"^(?P<sample>.+)_(?P<hand>[0-9]+)$", stem)
    if match:
        return match.group("sample"), int(match.group("hand"))
    return stem, None


def classify_json(path: Path, forced_kind: str | None, include_unknown: bool) -> tuple[str, int | None, str] | None:
    name = path.name
    if name in SKIP_JSON_NAMES or name.endswith(".jsonl"):
        return None
    for suffix, kind in JSON_SUFFIXES.items():
        if name.endswith(suffix):
            sample, hand_index = split_sample_and_hand(name[: -len(suffix)])
            return sample, hand_index, forced_kind or kind
    if include_unknown:
        sample, hand_index = split_sample_and_hand(path.stem)
        return sample, hand_index, forced_kind or "json"
    return None


def classify_array(path: Path, forced_kind: str | None) -> tuple[str, int | None, str] | None:
    name = path.name
    for suffix, kind in ARRAY_SUFFIXES.items():
        if name.endswith(suffix):
            sample, hand_index = split_sample_and_hand(name[: -len(suffix)])
            return sample, hand_index, forced_kind or kind
    return None


def classify_obj(path: Path, forced_kind: str | None) -> tuple[str, int | None, str] | None:
    if path.suffix.lower() != ".obj":
        return None
    sample, hand_index = split_sample_and_hand(path.stem)
    return sample, hand_index, forced_kind or "mesh_obj"


def classify_overlay(path: Path, forced_kind: str | None) -> tuple[str, int | None, str] | None:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    sample, hand_index = split_sample_and_hand(path.stem)
    return sample, hand_index, forced_kind or "render_overlay"


def classify_file(path: Path, forced_kind: str | None, include_unknown_json: bool) -> tuple[str, int | None, str] | None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return classify_json(path, forced_kind, include_unknown_json)
    if suffix in {".npy", ".npz"}:
        return classify_array(path, forced_kind)
    if suffix == ".obj":
        return classify_obj(path, forced_kind)
    if forced_kind == "render_overlay":
        return classify_overlay(path, forced_kind)
    return None


def new_record(sample_id: str) -> dict:
    return {
        "id": sample_id,
        "files": {},
        "hands": {},
        "metadata": {},
    }


def append_file(record: dict, kind: str, path: Path, hand_index: int | None, load_json: bool) -> None:
    entry = {"path": str(path), "name": path.name}
    if hand_index is not None:
        entry["hand_index"] = hand_index
        hand_key = str(hand_index)
        record["hands"].setdefault(hand_key, {"hand_index": hand_index, "files": {}})
        record["hands"][hand_key]["files"].setdefault(kind, []).append(str(path))
    if load_json and path.suffix.lower() == ".json":
        summary = summarize_json(path)
        if summary:
            entry["summary"] = summary
            merge_metadata(record, summary)
    record["files"].setdefault(kind, []).append(entry)


def summarize_json(path: Path) -> dict:
    try:
        payload = read_json(path)
    except Exception as exc:
        return {"readable": False, "error": str(exc)}

    if not isinstance(payload, dict):
        return {"readable": True, "top_level": type(payload).__name__}

    hands = payload.get("hands")
    summary = {
        "readable": True,
        "image": payload.get("image"),
        "width": payload.get("width"),
        "height": payload.get("height"),
    }
    if isinstance(hands, list):
        summary["num_hands"] = len(hands)
    elif "hand_index" in payload:
        summary["num_hands"] = 1
        summary["hand_index"] = payload.get("hand_index")
    elif "num_hands" in payload:
        summary["num_hands"] = payload.get("num_hands")
    return {key: value for key, value in summary.items() if value is not None}


def merge_metadata(record: dict, summary: dict) -> None:
    metadata = record["metadata"]
    for key in ["image", "width", "height"]:
        if key in summary and key not in metadata:
            metadata[key] = summary[key]
    if "num_hands" in summary:
        current = metadata.get("num_hands", 0)
        try:
            metadata["num_hands"] = max(int(current), int(summary["num_hands"]))
        except (TypeError, ValueError):
            metadata["num_hands"] = summary["num_hands"]


def normalize_records(records: dict[str, dict]) -> list[dict]:
    normalized = []
    for sample_id in sorted(records):
        record = records[sample_id]
        record["hands"] = [record["hands"][key] for key in sorted(record["hands"], key=lambda v: int(v))]
        record["file_counts"] = {kind: len(items) for kind, items in sorted(record["files"].items())}
        record["num_files"] = sum(record["file_counts"].values())
        normalized.append(record)
    return normalized


def build_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        counts = record["file_counts"]
        rows.append(
            {
                "id": record["id"],
                "num_files": record["num_files"],
                "num_hands": record["metadata"].get("num_hands", len(record["hands"])),
                "has_2d": int("joints2d_json" in counts or "joints2d_npy" in counts),
                "has_3d": int("joints3d_json" in counts or "joints3d_root_npy" in counts),
                "has_mano": int("mano_json" in counts or "mano_npz" in counts),
                "has_mesh": int("mesh_obj" in counts),
                "has_overlay": int("render_overlay" in counts),
                "file_counts": "; ".join(f"{key}:{value}" for key, value in counts.items()),
            }
        )
    return rows


def summarize(records: list[dict], scanned_dirs: list[str]) -> dict:
    file_type_counts = Counter()
    for record in records:
        file_type_counts.update(record["file_counts"])
    return {
        "num_samples": len(records),
        "num_files": sum(record["num_files"] for record in records),
        "scanned_dirs": scanned_dirs,
        "file_type_counts": dict(file_type_counts),
        "num_with_2d": sum(1 for row in build_rows(records) if row["has_2d"]),
        "num_with_3d": sum(1 for row in build_rows(records) if row["has_3d"]),
        "num_with_mano": sum(1 for row in build_rows(records) if row["has_mano"]),
        "num_with_mesh": sum(1 for row in build_rows(records) if row["has_mesh"]),
        "num_with_overlay": sum(1 for row in build_rows(records) if row["has_overlay"]),
    }


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# EgoHand3D Merged Exports",
        "",
        "## Summary",
        "",
        f"- Samples: `{payload['summary']['num_samples']}`",
        f"- Files: `{payload['summary']['num_files']}`",
        f"- With 2D joints: `{payload['summary']['num_with_2d']}`",
        f"- With 3D joints: `{payload['summary']['num_with_3d']}`",
        f"- With MANO: `{payload['summary']['num_with_mano']}`",
        f"- With mesh: `{payload['summary']['num_with_mesh']}`",
        "",
        "## File Types",
        "",
        "| Type | Count |",
        "| --- | ---: |",
    ]
    for key, value in payload["summary"]["file_type_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Samples", "", "| ID | Files | Types |", "| --- | ---: | --- |"])
    for record in payload["records"][:100]:
        types = ", ".join(record["file_counts"].keys())
        lines.append(f"| {record['id']} | {record['num_files']} | {types} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records: dict[str, dict] = {}
    scanned_dirs = []

    for root, forced_kind in configured_dirs(args):
        scanned_dirs.append(str(root))
        for path in iter_files(root, args.recursive):
            classified = classify_file(path, forced_kind, args.include_unknown_json)
            if classified is None:
                continue
            sample_id, hand_index, kind = classified
            record = records.setdefault(sample_id, new_record(sample_id))
            append_file(record, kind, path, hand_index, args.load_json)

    normalized = normalize_records(records)
    payload = {
        "summary": summarize(normalized, scanned_dirs),
        "records": normalized,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, payload)
    write_csv(
        args.out.with_suffix(".csv"),
        ["id", "num_files", "num_hands", "has_2d", "has_3d", "has_mano", "has_mesh", "has_overlay", "file_counts"],
        build_rows(normalized),
    )
    write_markdown(args.out.with_suffix(".md"), payload)
    print(f"[merge] {args.out} samples={len(normalized)} files={payload['summary']['num_files']}")


if __name__ == "__main__":
    main()
