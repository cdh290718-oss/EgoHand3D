from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egohand3d.io_utils import read_json, summarize_numeric, write_csv, write_json


JSON_KINDS = {
    "kpts2d": "*_kpts2d.json",
    "joints3d": "*_joints3d.json",
    "mano": "*_mano.json",
    "frame": "frame_*.json",
    "summary": "summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact EgoHand3D result report.")
    parser.add_argument("--result_dir", type=Path, required=True, help="Directory to scan")
    parser.add_argument("--image_dir", type=Path, default=None, help="Optional source image directory")
    parser.add_argument("--out_dir", type=Path, default=None, help="Report output directory")
    parser.add_argument("--recursive", action="store_true", help="Scan result_dir recursively")
    return parser.parse_args()


def iter_files(root: Path, pattern: str, recursive: bool) -> list[Path]:
    return sorted(root.rglob(pattern) if recursive else root.glob(pattern))


def detect_json_kind(path: Path) -> str:
    name = path.name
    if name == "summary.json":
        return "summary"
    if name.startswith("frame_") and name.endswith(".json"):
        return "frame"
    if name.endswith("_kpts2d.json"):
        return "kpts2d"
    if name.endswith("_joints3d.json"):
        return "joints3d"
    if name.endswith("_mano.json"):
        return "mano"
    return "other_json"


def hands_from_record(record: dict) -> list[dict]:
    if "hands" in record and isinstance(record["hands"], list):
        return record["hands"]
    if "hand_index" in record:
        return [record]
    return []


def summarize_json_files(json_files: list[Path]) -> tuple[list[dict], dict]:
    rows = []
    hand_counts = []
    bbox_scores = []
    kind_counter = Counter()
    unreadable = []

    for path in json_files:
        kind = detect_json_kind(path)
        kind_counter[kind] += 1
        try:
            record = read_json(path)
        except Exception as exc:
            unreadable.append({"path": str(path), "error": str(exc)})
            rows.append({"path": str(path), "kind": kind, "readable": False, "num_hands": ""})
            continue

        hands = hands_from_record(record)
        hand_counts.append(len(hands))
        for hand in hands:
            if "bbox_score" in hand:
                try:
                    bbox_scores.append(float(hand["bbox_score"]))
                except (TypeError, ValueError):
                    pass
        rows.append(
            {
                "path": str(path),
                "kind": kind,
                "readable": True,
                "image": record.get("image", ""),
                "num_hands": len(hands),
            }
        )

    summary = {
        "num_json_files": len(json_files),
        "json_kinds": dict(kind_counter),
        "unreadable_json": unreadable,
        "hands_per_json": summarize_numeric(hand_counts),
        "bbox_scores": summarize_numeric(bbox_scores),
        "total_hands": int(sum(hand_counts)),
    }
    return rows, summary


def count_result_files(root: Path, recursive: bool) -> dict:
    patterns = {
        "json": "*.json",
        "jsonl": "*.jsonl",
        "obj": "*.obj",
        "jpg": "*.jpg",
        "png": "*.png",
        "mp4": "*.mp4",
        "csv": "*.csv",
        "npy": "*.npy",
        "npz": "*.npz",
        "txt": "*.txt",
    }
    return {key: len(iter_files(root, pattern, recursive)) for key, pattern in patterns.items()}


def image_inventory(image_dir: Path | None) -> dict:
    if image_dir is None:
        return {"provided": False}
    suffix_counts = Counter()
    total = 0
    for path in image_dir.iterdir():
        if path.is_file():
            suffix_counts[path.suffix.lower()] += 1
            total += 1
    return {
        "provided": True,
        "image_dir": str(image_dir),
        "total_files": total,
        "suffix_counts": dict(suffix_counts),
    }


def write_markdown_report(path: Path, summary: dict, rows: list[dict]) -> None:
    lines = [
        "# EgoHand3D Result Report",
        "",
        "## Overview",
        "",
        f"- Result directory: `{summary['result_dir']}`",
        f"- Recursive scan: `{summary['recursive']}`",
        f"- Total hands found in JSON files: `{summary['json_summary']['total_hands']}`",
        "",
        "## File Counts",
        "",
        "| Type | Count |",
        "| --- | ---: |",
    ]
    for key, value in summary["file_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## JSON Kinds", "", "| Kind | Count |", "| --- | ---: |"])
    for key, value in summary["json_summary"]["json_kinds"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Detection Scores", ""])
    score_summary = summary["json_summary"]["bbox_scores"]
    if score_summary["count"]:
        lines.extend(
            [
                f"- Count: `{score_summary['count']}`",
                f"- Mean: `{score_summary['mean']:.6f}`",
                f"- Median: `{score_summary['median']:.6f}`",
                f"- Min: `{score_summary['min']:.6f}`",
                f"- Max: `{score_summary['max']:.6f}`",
            ]
        )
    else:
        lines.append("- No bbox scores found.")
    lines.extend(["", "## First JSON Records", "", "| Kind | Image | Hands | Path |", "| --- | --- | ---: | --- |"])
    for row in rows[:50]:
        lines.append(f"| {row.get('kind','')} | {row.get('image','')} | {row.get('num_hands','')} | `{row.get('path','')}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.result_dir / "report")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = iter_files(args.result_dir, "*.json", args.recursive)
    rows, json_summary = summarize_json_files(json_files)
    summary = {
        "result_dir": str(args.result_dir),
        "recursive": bool(args.recursive),
        "file_counts": count_result_files(args.result_dir, args.recursive),
        "image_inventory": image_inventory(args.image_dir),
        "json_summary": json_summary,
    }
    write_json(out_dir / "report.json", summary)
    write_csv(out_dir / "json_files.csv", ["path", "kind", "readable", "image", "num_hands"], rows)
    write_markdown_report(out_dir / "report.md", summary, rows)
    print(f"[report] {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
