from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compact_point_arrays(json_str: str) -> str:
    """Fold short numeric point arrays onto one line for readable JSON."""
    json_str = re.sub(
        r"\[\s*\n\s*([-\d.eE+]+),\s*\n\s*([-\d.eE+]+)\s*\n\s*\]",
        r"[\1, \2]",
        json_str,
    )
    return re.sub(
        r"\[\s*\n\s*([-\d.eE+]+),\s*\n\s*([-\d.eE+]+),\s*\n\s*([-\d.eE+]+)\s*\n\s*\]",
        r"[\1, \2, \3]",
        json_str,
    )


def to_jsonable(value):
    """Convert numpy and path objects into JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, record: dict, compact_points: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(record), ensure_ascii=False, indent=2)
    if compact_points:
        text = compact_point_arrays(text)
    path.write_text(text + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def collect_image_paths(img_folder: Path, patterns: Sequence[str] | None = None) -> list[Path]:
    if patterns:
        paths = [path for pattern in patterns for path in img_folder.glob(pattern)]
    else:
        paths = [p for p in img_folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted({p.resolve() for p in paths if p.is_file()})


def relative_or_name(path: Path, root: Path | None = None) -> str:
    if root is None:
        return path.name
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_numeric(values: Sequence[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
