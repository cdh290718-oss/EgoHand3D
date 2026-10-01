"""Preflight images and videos without loading a neural network."""
from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np

from .storage import file_identity, fingerprint

IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEOS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def natural_key(path: Path):
    return tuple((0, int(x)) if x.isdigit() else (1, x.lower())
                 for x in re.split(r"(\d+)", str(path)))


def scan_media(source: Path, recursive: bool = False, hash_inputs: bool = False) -> dict:
    source = source.resolve(strict=True)
    candidates = ([source] if source.is_file() else
                  sorted(source.rglob("*") if recursive else source.iterdir(), key=natural_key))
    records = []
    ignored = 0
    for path in candidates:
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in IMAGES | VIDEOS:
            ignored += 1
            continue
        record = {"identity": file_identity(path, hash_inputs), "kind": "image" if suffix in IMAGES else "video"}
        record["id"] = fingerprint(record["identity"])[:20]
        if record["kind"] == "image":
            image = cv2.imread(str(path))
            record["readable"] = image is not None
            if image is not None:
                record.update(width=int(image.shape[1]), height=int(image.shape[0]), frames=1)
        else:
            capture = cv2.VideoCapture(str(path))
            try:
                ok, image = capture.read()
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                record["readable"] = bool(ok and np.isfinite(fps) and fps > 0)
                if ok:
                    record.update(width=int(image.shape[1]), height=int(image.shape[0]),
                                  fps=fps if np.isfinite(fps) else None,
                                  frames=int(frames) if np.isfinite(frames) and frames > 0 else None)
            finally:
                capture.release()
        if not record["readable"]:
            record["error"] = "Cannot decode media or determine a positive video frame rate"
        records.append(record)
    return {"schema": "egohand3d.media/1", "source": str(source), "recursive": recursive,
            "files": records, "summary": {"files": len(records), "ignored_files": ignored,
            "readable": sum(r["readable"] for r in records),
            "unreadable": sum(not r["readable"] for r in records)}}


def iter_frames(manifest: dict, stride: int = 1, limit: int = 0, sequence_fps: float | None = None):
    """Yield decoded images and metadata; video timestamps assume constant frame rate.

    A decode failure is yielded rather than silently counted as a success. For an
    image sequence, users must explicitly supply its frame rate.
    """
    if stride < 1 or limit < 0 or (sequence_fps is not None and sequence_fps <= 0):
        raise ValueError("stride/fps must be positive and limit must be nonnegative")
    emitted = 0
    image_index = 0
    for item in manifest["files"]:
        path = Path(item["identity"]["path"])
        identity = file_identity(path, "sha256" in item["identity"])
        if identity != item["identity"]:
            raise ValueError(f"Input changed after preflight: {path}")
        common = {"source_path": str(path), "media_id": item["id"], "image": path.name}
        if item["kind"] == "image" or not item["readable"]:
            metadata = {**common, "sample_id": item["id"], "frame_index": image_index,
                        "sequence_id": fingerprint(manifest["source"])[:20] if sequence_fps else item["id"],
                        "timestamp_s": image_index / sequence_fps if sequence_fps else None,
                        "time_source": "user_frame_rate" if sequence_fps else "none"}
            image_index += 1
            yield metadata, cv2.imread(str(path)) if item["readable"] else None
            emitted += 1
        else:
            capture = cv2.VideoCapture(str(path))
            index = 0
            try:
                while not limit or emitted < limit:
                    ok, image = capture.read()
                    if not ok:
                        if item.get("frames") and index < item["frames"]:
                            yield {**common, "sample_id": f"{item['id']}_{index:08d}",
                                   "sequence_id": item["id"], "frame_index": index,
                                   "timestamp_s": index / item["fps"], "time_source": "container_fps",
                                   "decode_error": "Video ended before reported frame count"}, None
                        break
                    if index % stride == 0:
                        yield {**common, "sample_id": f"{item['id']}_{index:08d}",
                               "sequence_id": item["id"], "frame_index": index,
                               "timestamp_s": index / item["fps"], "time_source": "container_fps"}, image
                        emitted += 1
                    index += 1
            finally:
                capture.release()
        if limit and emitted >= limit:
            return
