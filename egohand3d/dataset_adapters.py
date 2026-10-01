"""Convert existing, trusted HOI4D annotations into an explicit evaluation manifest."""
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from .storage import file_hash, load_json, save_json


def prepare_hoi4d(run: Path, sample_dir: Path, out: Path) -> dict:
    manifest = load_json(run / "manifest.json")
    config = load_json(run / "run_config.json")
    source_manifest = sample_dir / "meta/manifest.json"
    annotations = load_json(source_manifest)
    by_path = {str((sample_dir / a["image"]).resolve()): a for a in annotations}
    if len(by_path) != len(annotations):
        raise ValueError("Duplicate HOI4D image entries")
    media = manifest["files"]
    if any(item["kind"] != "image" for item in media):
        raise ValueError("This adapter expects an image run")
    limit = config["options"]["limit"]
    if limit:
        media = media[:limit]
    frames = []
    labels = {}
    for item in media:
        key = item["identity"]["path"]
        if key not in by_path:
            raise ValueError(f"Input lacks a GT mapping: {key}")
        entry = by_path[key]
        label_path = sample_dir / entry["label"]
        # HOI4D stores annotations as pickle. Only use the user's trusted local dataset.
        with label_path.open("rb") as stream:
            annotation = pickle.load(stream)
        points = np.asarray(annotation["kps2D"], dtype=float)
        if points.shape != (21, 2) or not np.isfinite(points).all():
            raise ValueError(f"Invalid 2D label: {label_path}")
        minimum, maximum = points.min(0), points.max(0)
        padding = np.maximum(maximum - minimum, 1.) * 0.1
        bbox = np.concatenate([minimum - padding, maximum + padding])
        frames.append({"sample_id": item["id"], "source_sample_id": entry["sample_id"],
                       "hands": [{"hand_side": entry["hand_side"], "bbox_xyxy": bbox.tolist(),
                                  "bbox_source": "GT 2D joint extent with 10% padding per side",
                                  "joints_2d_xy": points.tolist(), "valid": [True] * 21}]})
        labels[str(label_path.resolve())] = file_hash(label_path)
    result = {"schema": "egohand3d.ground_truth/1", "joint_order": "OpenPose21", "length_unit": "m",
              "camera_axes": "x_right_y_down_z_forward", "labels_complete": False,
              "label_scope": "One annotated target hand per image; other visible hands may be unlabelled",
              "source_manifest_sha256": file_hash(source_manifest), "label_sha256": labels,
              "note": "2D evaluation only; MANO GT requires a separately verified joint/coordinate conversion",
              "frames": frames}
    save_json(out, result)
    return {"frames": len(frames), "out": str(out), "ground_truth_sha256": file_hash(out)}
