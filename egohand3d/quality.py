"""Transparent diagnostics, not learned joint confidence or proof of accuracy."""
from __future__ import annotations

import numpy as np

from .visualization import HAND_CONNECTIONS


def inspect_hand(hand: dict, min_detection_score: float = 0.5) -> dict:
    uv = np.asarray(hand["joints_2d_xy"], dtype=float)
    xyz = np.asarray(hand["joints_3d_camera"], dtype=float)
    width, height = hand["img_size_wh"]
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(xyz).all(axis=1)
    front = xyz[:, 2] > 1e-8
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    lengths = np.array([np.linalg.norm(xyz[a] - xyz[b]) for a, b in HAND_CONNECTIONS])
    flags = []
    if not finite.all():
        flags.append("nonfinite_geometry")
    if not front.all():
        flags.append("nonpositive_depth")
    if not inside.all():
        flags.append("joints_outside_image")
    if hand["bbox_score"] < min_detection_score:
        flags.append("low_detection_score")
    if np.any(lengths < 1e-5):
        flags.append("degenerate_bone")
    closure = hand.get("roundtrip")
    if closure and not closure["passed"]:
        flags.append("mano_roundtrip_failed")
    return {"flags": flags, "num_finite_joints": int(finite.sum()),
            "num_positive_depth": int(front.sum()), "num_inside_image": int(inside.sum()),
            "num_usable_joints": int((finite & front & inside).sum()),
            "detection_score": float(hand["bbox_score"]),
            "score_note": "Detector score is not per-joint confidence",
            "bone_lengths_m": [float(v) if np.isfinite(v) else None for v in lengths],
            "thresholds": {"min_detection_score": min_detection_score, "min_bone_length_m": 1e-5}}
