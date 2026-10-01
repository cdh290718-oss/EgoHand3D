"""Offline side-by-side videos comparing raw observations and processed poses."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .storage import load_json
from .visualization import draw_hand_skeleton, draw_label


def export_sequence_videos(directory: Path) -> list[str]:
    entries = load_json(directory / "index.json")["records"]
    groups = {}
    for entry in entries:
        groups.setdefault(entry["sequence_id"], []).append(entry)
    outputs = []
    for sequence_id, group in groups.items():
        group.sort(key=lambda e: e["timestamp_s"])
        times = np.array([e["timestamp_s"] for e in group])
        differences = np.diff(times)
        if not len(differences):
            continue
        if not np.allclose(differences, differences[0], rtol=1e-3, atol=1e-6):
            raise ValueError("Video export requires evenly spaced frames; JSON and plots support irregular timestamps")
        fps = float(1 / differences[0])
        output = directory / f"{sequence_id}_comparison.mp4"
        writer = None
        capture = None
        capture_path = None
        try:
            for entry in group:
                record = load_json(directory / entry["record"])
                source = Path(record["source_path"])
                if record["time_source"] == "container_fps":
                    if str(source) != capture_path:
                        if capture is not None:
                            capture.release()
                        capture = cv2.VideoCapture(str(source))
                        capture_path = str(source)
                    capture.set(cv2.CAP_PROP_POS_FRAMES, record["frame_index"])
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError(f"Cannot reread source video frame {record['frame_index']}")
                else:
                    image = cv2.imread(str(source))
                    if image is None:
                        raise ValueError(f"Cannot reread source image {source}")
                panels = [image.copy(), image.copy()]
                for hand in record["hands"]:
                    observed = hand.get("observation") == "detected"
                    if observed:
                        draw_hand_skeleton(panels[0], np.asarray(hand["joints_2d_xy"]))
                    points = np.asarray(hand["processed"]["joints_2d_xy"])
                    draw_hand_skeleton(panels[1], points)
                    location = tuple(np.rint(points[0]).astype(int))
                    draw_label(panels[1], f"ID {hand['track_id']} {'detected' if observed else 'INTERPOLATED'}",
                               location, (20, 230, 100) if observed else (20, 120, 250))
                for panel, title in zip(panels, ["Raw model observations", "Processed MANO; gaps explicitly labelled"]):
                    draw_label(panel, title, (15, 30), (240, 240, 240))
                combined = np.hstack(panels)
                if writer is None:
                    size = (combined.shape[1], combined.shape[0])
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                    if not writer.isOpened():
                        raise OSError("Cannot initialize MP4 writer")
                if (combined.shape[1], combined.shape[0]) != size:
                    raise ValueError("Sequence image dimensions changed")
                writer.write(combined)
        finally:
            if writer is not None:
                writer.release()
            if capture is not None:
                capture.release()
        outputs.append(str(output))
    return outputs
