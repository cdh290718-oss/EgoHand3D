"""Association and explicit postprocessing of observed hand sequences.

This is an application-level baseline, not an identity recognition model.
Interpolated records are marked and remain separate from detector observations.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .mano_codec import ManoDecoder, interpolate_packets
from .storage import file_hash, load_json, save_json


def association_cost(a: dict, b: dict) -> float:
    if a["hand_side"] != b["hand_side"]:
        return 1e6
    x, y = np.asarray(a["bbox_xyxy"]), np.asarray(b["bbox_xyxy"])
    intersection = np.maximum(0, np.minimum(x[2:], y[2:]) - np.maximum(x[:2], y[:2])).prod()
    area_x = np.maximum(0, x[2:] - x[:2]).prod()
    area_y = np.maximum(0, y[2:] - y[:2]).prod()
    iou = intersection / max(area_x + area_y - intersection, 1.)
    scale = max(np.linalg.norm(x[2:] - x[:2]), np.linalg.norm(y[2:] - y[:2]), 1.)
    distance = np.linalg.norm((x[:2] + x[2:] - y[:2] - y[2:]) / 2) / scale
    if distance > 1.5:
        return 1e6
    return float(0.6 * (1 - iou) + 0.4 * distance)


class HandTracker:
    def __init__(self, max_gap: int = 3, smoothing_seconds: float = 0.08):
        if max_gap < 0 or smoothing_seconds < 0:
            raise ValueError("max_gap and smoothing_seconds must be nonnegative")
        self.max_gap = max_gap
        self.tau = smoothing_seconds
        self.tracks = {}
        self.next_id = 1
        self.previous_time = None

    def update(self, hands: list[dict], index: int, timestamp: float):
        if not math.isfinite(timestamp) or (self.previous_time is not None and timestamp <= self.previous_time):
            raise ValueError("Sequence timestamps must be finite and strictly increasing")
        self.previous_time = timestamp
        self.tracks = {k: v for k, v in self.tracks.items() if index - v["index"] <= self.max_gap + 1}
        keys = list(self.tracks)
        assignments = {}
        if keys and hands:
            cost = np.array([[association_cost(self.tracks[k]["hand"], h) for h in hands] for k in keys])
            rr, cc = linear_sum_assignment(cost)
            assignments = {int(c): keys[int(r)] for r, c in zip(rr, cc) if cost[r, c] < 1.1}
        output = []
        gaps = []
        for position, hand in enumerate(hands):
            track_id = assignments.get(position)
            packet = hand["mano"]
            previous = self.tracks.get(track_id)
            if previous is None:
                track_id = self.next_id
                self.next_id += 1
                filtered = packet
            else:
                elapsed = timestamp - previous["timestamp"]
                # A long time gap should not retain a stale pose, even with few samples.
                weight = 1 - math.exp(-elapsed / self.tau) if self.tau else 1.
                filtered = interpolate_packets(previous["packet"], packet, weight)
                if index - previous["index"] > 1:
                    gaps.append({"track_id": track_id, "start": previous["index"], "end": index,
                                 "start_time": previous["timestamp"], "end_time": timestamp,
                                 "first": previous["packet"], "second": filtered})
            current = {**copy.deepcopy(hand), "track_id": track_id, "observation": "detected",
                       "processed_mano": filtered}
            self.tracks[track_id] = {"hand": hand, "packet": filtered, "index": index, "timestamp": timestamp}
            output.append(current)
        return output, gaps


def geometry_record(decoder: ManoDecoder, packet: dict) -> dict:
    decoded = decoder.decode(packet)
    joints = decoded["joints_camera"]
    return {"joints_3d_root": decoded["joints_model"].tolist(), "joints_3d_camera": joints.tolist(),
            "joints_2d_xy": decoded["joints_2d"].tolist(), "cam_t_full": packet["translation_camera"],
            "positive_depth": decoded["positive_depth"].tolist()}


def process_sequence(run: Path, out: Path, mano_dir: Path, max_gap: int = 3,
                     smoothing_seconds: float = 0.08, fill_gaps: bool = False) -> dict:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new, empty sequence output directory")
    run_index = load_json(run / "index.json")["records"]
    if not run_index:
        raise ValueError("No completed frames")
    groups = {}
    for entry in run_index:
        if entry["timestamp_s"] is None:
            raise ValueError("Timestamps required: process a video or use --sequence-fps for image sequences")
        groups.setdefault(entry["sequence_id"], []).append(entry)
    for entries in groups.values():
        entries.sort(key=lambda r: r["frame_index"])
    decoder = ManoDecoder.from_directory(mano_dir)
    out.mkdir(parents=True, exist_ok=True)
    count_observed = count_filled = 0
    new_index = []
    track_count = 0
    source_hashes = {}
    traces = []
    source_spacing = {}
    for sequence_id, entries in groups.items():
        tracker = HandTracker(max_gap, smoothing_seconds)
        source_spacing[sequence_id] = float(np.median(np.diff([e["timestamp_s"] for e in entries]))) if len(entries) > 1 else None
        for index, entry in enumerate(entries):
            source_hashes[entry["record"]] = file_hash(run / entry["record"])
            record = load_json(run / entry["record"])
            hands, gaps = tracker.update(record["hands"], index, entry["timestamp_s"])
            for hand in hands:
                hand["processed"] = geometry_record(decoder, hand["processed_mano"])
                traces.append({"sequence_id": sequence_id, "track_id": hand["track_id"],
                               "frame_index": entry["frame_index"], "timestamp_s": entry["timestamp_s"],
                               "raw_joints": hand["joints_3d_camera"],
                               "processed_joints": hand["processed"]["joints_3d_camera"]})
            record["hands"] = hands
            record["postprocessing"] = {"smoothing_seconds": smoothing_seconds,
                                        "raw_predictions_preserved": True, "fill_gaps": fill_gaps}
            output_path = out / entry["record"]
            save_json(output_path, record)
            new_index.append(dict(entry))
            count_observed += len(hands)
            if fill_gaps:
                for gap in gaps:
                    for missing_index in range(gap["start"] + 1, gap["end"]):
                        missing_entry = entries[missing_index]
                        path = out / missing_entry["record"]
                        missing_frame = load_json(path)
                        weight = (missing_entry["timestamp_s"] - gap["start_time"]) / (gap["end_time"] - gap["start_time"])
                        packet = interpolate_packets(gap["first"], gap["second"], weight)
                        missing_frame["hands"].append({"track_id": gap["track_id"],
                            "hand_side": packet["hand_side"], "observation": "interpolated",
                            "processed_mano": packet, "processed": geometry_record(decoder, packet),
                            "interpolation": {"from_sample": entries[gap["start"]]["sample_id"],
                                              "to_sample": entries[gap["end"]]["sample_id"], "weight": weight}})
                        save_json(path, missing_frame)
                        count_filled += 1
        track_count += tracker.next_id - 1
    # Compute velocities only on consecutive detected observations, never gap fills.
    speeds = {"raw": [], "processed": []}
    previous = {}
    for row in traces:
        key = (row["sequence_id"], row["track_id"])
        prior = previous.get(key)
        dt = row["timestamp_s"] - prior["timestamp_s"] if prior else None
        spacing = source_spacing[row["sequence_id"]]
        if prior and spacing is not None and dt <= 1.5 * spacing:
            for variant in speeds:
                delta = np.asarray(row[f"{variant}_joints"]) - np.asarray(prior[f"{variant}_joints"])
                speeds[variant].append(float(np.linalg.norm(delta, axis=1).mean() / dt))
        previous[key] = row
    summary = {"schema": "egohand3d.sequence/1", "source_run": str(run.resolve()),
               "frames": len(new_index), "sequences": len(groups), "tracks": track_count,
               "detected_hands": count_observed, "interpolated_hands": count_filled,
               "max_gap_samples": max_gap, "smoothing_seconds": smoothing_seconds,
               "mean_joint_speed_m_s": {k: float(np.mean(v)) if v else None for k, v in speeds.items()},
               "speed_note": "Descriptive motion statistic, not accuracy or a ground-truth jitter metric",
               "source_record_sha256": source_hashes}
    save_json(out / "index.json", {"schema": "egohand3d.index/1", "records": new_index})
    save_json(out / "summary.json", summary)
    from .reports import build_sequence_report
    build_sequence_report(out, traces, summary)
    return summary
