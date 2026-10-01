"""Explicit evaluation with fixed manifests and prediction-independent matching."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def similarity_align(prediction, target) -> np.ndarray:
    x, y = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError("Alignment requires matching Nx3 arrays, N >= 3")
    xc, yc = x - x.mean(0), y - y.mean(0)
    variance = np.sum(xc * xc)
    if variance < 1e-12 or np.linalg.matrix_rank(xc) < 2 or np.linalg.matrix_rank(yc) < 2:
        raise ValueError("Degenerate joints cannot define a similarity alignment")
    u, singular, vt = np.linalg.svd(xc.T @ yc)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    scale = np.sum(singular * np.diag(correction)) / variance
    return scale * xc @ rotation + y.mean(0)


def bbox_iou(first, second) -> float:
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite([a, b]).all():
        raise ValueError("Bounding boxes must be finite xyxy arrays")
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.


def match_hands(predictions: list[dict], targets: list[dict], threshold: float = 0.1):
    """One-to-one same-side IoU matching; never choose by joint error."""
    observed = [i for i, p in enumerate(predictions) if p.get("observation", "detected") == "detected"]
    if not observed or not targets:
        return []
    cost = np.full((len(observed), len(targets)), 1e6)
    for i, pi in enumerate(observed):
        for j, target in enumerate(targets):
            prediction = predictions[pi]
            if prediction["hand_side"] == target["hand_side"]:
                overlap = bbox_iou(prediction["bbox_xyxy"], target["bbox_xyxy"])
                if overlap >= threshold:
                    cost[i, j] = 1 - overlap
    ii, jj = linear_sum_assignment(cost)
    return [(observed[int(i)], int(j)) for i, j in zip(ii, jj) if cost[i, j] < 1e6]


def error_summary(values) -> dict:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {"count": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95))}


def evaluate_frames(targets: list[dict], predictions: dict, variant: str = "raw",
                    match_iou: float = 0.1, pck_thresholds=(5., 10., 20., 50.)) -> dict:
    if variant not in {"raw", "processed"} or not 0 <= match_iou <= 1:
        raise ValueError("Invalid variant/matching threshold")
    if not pck_thresholds or any(t <= 0 for t in pck_thresholds):
        raise ValueError("PCK thresholds must be positive")
    ids = [str(frame["sample_id"]) for frame in targets]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ground-truth sample IDs")
    errors = {"2d_px": [], "camera_mpjpe_mm": [], "root_mpjpe_mm": [], "pa_mpjpe_mm": []}
    gt_hands = matched = detected = total_2d = degenerate_pa = missing_frames = 0
    details = []
    for frame in targets:
        record = predictions.get(str(frame["sample_id"]))
        hands = record.get("hands", []) if record else []
        missing_frames += record is None
        labels = frame["hands"]
        gt_hands += len(labels)
        detected += sum(h.get("observation", "detected") == "detected" for h in hands)
        valid_labels = []
        for target in labels:
            if target["hand_side"] not in {"left", "right"}:
                raise ValueError("GT hand_side must be left/right")
            bbox_iou(target["bbox_xyxy"], target["bbox_xyxy"])
            valid = np.asarray(target.get("valid", [True] * 21), dtype=bool)
            if valid.shape != (21,):
                raise ValueError("GT valid mask must have 21 entries")
            for key, dim in [("joints_2d_xy", 2), ("joints_3d_camera", 3)]:
                if key in target:
                    points = np.asarray(target[key], dtype=float)
                    if points.shape != (21, dim) or not np.isfinite(points[valid]).all():
                        raise ValueError(f"Invalid GT {key}")
            if "joints_2d_xy" in target:
                total_2d += int(valid.sum())
            valid_labels.append(valid)
        pairs = match_hands(hands, labels, match_iou)
        matched += len(pairs)
        for pi, gi in pairs:
            target = labels[gi]
            hand = hands[pi]
            predicted = hand if variant == "raw" else hand.get("processed")
            if predicted is None:
                raise ValueError("Processed evaluation requires processed geometry for every matched hand")
            valid = valid_labels[gi]
            row = {"sample_id": frame["sample_id"], "gt_hand_index": gi, "prediction_index": pi}
            if "joints_2d_xy" in target:
                points = np.asarray(predicted["joints_2d_xy"], dtype=float)
                if points.shape != (21, 2) or not np.isfinite(points).all():
                    raise ValueError("Invalid predicted 2D joints")
                err = np.linalg.norm(points[valid] - np.asarray(target["joints_2d_xy"])[valid], axis=1)
                errors["2d_px"].extend(err.tolist())
                row["mean_2d_px"] = float(err.mean()) if len(err) else None
            if "joints_3d_camera" in target:
                points = np.asarray(predicted["joints_3d_camera"], dtype=float)
                truth = np.asarray(target["joints_3d_camera"], dtype=float)
                if points.shape != (21, 3) or not np.isfinite(points).all():
                    raise ValueError("Invalid predicted 3D joints")
                camera = np.linalg.norm(points[valid] - truth[valid], axis=1) * 1000
                errors["camera_mpjpe_mm"].extend(camera.tolist())
                if valid[0]:
                    root = np.linalg.norm((points - points[0])[valid] - (truth - truth[0])[valid], axis=1) * 1000
                    errors["root_mpjpe_mm"].extend(root.tolist())
                try:
                    aligned = similarity_align(points[valid], truth[valid])
                    pa = np.linalg.norm(aligned - truth[valid], axis=1) * 1000
                    errors["pa_mpjpe_mm"].extend(pa.tolist())
                except ValueError:
                    degenerate_pa += 1
            details.append(row)
    pixel_errors = np.asarray(errors["2d_px"])
    pck = {str(float(threshold)): {"matched_only": float(np.mean(pixel_errors <= threshold)) if pixel_errors.size else None,
           "all_gt_joints": int(np.sum(pixel_errors <= threshold)) / total_2d if total_2d else None}
           for threshold in sorted(set(pck_thresholds))}
    return {"schema": "egohand3d.benchmark/1", "variant": variant, "frames": len(targets),
            "missing_prediction_frames": missing_frames, "gt_hands": gt_hands, "predicted_hands": detected,
            "matched_hands": matched, "missed_hands": gt_hands - matched, "unmatched_predictions": detected - matched,
            "hand_recall": matched / gt_hands if gt_hands else None,
            "hand_precision": matched / detected if detected else None,
            "metrics": {key: error_summary(value) for key, value in errors.items()},
            "pck_px": pck, "num_gt_2d_joints": total_2d, "degenerate_pa_hands": degenerate_pa,
            "protocol": {"matching": "one_to_one_same_side_bbox_iou", "minimum_iou": match_iou,
                         "interpolated_hands": "excluded", "3d_unit": "m", "joint_order": "OpenPose21",
                         "mean_error_denominator": "valid joints in matched hands; report recall alongside",
                         "all_gt_pck": "unmatched/missing hands count as incorrect"}, "details": details}
