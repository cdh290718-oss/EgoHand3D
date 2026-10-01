from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_COLORS = [
    (60, 220, 255),
    (80, 200, 120),
    (255, 180, 80),
    (180, 120, 255),
    (255, 100, 130),
]


def hand_color(is_right: bool) -> tuple[int, int, int]:
    return (40, 190, 255) if is_right else (255, 130, 40)


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_bbox(
    image: np.ndarray,
    bbox_xyxy: list[float] | None,
    label: str,
    color: tuple[int, int, int],
) -> None:
    if not bbox_xyxy:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    draw_label(image, label, (x1, max(18, y1 - 6)), color)


def normalize_scores(points: np.ndarray, scores: list[float] | None) -> np.ndarray:
    if scores is None:
        return np.ones((points.shape[0],), dtype=np.float32)
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if arr.size != points.shape[0]:
        return np.ones((points.shape[0],), dtype=np.float32)
    return arr


def draw_hand_skeleton(
    image: np.ndarray,
    points_xy: np.ndarray,
    scores: list[float] | None = None,
    score_threshold: float = 0.0,
    line_thickness: int = 2,
    point_radius: int = 4,
) -> np.ndarray:
    out = image
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    conf = normalize_scores(points, scores)
    for idx, (a, b) in enumerate(HAND_CONNECTIONS):
        if a >= len(points) or b >= len(points):
            continue
        if conf[a] < score_threshold or conf[b] < score_threshold:
            continue
        color = FINGER_COLORS[min(idx // 4, len(FINGER_COLORS) - 1)]
        pa = tuple(np.round(points[a]).astype(int).tolist())
        pb = tuple(np.round(points[b]).astype(int).tolist())
        cv2.line(out, pa, pb, color, line_thickness, cv2.LINE_AA)
    for joint, (x, y) in enumerate(points):
        if conf[joint] < score_threshold:
            continue
        cv2.circle(out, (int(round(x)), int(round(y))), point_radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (int(round(x)), int(round(y))), point_radius, (255, 255, 255), -1, cv2.LINE_AA)
    return out


def hand_points_2d(hand: dict) -> tuple[np.ndarray | None, list[float] | None]:
    for key in ("kpts_2d_xy", "joints_2d_xy"):
        if key in hand:
            points = np.asarray(hand[key], dtype=np.float32)
            scores = hand.get("kpts_2d_scores") or hand.get("joints_2d_scores")
            return points, scores
    return None, None


def draw_record_on_image(
    image_bgr: np.ndarray,
    record: dict,
    score_threshold: float = 0.0,
    draw_boxes: bool = True,
    draw_indices: bool = False,
) -> np.ndarray:
    out = image_bgr.copy()
    for hand in record.get("hands", []):
        is_right = bool(hand.get("is_right", False))
        color = hand_color(is_right)
        label = f"{'R' if is_right else 'L'}#{hand.get('hand_index', 0)}"
        if "bbox_score" in hand:
            label += f" {float(hand['bbox_score']):.2f}"
        if draw_boxes:
            draw_bbox(out, hand.get("bbox_xyxy"), label, color)
        points, scores = hand_points_2d(hand)
        if points is None:
            continue
        draw_hand_skeleton(out, points, scores=scores, score_threshold=score_threshold)
        if draw_indices:
            for idx, (x, y) in enumerate(points):
                cv2.putText(
                    out,
                    str(idx),
                    (int(round(x)) + 4, int(round(y)) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    return out


def make_contact_sheet(
    image_paths: list[Path],
    out_path: Path,
    cols: int = 5,
    thumb_w: int = 384,
) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_w / max(w, 1)
        thumb_h = max(1, int(round(h * scale)))
        thumbs.append(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    thumb_h = max(t.shape[0] for t in thumbs)
    rows = []
    for start in range(0, len(thumbs), cols):
        row_imgs = []
        for thumb in thumbs[start:start + cols]:
            if thumb.shape[0] < thumb_h:
                pad = np.zeros((thumb_h - thumb.shape[0], thumb.shape[1], 3), dtype=thumb.dtype)
                thumb = np.vstack([thumb, pad])
            row_imgs.append(thumb)
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))
        rows.append(np.hstack(row_imgs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))
