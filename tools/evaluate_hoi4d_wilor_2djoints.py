from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_compare_overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, title: str) -> np.ndarray:
    out = image.copy()
    pred_i = np.round(pred).astype(int)
    gt_i = np.round(gt).astype(int)

    for a, b in CONNECTIONS:
        cv2.line(out, tuple(gt_i[a]), tuple(gt_i[b]), (0, 220, 0), 4, cv2.LINE_AA)
        cv2.line(out, tuple(pred_i[a]), tuple(pred_i[b]), (0, 90, 255), 2, cv2.LINE_AA)

    for x, y in gt_i:
        cv2.circle(out, (int(x), int(y)), 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 5, (0, 220, 0), -1, cv2.LINE_AA)

    for x, y in pred_i:
        cv2.circle(out, (int(x), int(y)), 7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 4, (0, 90, 255), -1, cv2.LINE_AA)

    cv2.putText(
        out,
        "GT green, WiLoR orange/red",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        title,
        (18, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return out


def write_frame_csv(path: Path, pred: np.ndarray, gt: np.ndarray, err: np.ndarray) -> None:
    rows = ["joint,pred_x,pred_y,gt_x,gt_y,error_px"]
    for joint, (p, g, e) in enumerate(zip(pred, gt, err)):
        rows.append(f"{joint},{p[0]:.6f},{p[1]:.6f},{g[0]:.6f},{g[1]:.6f},{e:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_contact_sheet(image_paths: list[Path], out_path: Path, cols: int = 5, thumb_w: int = 384) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_w / max(w, 1)
        thumb_h = int(round(h * scale))
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


def load_gt_kps(label_path: Path) -> np.ndarray:
    with label_path.open("rb") as f:
        ann = pickle.load(f)
    return np.asarray(ann["kps2D"], dtype=np.float64).reshape(21, 2)


def select_hand(hands: list[dict], gt: np.ndarray, expected_right: bool) -> tuple[int, dict, np.ndarray, np.ndarray]:
    candidates = [(idx, h) for idx, h in enumerate(hands) if bool(h.get("is_right")) == expected_right]
    if not candidates:
        candidates = list(enumerate(hands))
    scored = []
    for idx, hand in candidates:
        pred = np.asarray(hand["kpts_2d_xy"], dtype=np.float64).reshape(21, 2)
        err = np.linalg.norm(pred - gt, axis=1)
        scored.append((float(err.mean()), idx, hand, pred, err))
    _, idx, hand, pred, err = min(scored, key=lambda x: x[0])
    return idx, hand, pred, err


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare WiLoR 2D joints JSONs with HOI4D sample pickle kps2D.")
    parser.add_argument("--sample_dir", type=Path, default=Path("hoi4d_samples2"))
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--no_overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.sample_dir / "meta" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    csv_dir = args.out_dir / "csv"
    overlay_dir = args.out_dir / "overlays"
    pred_npy_dir = args.out_dir / "wilor_pred_kps2d_npy"
    pred_txt_dir = args.out_dir / "wilor_pred_kps2d_txt"
    for path in [csv_dir, pred_npy_dir, pred_txt_dir]:
        path.mkdir(parents=True, exist_ok=True)
    if not args.no_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    missed_images = []
    all_errors = []
    overlay_paths: list[Path] = []

    for item in manifest:
        image_path = args.sample_dir / item["image"]
        label_path = args.sample_dir / item["label"]
        gt = load_gt_kps(label_path)
        stem = image_path.stem
        pred_path = args.pred_dir / f"{stem}_kpts2d.json"
        expected_right = item["hand_side"].lower() == "right"

        base_record = {
            "sample_id": item["sample_id"],
            "image": image_path.name,
            "hand_side": item["hand_side"],
            "pred_json": str(pred_path),
        }

        if not pred_path.exists():
            missed_images.append(stem)
            frames.append({**base_record, "detected": False, "num_hands": 0})
            print(f"[{stem}] missed: no prediction json")
            continue

        pred_json = json.loads(pred_path.read_text(encoding="utf-8"))
        hands = pred_json.get("hands", [])
        if not hands:
            missed_images.append(stem)
            frames.append({**base_record, "detected": False, "num_hands": 0})
            print(f"[{stem}] missed: empty hands")
            continue

        selected_idx, selected_hand, pred, err = select_hand(hands, gt, expected_right)
        all_errors.append(err)

        np.save(pred_npy_dir / f"{stem}_wilor_pred_kps2d.npy", pred.astype(np.float32))
        np.savetxt(pred_txt_dir / f"{stem}_wilor_pred_kps2d.txt", pred, fmt="%.6f")
        csv_path = csv_dir / f"{stem}_wilor_vs_gt.csv"
        write_frame_csv(csv_path, pred, gt, err)

        overlay_path = None
        if not args.no_overlays:
            image = cv2.imread(str(image_path))
            if image is not None:
                overlay = draw_compare_overlay(
                    image,
                    pred,
                    gt,
                    f"mean={err.mean():.3f}px median={np.median(err):.3f}px",
                )
                overlay_path = overlay_dir / f"{stem}_wilor_vs_gt.jpg"
                cv2.imwrite(str(overlay_path), overlay)
                overlay_paths.append(overlay_path)

        record = {
            **base_record,
            "detected": True,
            "num_hands": len(hands),
            "selected_hand": int(selected_idx),
            "is_right": bool(selected_hand.get("is_right")),
            "mean_err_px": float(err.mean()),
            "median_err_px": float(np.median(err)),
            "rmse_err_px": float(np.sqrt(np.mean(err ** 2))),
            "min_err_px": float(err.min()),
            "max_err_px": float(err.max()),
            "max_err_joint": int(np.argmax(err)),
            "csv": str(csv_path),
            "overlay": str(overlay_path) if overlay_path else None,
            "per_joint_error_px": err.tolist(),
        }
        frames.append(record)
        print(
            f"[{stem}] hands={len(hands)} selected={selected_idx} "
            f"right={record['is_right']} mean={record['mean_err_px']:.4f}px "
            f"median={record['median_err_px']:.4f}px max={record['max_err_px']:.4f}px"
        )

    if all_errors:
        all_err = np.stack(all_errors, axis=0)
        overall = {
            "num_images": int(len(frames)),
            "num_detected": int(len(all_errors)),
            "num_missed": int(len(missed_images)),
            "num_eval_joints": int(all_err.size),
            "mean_err_px": float(all_err.mean()),
            "median_err_px": float(np.median(all_err)),
            "rmse_err_px": float(np.sqrt(np.mean(all_err ** 2))),
            "min_err_px": float(all_err.min()),
            "max_err_px": float(all_err.max()),
        }
        per_joint = []
        for joint in range(all_err.shape[1]):
            vals = all_err[:, joint]
            per_joint.append(
                {
                    "joint": joint,
                    "mean_err_px": float(vals.mean()),
                    "median_err_px": float(np.median(vals)),
                    "max_err_px": float(vals.max()),
                }
            )
    else:
        overall = {
            "num_images": int(len(frames)),
            "num_detected": 0,
            "num_missed": int(len(missed_images)),
            "num_eval_joints": 0,
        }
        per_joint = []

    summary = {
        "metric": "Euclidean pixel error per joint: WiLoR kpts_2d_xy vs GT pickle kps2D",
        "sample_dir": str(args.sample_dir),
        "manifest": str(manifest_path),
        "pred_dir": str(args.pred_dir),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "overall": overall,
        "missed_images": missed_images,
        "frames": frames,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_rows = [
        "image,detected,num_hands,selected_hand,is_right,mean_err_px,median_err_px,rmse_err_px,min_err_px,max_err_px,max_err_joint,csv,overlay"
    ]
    for record in frames:
        summary_rows.append(
            f"{record['image']},{record.get('detected', False)},"
            f"{record.get('num_hands', 0)},{record.get('selected_hand', '')},"
            f"{record.get('is_right', '')},{record.get('mean_err_px', '')},"
            f"{record.get('median_err_px', '')},{record.get('rmse_err_px', '')},"
            f"{record.get('min_err_px', '')},{record.get('max_err_px', '')},"
            f"{record.get('max_err_joint', '')},{record.get('csv', '')},"
            f"{record.get('overlay', '') or ''}"
        )
    (args.out_dir / "summary.csv").write_text("\n".join(summary_rows) + "\n", encoding="utf-8")

    (args.out_dir / "per_joint_summary.json").write_text(
        json.dumps(per_joint, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pj_rows = ["joint,mean_err_px,median_err_px,max_err_px"]
    for item in per_joint:
        pj_rows.append(f"{item['joint']},{item['mean_err_px']:.10f},{item['median_err_px']:.10f},{item['max_err_px']:.10f}")
    (args.out_dir / "per_joint_summary.csv").write_text("\n".join(pj_rows) + "\n", encoding="utf-8")

    if overlay_paths:
        write_contact_sheet(overlay_paths[:25], args.out_dir / "contact_sheet_first_25.jpg")
        detected_frames = [r for r in frames if r.get("detected")]
        worst = sorted(detected_frames, key=lambda x: x["mean_err_px"], reverse=True)[:25]
        write_contact_sheet([Path(item["overlay"]) for item in worst if item.get("overlay")], args.out_dir / "contact_sheet_worst_25.jpg")

    print(f"[summary] {args.out_dir / 'summary.json'}")
    print(
        f"[overall] n={overall['num_images']} detected={overall['num_detected']} "
        f"missed={overall['num_missed']} mean={overall.get('mean_err_px', float('nan')):.6f}px "
        f"median={overall.get('median_err_px', float('nan')):.6f}px "
        f"rmse={overall.get('rmse_err_px', float('nan')):.6f}px "
        f"max={overall.get('max_err_px', float('nan')):.6f}px"
    )


if __name__ == "__main__":
    main()
