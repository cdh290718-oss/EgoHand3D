from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np
import torch



KPT_SCORE_SOURCE = (
    "WiLoR does not output independent keypoint confidence. "
    "Each keypoint score is the hand detector score when the projected point is "
    "inside the frame with positive depth, otherwise 0."
)


def compact_points(json_str: str) -> str:
    return re.sub(
        r"\[\s*\n\s*([-\d.eE+]+),\s*\n\s*([-\d.eE+]+)\s*\n\s*\]",
        r"[\1, \2]",
        json_str,
    )


def save_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(compact_points(json.dumps(record, ensure_ascii=False, indent=2)), encoding="utf-8")


def project_full_img(
    points_xyz: np.ndarray,
    cam_trans: np.ndarray,
    focal_length: float,
    img_res_wh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    w, h = float(img_res_wh[0]), float(img_res_wh[1])
    camera_center = (w / 2.0, h / 2.0)

    k = np.eye(3, dtype=np.float32)
    k[0, 0] = float(focal_length)
    k[1, 1] = float(focal_length)
    k[0, 2] = float(camera_center[0])
    k[1, 2] = float(camera_center[1])

    points = points_xyz.astype(np.float32) + cam_trans.astype(np.float32)
    depth = points[..., 2].copy()
    points = points / points[..., -1:]
    uvw = (k @ points.T).T
    return uvw[..., :2], depth


def keypoint_scores_from_visibility(
    kpts_2d: np.ndarray,
    depth: np.ndarray,
    img_res_wh: np.ndarray,
    detector_score: float,
) -> list[float]:
    w, h = float(img_res_wh[0]), float(img_res_wh[1])
    valid = (
        (depth > 0)
        & (kpts_2d[:, 0] >= 0)
        & (kpts_2d[:, 0] < w)
        & (kpts_2d[:, 1] >= 0)
        & (kpts_2d[:, 1] < h)
    )
    return [float(detector_score) if ok else 0.0 for ok in valid.tolist()]


def run_frame(
    img_bgr: np.ndarray,
    model,
    model_cfg,
    detector,
    device: torch.device,
    detector_conf: float,
    rescale_factor: float,
    batch_size: int,
) -> list[dict]:
    result = detector(img_bgr, conf=detector_conf, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    det_scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    right = result.boxes.cls.detach().cpu().numpy().astype(np.float32)

    dataset = ViTDetDataset(model_cfg, img_bgr, boxes, right, rescale_factor=rescale_factor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    hands: list[dict] = []
    source_offset = 0
    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        multiplier = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]

        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = (
            model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        )
        scaled_focal_length_f = float(scaled_focal_length.detach().cpu().item())
        pred_cam_t_full = cam_crop_to_full(
            pred_cam, box_center, box_size, img_size, scaled_focal_length
        ).detach().cpu().numpy()

        current_batch_size = batch["img"].shape[0]
        for n in range(current_batch_size):
            source_idx = source_offset + n
            is_right = float(batch["right"][n].detach().cpu().numpy())
            joints = out["pred_keypoints_3d"][n].detach().cpu().numpy()
            joints[:, 0] = (2 * is_right - 1) * joints[:, 0]

            img_res_wh = img_size[n].detach().cpu().numpy()
            kpts_2d, depth = project_full_img(
                joints,
                pred_cam_t_full[n],
                scaled_focal_length_f,
                img_res_wh,
            )
            detector_score = float(det_scores[source_idx])

            hands.append(
                {
                    "hand_index": len(hands),
                    "is_right": bool(is_right),
                    "bbox_xyxy": boxes[source_idx].astype(float).tolist(),
                    "bbox_score": detector_score,
                    "kpts_2d_xy": kpts_2d.astype(float).tolist(),
                    "kpts_2d_scores": keypoint_scores_from_visibility(
                        kpts_2d,
                        depth,
                        img_res_wh,
                        detector_score,
                    ),
                    "kpts_score_source": KPT_SCORE_SOURCE,
                }
            )
        source_offset += current_batch_size

    return hands


def draw_hands(img_bgr: np.ndarray, hands: list[dict], score_threshold: float) -> np.ndarray:
    vis = img_bgr.copy()
    for hand in hands:
        x1, y1, x2, y2 = [int(round(v)) for v in hand["bbox_xyxy"]]
        color = (40, 190, 255) if hand["is_right"] else (255, 130, 40)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{'R' if hand['is_right'] else 'L'} {hand['bbox_score']:.2f}"
        cv2.putText(
            vis,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        kpts_xy = np.asarray(hand["kpts_2d_xy"], dtype=np.float32)
        kpt_scores = np.asarray(hand["kpts_2d_scores"], dtype=np.float32)[:, None]
        kpts_xyc = np.concatenate([kpts_xy, kpt_scores], axis=1)
        vis = render_openpose(vis, kpts_xyc)

        for x, y, score in kpts_xyc:
            if score < score_threshold:
                continue
            cv2.circle(vis, (int(round(x)), int(round(y))), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return vis


def open_writer(output_video: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_video}")
    return writer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect hands in a video, draw hand boxes/keypoints, and save per-frame scores."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out_folder", default="video_hands_out", help="Output directory")
    parser.add_argument("--output_video", default=None, help="Output overlay video path")
    parser.add_argument("--checkpoint_path", default="./pretrained_models/wilor_final.ckpt")
    parser.add_argument("--cfg_path", default="./pretrained_models/model_config.yaml")
    parser.add_argument("--detector_path", default="./pretrained_models/detector.pt")
    parser.add_argument("--detector_conf", type=float, default=0.3)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1, help="Process and write every Nth frame")
    parser.add_argument("--max_frames", type=int, default=-1, help="Stop after N processed frames")
    parser.add_argument("--draw_score_threshold", type=float, default=0.1)
    args = parser.parse_args()

    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor
    from wilor.utils import recursive_to
    from wilor.utils.camera import cam_crop_to_full
    from wilor.utils.render_openpose import render_openpose
    from wilor.utils.yolo_loader import load_yolo_detector

    globals()["ViTDetDataset"] = ViTDetDataset
    globals()["recursive_to"] = recursive_to
    globals()["cam_crop_to_full"] = cam_crop_to_full
    globals()["render_openpose"] = render_openpose

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    out_root = Path(args.out_folder)
    json_dir = out_root / "jsons"
    jsonl_path = out_root / "frames.jsonl"
    output_video = Path(args.output_video) if args.output_video else out_root / "overlay.mp4"
    json_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_writer(output_video, fps / args.stride, width, height)

    print(f"[INFO] Loading WiLoR checkpoint: {args.checkpoint_path}")
    model, model_cfg = load_wilor(checkpoint_path=args.checkpoint_path, cfg_path=args.cfg_path)
    print(f"[INFO] Loading hand detector: {args.detector_path}")
    detector = load_yolo_detector(args.detector_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device).eval()
    detector = detector.to(device)

    summary = {
        "video": str(Path(args.video).resolve()),
        "overlay_video": str(output_video.resolve()),
        "fps": fps,
        "output_fps": fps / args.stride,
        "width": width,
        "height": height,
        "stride": args.stride,
        "detector_conf": args.detector_conf,
        "kpts_score_source": KPT_SCORE_SOURCE,
        "frames": [],
    }

    frame_index = 0
    processed = 0
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % args.stride != 0:
                frame_index += 1
                continue
            if args.max_frames >= 0 and processed >= args.max_frames:
                break

            hands = run_frame(
                img_bgr=frame,
                model=model,
                model_cfg=model_cfg,
                detector=detector,
                device=device,
                detector_conf=args.detector_conf,
                rescale_factor=args.rescale_factor,
                batch_size=args.batch_size,
            )

            record = {
                "frame_index": frame_index,
                "time_sec": frame_index / fps,
                "hands": hands,
            }
            frame_json = json_dir / f"frame_{frame_index:06d}.json"
            save_json(frame_json, record)
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

            writer.write(draw_hands(frame, hands, args.draw_score_threshold))
            summary["frames"].append(
                {
                    "frame_index": frame_index,
                    "time_sec": frame_index / fps,
                    "num_hands": len(hands),
                    "json": str(frame_json),
                }
            )

            processed += 1
            print(f"[{processed}] frame={frame_index}, hands={len(hands)}, json={frame_json}")
            frame_index += 1

    cap.release()
    writer.release()
    save_json(out_root / "summary.json", summary)

    print(f"[INFO] Done. Overlay video: {output_video}")
    print(f"[INFO] Per-frame JSON: {json_dir}")
    print(f"[INFO] JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
