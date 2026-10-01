from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.utils.camera import cam_crop_to_full
from wilor.utils.yolo_loader import load_yolo_detector

from .io_utils import collect_image_paths


@dataclass
class RuntimeBundle:
    model: Any
    model_cfg: Any
    detector: Any
    device: torch.device


def load_runtime(
    checkpoint: str,
    cfg: str,
    detector_path: str,
    device_name: str = "auto",
) -> RuntimeBundle:
    model, model_cfg = load_wilor(checkpoint_path=checkpoint, cfg_path=cfg)
    detector = load_yolo_detector(detector_path)
    if device_name == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device_name)
    model = model.to(device).eval()
    detector = detector.to(device)
    return RuntimeBundle(model=model, model_cfg=model_cfg, detector=detector, device=device)


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


def detection_arrays(detector_result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if detector_result.boxes is None or len(detector_result.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    boxes = detector_result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = detector_result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    right = detector_result.boxes.cls.detach().cpu().numpy().astype(np.float32)
    return boxes, scores, right


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


def hand_side_label(is_right: bool) -> str:
    return "right" if is_right else "left"


def _tensor_record(tensor: torch.Tensor, index: int):
    return tensor[index].detach().cpu().numpy()


def _mano_params_for_hand(pred_mano_params: dict[str, torch.Tensor], index: int) -> dict:
    out = {}
    for name, value in pred_mano_params.items():
        arr = _tensor_record(value, index)
        out[name] = {
            "shape": list(arr.shape),
            "values": arr.astype(float).tolist(),
        }
    return out


def infer_hands_from_image(
    img_bgr: np.ndarray,
    runtime: RuntimeBundle,
    detector_conf: float = 0.3,
    rescale_factor: float = 2.0,
    batch_size: int = 16,
    include_vertices: bool = False,
    include_mano: bool = False,
) -> list[dict]:
    result = runtime.detector(img_bgr, conf=detector_conf, verbose=False)[0]
    boxes, scores, right = detection_arrays(result)
    if len(boxes) == 0:
        return []

    dataset = ViTDetDataset(
        runtime.model_cfg,
        img_bgr,
        boxes,
        right,
        rescale_factor=rescale_factor,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    hands: list[dict] = []
    source_offset = 0
    for batch in dataloader:
        batch = recursive_to(batch, runtime.device)
        with torch.no_grad():
            out = runtime.model(batch)

        pred_cam = out["pred_cam"].clone()
        multiplier = 2 * batch["right"] - 1
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]

        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = (
            runtime.model_cfg.EXTRA.FOCAL_LENGTH
            / runtime.model_cfg.MODEL.IMAGE_SIZE
            * img_size.max()
        )
        scaled_focal_length_f = float(scaled_focal_length.detach().cpu().item())
        pred_cam_t_full = cam_crop_to_full(
            pred_cam,
            box_center,
            box_size,
            img_size,
            scaled_focal_length,
        ).detach().cpu().numpy()

        current_batch_size = batch["img"].shape[0]
        for n in range(current_batch_size):
            source_idx = source_offset + n
            is_right_f = float(batch["right"][n].detach().cpu().numpy())
            is_right = bool(is_right_f)
            joints = out["pred_keypoints_3d"][n].detach().cpu().numpy().astype(np.float32)
            joints[:, 0] = (2 * is_right_f - 1) * joints[:, 0]
            cam_t = pred_cam_t_full[n].astype(np.float32)
            img_res_wh = img_size[n].detach().cpu().numpy().astype(np.float32)
            kpts_2d, depth = project_full_img(
                joints,
                cam_t,
                scaled_focal_length_f,
                img_res_wh,
            )

            record = {
                "hand_index": len(hands),
                "source_index": int(source_idx),
                "is_right": is_right,
                "hand_side": hand_side_label(is_right),
                "bbox_xyxy": boxes[source_idx].astype(float).tolist(),
                "bbox_score": float(scores[source_idx]),
                "box_center_xy": box_center[n].detach().cpu().numpy().astype(float).tolist(),
                "box_size": float(box_size[n].detach().cpu().item()),
                "img_size_wh": img_res_wh.astype(float).tolist(),
                "focal_length_px": scaled_focal_length_f,
                "pred_cam_crop": pred_cam[n].detach().cpu().numpy().astype(float).tolist(),
                "cam_t_full": cam_t.astype(float).tolist(),
                "joints_3d_root": joints.astype(float).tolist(),
                "joints_3d_camera": (joints + cam_t).astype(float).tolist(),
                "joints_2d_xy": kpts_2d.astype(float).tolist(),
                "joints_depth": depth.astype(float).tolist(),
                "joints_2d_scores": keypoint_scores_from_visibility(
                    kpts_2d,
                    depth,
                    img_res_wh,
                    float(scores[source_idx]),
                ),
            }

            if include_vertices:
                verts = out["pred_vertices"][n].detach().cpu().numpy().astype(np.float32)
                verts[:, 0] = (2 * is_right_f - 1) * verts[:, 0]
                record["vertices_3d_root"] = verts.astype(float).tolist()
                record["vertices_3d_camera"] = (verts + cam_t).astype(float).tolist()

            if include_mano:
                record["pred_mano_params"] = _mano_params_for_hand(out["pred_mano_params"], n)

            hands.append(record)
        source_offset += current_batch_size

    return hands


def infer_image_path(
    image_path: Path,
    runtime: RuntimeBundle,
    detector_conf: float = 0.3,
    rescale_factor: float = 2.0,
    batch_size: int = 16,
    include_vertices: bool = False,
    include_mano: bool = False,
) -> dict:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return {
            "image": image_path.name,
            "image_path": str(image_path),
            "readable": False,
            "hands": [],
        }
    hands = infer_hands_from_image(
        img_bgr=img_bgr,
        runtime=runtime,
        detector_conf=detector_conf,
        rescale_factor=rescale_factor,
        batch_size=batch_size,
        include_vertices=include_vertices,
        include_mano=include_mano,
    )
    return {
        "image": image_path.name,
        "image_path": str(image_path),
        "readable": True,
        "width": int(img_bgr.shape[1]),
        "height": int(img_bgr.shape[0]),
        "num_hands": len(hands),
        "hands": hands,
    }


def run_folder(
    img_folder: Path,
    runtime: RuntimeBundle,
    file_type: list[str] | None = None,
    detector_conf: float = 0.3,
    rescale_factor: float = 2.0,
    batch_size: int = 16,
    include_vertices: bool = False,
    include_mano: bool = False,
) -> list[dict]:
    records = []
    for image_path in collect_image_paths(img_folder, file_type):
        records.append(
            infer_image_path(
                image_path=image_path,
                runtime=runtime,
                detector_conf=detector_conf,
                rescale_factor=rescale_factor,
                batch_size=batch_size,
                include_vertices=include_vertices,
                include_mano=include_mano,
            )
        )
    return records
