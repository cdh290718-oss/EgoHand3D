from pathlib import Path
import argparse
import os
import json
import re

import cv2
import numpy as np
import torch

from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.utils.camera import cam_crop_to_full
from wilor.utils.yolo_loader import load_yolo_detector


def _compact_2d_points(json_str: str) -> str:
    """把 JSON 中每个 [x, y] 内层数组折叠到同一行，方便阅读。"""
    return re.sub(
        r'\[\s*\n\s*([-\d.eE+]+),\s*\n\s*([-\d.eE+]+)\s*\n\s*\]',
        r'[\1, \2]',
        json_str,
    )


def project_full_img(points_xyz: np.ndarray, cam_trans: np.ndarray, focal_length: float, img_res_wh):
    """把 3D 点投影到整张图上的 2D 像素坐标系。

    Args:
        points_xyz: (N, 3) 的 3D 点（WiLoR 输出的相机坐标系下点，后面会加上相机平移）
        cam_trans: (3,) 相机平移（恢复到整图坐标系后的平移）
        focal_length: 焦距（像素单位）
        img_res_wh: (W, H)，用于确定主点 (cx, cy)
    """
    w, h = float(img_res_wh[0]), float(img_res_wh[1])
    # 主点默认取图像中心（像素坐标系：x 向右，y 向下）
    camera_center = (w / 2.0, h / 2.0)  # (cx, cy)

    # 简化 pinhole 相机内参矩阵 K
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = float(focal_length)
    K[1, 1] = float(focal_length)
    K[0, 2] = float(camera_center[0])
    K[1, 2] = float(camera_center[1])

    # 转到相机平移后的坐标，再做透视除法得到归一化平面坐标
    points = points_xyz.astype(np.float32) + cam_trans.astype(np.float32)
    points = points / points[..., -1:]
    uvw = (K @ points.T).T
    return uvw[..., :2]


def main():
    parser = argparse.ArgumentParser(description="WiLoR demo: export 2D joints (21 keypoints)")
    # 输入/输出参数
    parser.add_argument("--img_folder", type=str, default="./", help="Folder with input images")
    parser.add_argument("--out_folder", type=str, default="demo_out", help="Output folder to save results")
    # bbox 外扩比例：会影响裁剪范围，太小可能截断手部，太大会引入背景
    parser.add_argument("--rescale_factor", type=float, default=2.0, help="Factor for padding the bbox")
    parser.add_argument("--file_type", nargs="+", default=["*.jpg", "*.png", "*.jpeg"], help="Image extensions")
    parser.add_argument("--checkpoint", type=str, default="./pretrained_models/wilor_final.ckpt", help="WiLoR checkpoint path")
    parser.add_argument("--cfg", type=str, default="./pretrained_models/model_config.yaml", help="WiLoR model config path")
    parser.add_argument("--detector", type=str, default="./pretrained_models/detector.pt", help="YOLO hand detector path")
    parser.add_argument(
        "--save_per_hand",
        action="store_true",
        default=False,
        help="If set, save one json per detected hand (otherwise one merged json per image).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    # 加载 WiLoR（用于回归 3D joints）与 YOLO 检测器（用于给出手的 bbox）
    model, model_cfg = load_wilor(
        checkpoint_path=args.checkpoint,
        cfg_path=args.cfg,
    )
    detector = load_yolo_detector(args.detector)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device).eval()
    detector = detector.to(device)

    # 收集待处理图片
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            continue

        # 1) 手部检测，拿到每只手的 bbox 与左右手标记
        detections = detector(img_cv2, conf=0.3, verbose=False)[0]
        bboxes = []
        is_right = []
        for det in detections:
            bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(bbox[:4].tolist())

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # 2) 根据 bbox 构建裁剪数据，WiLoR 的输入是“裁剪后的手部 patch”
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        # 默认：每张图输出一个 json，里面包含多只手
        merged = {
            "image": os.path.basename(str(img_path)),
            "hands": [],
        }

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                # 3) WiLoR 前向：得到相机参数、3D joints 等
                out = model(batch)

            # 4) 左右手对齐：对相机 x 方向做镜像，保证后续还原一致
            multiplier = (2 * batch["right"] - 1)
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()

            # 5) 将裁剪坐标系下的相机参数，恢复到“整图坐标系”的相机平移 cam_t
            scaled_focal_length = (
                model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            )
            scaled_focal_length_f = float(scaled_focal_length.detach().cpu().item())

            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()

            bs = batch["img"].shape[0]
            for n in range(bs):
                # 6) 取 21 个 3D 关节点，并对左手在 x 方向镜像到统一坐标系
                joints = out["pred_keypoints_3d"][n].detach().cpu().numpy()  # (21, 3)
                right_n = float(batch["right"][n].cpu().numpy())
                joints[:, 0] = (2 * right_n - 1) * joints[:, 0]

                cam_t = pred_cam_t_full[n]
                img_res_wh = img_size[n].detach().cpu().numpy()  # [W, H]

                # 7) 把 3D joints 投影到整图像素坐标，得到 (21, 2) 的 (x, y)
                kpts_2d = project_full_img(joints, cam_t, scaled_focal_length_f, img_res_wh)  # (21, 2)

                hand_rec = {
                    "hand_index": int(n),
                    "is_right": bool(right_n),
                    "kpts_2d_xy": kpts_2d.tolist(),
                }

                if args.save_per_hand:
                    # 每只手一个 json：方便下游直接按 hand_index 读取
                    out_path = os.path.join(args.out_folder, f"{img_fn}_{n}_kpts2d.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        text = json.dumps(
                            {"image": merged["image"], **hand_rec},
                            ensure_ascii=False,
                            indent=2,
                        )
                        f.write(_compact_2d_points(text))
                else:
                    merged["hands"].append(hand_rec)

        if not args.save_per_hand:
            # 每张图一个 json：里面是 hands 列表
            out_path = os.path.join(args.out_folder, f"{img_fn}_kpts2d.json")
            with open(out_path, "w", encoding="utf-8") as f:
                text = json.dumps(merged, ensure_ascii=False, indent=2)
                f.write(_compact_2d_points(text))


if __name__ == "__main__":
    main()
