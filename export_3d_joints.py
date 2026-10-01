from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from egohand3d.inference import infer_image_path, load_runtime
from egohand3d.io_utils import collect_image_paths, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 21 3D hand joints, camera translation, and projected 2D joints."
    )
    parser.add_argument("--img_folder", type=Path, default=Path("./"), help="Folder with input images")
    parser.add_argument("--out_folder", type=Path, default=Path("outputs/3d_joints"))
    parser.add_argument("--file_type", nargs="+", default=["*.jpg", "*.png", "*.jpeg"])
    parser.add_argument("--checkpoint", default="./pretrained_models/wilor_final.ckpt")
    parser.add_argument("--cfg", default="./pretrained_models/model_config.yaml")
    parser.add_argument("--detector", default="./pretrained_models/detector.pt")
    parser.add_argument("--detector_conf", type=float, default=0.3)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--save_per_hand", action="store_true")
    parser.add_argument("--save_npy", action="store_true", help="Also save per-hand root/camera 3D joints as .npy")
    parser.add_argument("--include_vertices", action="store_true", help="Include MANO mesh vertices in JSON; large files")
    return parser.parse_args()


def slim_hand_record(hand: dict, include_vertices: bool) -> dict:
    keys = [
        "hand_index",
        "source_index",
        "is_right",
        "hand_side",
        "bbox_xyxy",
        "bbox_score",
        "box_center_xy",
        "box_size",
        "img_size_wh",
        "focal_length_px",
        "pred_cam_crop",
        "cam_t_full",
        "joints_3d_root",
        "joints_3d_camera",
        "joints_2d_xy",
        "joints_depth",
        "joints_2d_scores",
    ]
    if include_vertices:
        keys.extend(["vertices_3d_root", "vertices_3d_camera"])
    return {key: hand[key] for key in keys if key in hand}


def save_npy_outputs(out_folder: Path, image_stem: str, hand: dict) -> None:
    hand_idx = int(hand["hand_index"])
    npy_dir = out_folder / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_dir / f"{image_stem}_{hand_idx}_joints3d_root.npy", np.asarray(hand["joints_3d_root"], dtype=np.float32))
    np.save(npy_dir / f"{image_stem}_{hand_idx}_joints3d_camera.npy", np.asarray(hand["joints_3d_camera"], dtype=np.float32))
    np.save(npy_dir / f"{image_stem}_{hand_idx}_joints2d.npy", np.asarray(hand["joints_2d_xy"], dtype=np.float32))


def main() -> None:
    args = parse_args()
    args.out_folder.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(args.checkpoint, args.cfg, args.detector, device_name=args.device)
    image_paths = collect_image_paths(args.img_folder, args.file_type)
    records = []

    for image_path in image_paths:
        record = infer_image_path(
            image_path=image_path,
            runtime=runtime,
            detector_conf=args.detector_conf,
            rescale_factor=args.rescale_factor,
            batch_size=args.batch_size,
            include_vertices=args.include_vertices,
            include_mano=False,
        )
        image_stem = image_path.stem
        record["hands"] = [slim_hand_record(hand, args.include_vertices) for hand in record.get("hands", [])]
        records.append(record)

        if args.save_per_hand:
            for hand in record["hands"]:
                hand_record = {
                    "image": record["image"],
                    "image_path": record["image_path"],
                    "width": record.get("width"),
                    "height": record.get("height"),
                    **hand,
                }
                write_json(args.out_folder / f"{image_stem}_{hand['hand_index']}_joints3d.json", hand_record)
                if args.save_npy:
                    save_npy_outputs(args.out_folder, image_stem, hand)
        else:
            write_json(args.out_folder / f"{image_stem}_joints3d.json", record)
            if args.save_npy:
                for hand in record["hands"]:
                    save_npy_outputs(args.out_folder, image_stem, hand)

        print(f"[3d] {image_path.name}: hands={record.get('num_hands', 0)}")

    summary = {
        "img_folder": str(args.img_folder),
        "out_folder": str(args.out_folder),
        "num_images": len(records),
        "num_readable": sum(1 for r in records if r.get("readable")),
        "num_hands": sum(len(r.get("hands", [])) for r in records),
        "records": [
            {
                "image": r["image"],
                "readable": r.get("readable", False),
                "num_hands": len(r.get("hands", [])),
            }
            for r in records
        ],
    }
    write_json(args.out_folder / "summary.json", summary)
    write_jsonl(args.out_folder / "records.jsonl", records)
    print(f"[done] wrote {args.out_folder}")


if __name__ == "__main__":
    main()
