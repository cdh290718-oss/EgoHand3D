from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from egohand3d.inference import infer_image_path, load_runtime
from egohand3d.io_utils import collect_image_paths, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MANO pose/shape parameters, camera parameters, and hand metadata."
    )
    parser.add_argument("--img_folder", type=Path, default=Path("./"), help="Folder with input images")
    parser.add_argument("--out_folder", type=Path, default=Path("outputs/mano_params"))
    parser.add_argument("--file_type", nargs="+", default=["*.jpg", "*.png", "*.jpeg"])
    parser.add_argument("--checkpoint", default="./pretrained_models/wilor_final.ckpt")
    parser.add_argument("--cfg", default="./pretrained_models/model_config.yaml")
    parser.add_argument("--detector", default="./pretrained_models/detector.pt")
    parser.add_argument("--detector_conf", type=float, default=0.3)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--save_per_hand", action="store_true")
    parser.add_argument("--save_npz", action="store_true", help="Also save compact per-hand .npz files")
    return parser.parse_args()


def mano_hand_record(hand: dict) -> dict:
    return {
        "hand_index": hand["hand_index"],
        "source_index": hand["source_index"],
        "is_right": hand["is_right"],
        "hand_side": hand["hand_side"],
        "bbox_xyxy": hand["bbox_xyxy"],
        "bbox_score": hand["bbox_score"],
        "box_center_xy": hand["box_center_xy"],
        "box_size": hand["box_size"],
        "img_size_wh": hand["img_size_wh"],
        "focal_length_px": hand["focal_length_px"],
        "pred_cam_crop": hand["pred_cam_crop"],
        "cam_t_full": hand["cam_t_full"],
        "pred_mano_params": hand.get("pred_mano_params", {}),
    }


def save_npz_outputs(out_folder: Path, image_stem: str, hand: dict) -> None:
    npz_dir = out_folder / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    hand_idx = int(hand["hand_index"])
    params = hand.get("pred_mano_params", {})
    arrays = {
        "pred_cam_crop": np.asarray(hand["pred_cam_crop"], dtype=np.float32),
        "cam_t_full": np.asarray(hand["cam_t_full"], dtype=np.float32),
        "bbox_xyxy": np.asarray(hand["bbox_xyxy"], dtype=np.float32),
    }
    for name, payload in params.items():
        arrays[name] = np.asarray(payload["values"], dtype=np.float32)
    np.savez_compressed(npz_dir / f"{image_stem}_{hand_idx}_mano.npz", **arrays)


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
            include_vertices=False,
            include_mano=True,
        )
        image_stem = image_path.stem
        record["hands"] = [mano_hand_record(hand) for hand in record.get("hands", [])]
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
                write_json(args.out_folder / f"{image_stem}_{hand['hand_index']}_mano.json", hand_record)
                if args.save_npz:
                    save_npz_outputs(args.out_folder, image_stem, hand)
        else:
            write_json(args.out_folder / f"{image_stem}_mano.json", record)
            if args.save_npz:
                for hand in record["hands"]:
                    save_npz_outputs(args.out_folder, image_stem, hand)

        print(f"[mano] {image_path.name}: hands={record.get('num_hands', 0)}")

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
