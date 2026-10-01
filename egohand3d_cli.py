from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_python(script: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, str(ROOT / script), *extra_args]
    print("[cmd]", " ".join(cmd))
    return subprocess.call(cmd)


def add_common_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--img_folder", default="examples/images")
    parser.add_argument("--out_folder", required=True)
    parser.add_argument("--checkpoint", default="pretrained_models/wilor_final.ckpt")
    parser.add_argument("--cfg", default="pretrained_models/model_config.yaml")
    parser.add_argument("--detector", default="pretrained_models/detector.pt")
    parser.add_argument("--detector_conf", default=None)
    parser.add_argument("--rescale_factor", default=None)
    parser.add_argument("--batch_size", default=None)
    parser.add_argument("--device", default=None)


def append_if_present(args: list[str], flag: str, value) -> None:
    if value is not None:
        args.extend([flag, str(value)])


def command_reconstruct(ns: argparse.Namespace) -> int:
    args = [
        "--img_folder", ns.img_folder,
        "--out_folder", ns.out_folder,
        "--checkpoint", ns.checkpoint,
        "--cfg", ns.cfg,
        "--detector", ns.detector,
    ]
    append_if_present(args, "--rescale_factor", ns.rescale_factor)
    return run_python("detect_and_reconstruct.py", args)


def command_export_2d(ns: argparse.Namespace) -> int:
    args = [
        "--img_folder", ns.img_folder,
        "--out_folder", ns.out_folder,
        "--checkpoint", ns.checkpoint,
        "--cfg", ns.cfg,
        "--detector", ns.detector,
    ]
    append_if_present(args, "--rescale_factor", ns.rescale_factor)
    if ns.save_per_hand:
        args.append("--save_per_hand")
    return run_python("export_2d_joints.py", args)


def command_export_3d(ns: argparse.Namespace) -> int:
    args = [
        "--img_folder", ns.img_folder,
        "--out_folder", ns.out_folder,
        "--checkpoint", ns.checkpoint,
        "--cfg", ns.cfg,
        "--detector", ns.detector,
    ]
    append_if_present(args, "--detector_conf", ns.detector_conf)
    append_if_present(args, "--rescale_factor", ns.rescale_factor)
    append_if_present(args, "--batch_size", ns.batch_size)
    append_if_present(args, "--device", ns.device)
    if ns.save_per_hand:
        args.append("--save_per_hand")
    if ns.save_npy:
        args.append("--save_npy")
    if ns.include_vertices:
        args.append("--include_vertices")
    return run_python("export_3d_joints.py", args)


def command_export_mano(ns: argparse.Namespace) -> int:
    args = [
        "--img_folder", ns.img_folder,
        "--out_folder", ns.out_folder,
        "--checkpoint", ns.checkpoint,
        "--cfg", ns.cfg,
        "--detector", ns.detector,
    ]
    append_if_present(args, "--detector_conf", ns.detector_conf)
    append_if_present(args, "--rescale_factor", ns.rescale_factor)
    append_if_present(args, "--batch_size", ns.batch_size)
    append_if_present(args, "--device", ns.device)
    if ns.save_per_hand:
        args.append("--save_per_hand")
    if ns.save_npz:
        args.append("--save_npz")
    return run_python("export_mano_params.py", args)


def command_video(ns: argparse.Namespace) -> int:
    args = [
        "--video", ns.video,
        "--out_folder", ns.out_folder,
        "--checkpoint_path", ns.checkpoint,
        "--cfg_path", ns.cfg,
        "--detector_path", ns.detector,
    ]
    append_if_present(args, "--output_video", ns.output_video)
    append_if_present(args, "--detector_conf", ns.detector_conf)
    append_if_present(args, "--rescale_factor", ns.rescale_factor)
    append_if_present(args, "--batch_size", ns.batch_size)
    append_if_present(args, "--stride", ns.stride)
    append_if_present(args, "--max_frames", ns.max_frames)
    return run_python("tools/detect_video_hands.py", args)


def command_evaluate(ns: argparse.Namespace) -> int:
    args = ["--sample_dir", ns.sample_dir, "--pred_dir", ns.pred_dir, "--out_dir", ns.out_dir]
    append_if_present(args, "--checkpoint", ns.checkpoint)
    if ns.no_overlays:
        args.append("--no_overlays")
    return run_python("tools/evaluate_hoi4d_wilor_2djoints.py", args)


def command_report(ns: argparse.Namespace) -> int:
    args = ["--result_dir", ns.result_dir]
    append_if_present(args, "--image_dir", ns.image_dir)
    append_if_present(args, "--out_dir", ns.out_dir)
    if ns.recursive:
        args.append("--recursive")
    return run_python("tools/build_report.py", args)


def command_manifest(ns: argparse.Namespace) -> int:
    args = []
    append_if_present(args, "--image_dir", ns.image_dir)
    append_if_present(args, "--video_dir", ns.video_dir)
    append_if_present(args, "--pred_dir", ns.pred_dir)
    append_if_present(args, "--label_dir", ns.label_dir)
    append_if_present(args, "--out", ns.out)
    if ns.recursive:
        args.append("--recursive")
    if ns.require_predictions:
        args.append("--require_predictions")
    if ns.require_labels:
        args.append("--require_labels")
    return run_python("tools/build_manifest.py", args)


def command_merge(ns: argparse.Namespace) -> int:
    args = []
    for input_dir in ns.input_dir or []:
        args.extend(["--input_dir", input_dir])
    append_if_present(args, "--joints2d_dir", ns.joints2d_dir)
    append_if_present(args, "--joints3d_dir", ns.joints3d_dir)
    append_if_present(args, "--mano_dir", ns.mano_dir)
    append_if_present(args, "--mesh_dir", ns.mesh_dir)
    append_if_present(args, "--overlay_dir", ns.overlay_dir)
    append_if_present(args, "--out", ns.out)
    if ns.recursive:
        args.append("--recursive")
    if ns.load_json:
        args.append("--load_json")
    if ns.include_unknown_json:
        args.append("--include_unknown_json")
    return run_python("tools/merge_exports.py", args)


def command_visualize(ns: argparse.Namespace) -> int:
    args = ["--img_folder", ns.img_folder, "--json_dir", ns.json_dir, "--out_folder", ns.out_folder]
    append_if_present(args, "--json_suffix", ns.json_suffix)
    append_if_present(args, "--score_threshold", ns.score_threshold)
    if ns.draw_indices:
        args.append("--draw_indices")
    if ns.no_boxes:
        args.append("--no_boxes")
    if ns.contact_sheet:
        args.append("--contact_sheet")
    return run_python("tools/visualize_joints.py", args)


def command_validate(ns: argparse.Namespace) -> int:
    args = [
        "--checkpoint", ns.checkpoint,
        "--cfg", ns.cfg,
        "--detector", ns.detector,
        "--mano_right", ns.mano_right,
        "--mano_left", ns.mano_left,
        "--out", ns.out,
    ]
    append_if_present(args, "--image_dir", ns.image_dir)
    if ns.load_models:
        args.append("--load_models")
    return run_python("tools/validate_setup.py", args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified EgoHand3D command line interface.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reconstruct", help="Export OBJ meshes and overlay renderings")
    add_common_infer_args(p)
    p.set_defaults(func=command_reconstruct)

    p = sub.add_parser("export-2d", help="Export projected 2D hand joints")
    add_common_infer_args(p)
    p.add_argument("--save_per_hand", action="store_true")
    p.set_defaults(func=command_export_2d)

    p = sub.add_parser("export-3d", help="Export 3D hand joints and camera metadata")
    add_common_infer_args(p)
    p.add_argument("--save_per_hand", action="store_true")
    p.add_argument("--save_npy", action="store_true")
    p.add_argument("--include_vertices", action="store_true")
    p.set_defaults(func=command_export_3d)

    p = sub.add_parser("export-mano", help="Export MANO pose/shape and camera parameters")
    add_common_infer_args(p)
    p.add_argument("--save_per_hand", action="store_true")
    p.add_argument("--save_npz", action="store_true")
    p.set_defaults(func=command_export_mano)

    p = sub.add_parser("video", help="Detect and visualize hands in a video")
    p.add_argument("--video", required=True)
    p.add_argument("--out_folder", required=True)
    p.add_argument("--output_video", default=None)
    p.add_argument("--checkpoint", default="pretrained_models/wilor_final.ckpt")
    p.add_argument("--cfg", default="pretrained_models/model_config.yaml")
    p.add_argument("--detector", default="pretrained_models/detector.pt")
    p.add_argument("--detector_conf", default=None)
    p.add_argument("--rescale_factor", default=None)
    p.add_argument("--batch_size", default=None)
    p.add_argument("--stride", default=None)
    p.add_argument("--max_frames", default=None)
    p.set_defaults(func=command_video)

    p = sub.add_parser("evaluate", help="Evaluate 2D joints against HOI4D-style labels")
    p.add_argument("--sample_dir", required=True)
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--no_overlays", action="store_true")
    p.set_defaults(func=command_evaluate)

    p = sub.add_parser("report", help="Build a report for an output directory")
    p.add_argument("--result_dir", required=True)
    p.add_argument("--image_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--recursive", action="store_true")
    p.set_defaults(func=command_report)

    p = sub.add_parser("manifest", help="Build an image/video processing manifest")
    p.add_argument("--image_dir", default=None)
    p.add_argument("--video_dir", default=None)
    p.add_argument("--pred_dir", default=None)
    p.add_argument("--label_dir", default=None)
    p.add_argument("--out", default="outputs/manifest.json")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--require_predictions", action="store_true")
    p.add_argument("--require_labels", action="store_true")
    p.set_defaults(func=command_manifest)

    p = sub.add_parser("merge", help="Merge exported 2D/3D/MANO/mesh files into one index")
    p.add_argument("--input_dir", action="append", default=[])
    p.add_argument("--joints2d_dir", default=None)
    p.add_argument("--joints3d_dir", default=None)
    p.add_argument("--mano_dir", default=None)
    p.add_argument("--mesh_dir", default=None)
    p.add_argument("--overlay_dir", default=None)
    p.add_argument("--out", default="outputs/merged_exports.json")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--load_json", action="store_true")
    p.add_argument("--include_unknown_json", action="store_true")
    p.set_defaults(func=command_merge)

    p = sub.add_parser("visualize", help="Draw JSON joints on source images")
    p.add_argument("--img_folder", required=True)
    p.add_argument("--json_dir", required=True)
    p.add_argument("--out_folder", required=True)
    p.add_argument("--json_suffix", default=None)
    p.add_argument("--score_threshold", default=None)
    p.add_argument("--draw_indices", action="store_true")
    p.add_argument("--no_boxes", action="store_true")
    p.add_argument("--contact_sheet", action="store_true")
    p.set_defaults(func=command_visualize)

    p = sub.add_parser("validate", help="Validate dependencies and required runtime files")
    p.add_argument("--checkpoint", default="pretrained_models/wilor_final.ckpt")
    p.add_argument("--cfg", default="pretrained_models/model_config.yaml")
    p.add_argument("--detector", default="pretrained_models/detector.pt")
    p.add_argument("--mano_right", default="mano_data/MANO_RIGHT.pkl")
    p.add_argument("--mano_left", default="mano_data/MANO_LEFT.pkl")
    p.add_argument("--image_dir", default=None)
    p.add_argument("--out", default="outputs/validate_setup.json")
    p.add_argument("--load_models", action="store_true")
    p.set_defaults(func=command_validate)

    from egohand3d.workflow_cli import add_workflow_commands
    add_workflow_commands(sub)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
