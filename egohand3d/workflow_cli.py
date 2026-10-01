"""Register the headless workflow in the existing command line."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def probability(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be in [0,1]")
    return value


def _execute(args):
    try:
        result = execute(args)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"[error] {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result.get("state") == "cancelled":
        return 130
    if result.get("state") in {"completed_with_errors", "failed"} or result.get("passed") is False or result.get("roundtrip_failures", 0):
        return 1
    if args.command == "scan" and result["summary"]["unreadable"]:
        return 1
    return 0


def execute(args):
    from .storage import file_hash, load_json, save_json
    if args.command == "scan":
        from .media import scan_media
        result = scan_media(args.input, args.recursive, args.hash_inputs)
        save_json(args.out, result)
        return result
    if args.command == "process":
        from .pipeline import process
        for key in ["input", "out", "checkpoint", "cfg", "detector"]:
            setattr(args, key, str(Path(getattr(args, key)).resolve()))
        if args.recursive and Path(args.out).is_relative_to(Path(args.input)):
            raise ValueError("Output must be outside a recursively scanned input directory")
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            return process(args)
        finally:
            os.chdir(previous)
    if args.command == "jobs":
        from .jobs import inspect_jobs
        return inspect_jobs(args.run, args.cancel)
    if args.command == "mano-check":
        from .mano_codec import ManoDecoder, validate_packet
        payload = load_json(args.input)
        if "hands" in payload:
            decoder = ManoDecoder.from_directory(args.mano_dir, args.device)
            checks = [decoder.check(hand["mano"], hand) for hand in payload["hands"]]
            result = {"hands": len(checks), "passed": bool(checks) and all(c["passed"] for c in checks), "checks": checks}
        else:
            validate_packet(payload)
            result = {"passed": True, "validation": "Schema and rotation equivalence; no reference geometry supplied"}
        save_json(args.out, result)
        return result
    if args.command == "sequence":
        from .sequence import process_sequence
        return process_sequence(args.run, args.out, args.mano_dir, args.max_gap, args.smoothing_seconds, args.fill_gaps)
    if args.command == "prepare-gt":
        from .dataset_adapters import prepare_hoi4d
        return prepare_hoi4d(args.run, args.sample_dir, args.out)
    if args.command == "render-sequence":
        from .video_export import export_sequence_videos
        return {"videos": export_sequence_videos(args.run)}
    if args.command == "benchmark":
        from .metrics import evaluate_frames
        from .reports import build_benchmark_report
        truth = load_json(args.ground_truth)
        if truth.get("schema") != "egohand3d.ground_truth/1" or truth.get("joint_order") != "OpenPose21":
            raise ValueError("GT needs schema egohand3d.ground_truth/1 and joint_order OpenPose21")
        if truth.get("length_unit") != "m" or truth.get("camera_axes") != "x_right_y_down_z_forward":
            raise ValueError("GT must declare meters and camera axes x_right_y_down_z_forward")
        entries = load_json(args.run / "index.json")["records"]
        predictions, identities = {}, {}
        wanted = {str(f["sample_id"]) for f in truth["frames"]}
        for entry in entries:
            key = str(entry["sample_id"])
            if key in predictions:
                raise ValueError("Duplicate prediction sample IDs")
            if key in wanted:
                predictions[key] = load_json(args.run / entry["record"])
                identities[entry["record"]] = file_hash(args.run / entry["record"])
        result = evaluate_frames(truth["frames"], predictions, args.variant, args.match_iou, args.pck_thresholds)
        result["protocol"]["labels_complete"] = truth.get("labels_complete", True)
        result["protocol"]["label_scope"] = truth.get("label_scope", "All relevant hands annotated")
        result["matched_prediction_fraction"] = result["hand_precision"]
        if not truth.get("labels_complete", True):
            result["hand_precision"] = None
        result.update(ground_truth_sha256=file_hash(args.ground_truth),
                      evaluator_sha256=file_hash(Path(__file__).with_name("metrics.py")),
                      prediction_sha256=identities, source_run=str(args.run.resolve()))
        if args.out.exists() and any(args.out.iterdir()):
            raise ValueError("Benchmark output directory must be empty")
        args.out.mkdir(parents=True, exist_ok=True)
        build_benchmark_report(args.out, result)
        return {key: value for key, value in result.items() if key not in {"details", "prediction_sha256"}}
    if args.command == "compare":
        from .reports import compare_benchmarks
        return compare_benchmarks(args.summaries, args.out)
    if args.command == "package":
        from .reports import package_results
        return package_results(args.run, args.out)
    raise ValueError(f"Unsupported command: {args.command}")


def add_workflow_commands(sub):
    p = sub.add_parser("scan", help="Preflight image/video inputs without loading models")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--hash-inputs", action="store_true")
    p.set_defaults(func=_execute)

    p = sub.add_parser("process", help="One inference pass: 2D/3D/MANO/mesh, quality checks and resume")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default=str(ROOT / "pretrained_models/wilor_final.ckpt"))
    p.add_argument("--cfg", default=str(ROOT / "pretrained_models/model_config.yaml"))
    p.add_argument("--detector", default=str(ROOT / "pretrained_models/detector.pt"))
    p.add_argument("--device", default="auto")
    p.add_argument("--detector-conf", type=probability, default=0.3)
    p.add_argument("--rescale-factor", type=positive_float, default=2.)
    p.add_argument("--batch-size", type=positive_int, default=16)
    p.add_argument("--stride", type=positive_int, default=1)
    p.add_argument("--limit", type=nonnegative_int, default=0)
    p.add_argument("--sequence-fps", type=positive_float)
    p.add_argument("--retries", type=nonnegative_int, default=1)
    for flag in ["recursive", "hash-inputs", "hash-assets", "resume", "fail-fast", "no-mesh", "no-overlay"]:
        p.add_argument("--" + flag, action="store_true")
    p.set_defaults(func=_execute)

    p = sub.add_parser("jobs", help="Inspect a run or request cancellation between frames")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--cancel", action="store_true")
    p.set_defaults(func=_execute)

    p = sub.add_parser("mano-check", help="Validate MANO interchange or reconstruct a complete frame")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mano-dir", type=Path, default=ROOT / "mano_data")
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=_execute)

    p = sub.add_parser("sequence", help="Associate hands, smooth poses, optionally fill bounded gaps")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mano-dir", type=Path, default=ROOT / "mano_data")
    p.add_argument("--max-gap", type=nonnegative_int, default=3)
    p.add_argument("--smoothing-seconds", type=positive_float, default=0.08)
    p.add_argument("--fill-gaps", action="store_true")
    p.set_defaults(func=_execute)

    p = sub.add_parser("prepare-gt", help="Adapt trusted local HOI4D pickle labels to fixed 2D GT")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--sample-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_execute)

    p = sub.add_parser("render-sequence", help="Export offline raw/processed comparison videos")
    p.add_argument("--run", type=Path, required=True)
    p.set_defaults(func=_execute)

    p = sub.add_parser("benchmark", help="Evaluate fixed GT with one-to-one IoU matching")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variant", choices=["raw", "processed"], default="raw")
    p.add_argument("--match-iou", type=probability, default=0.1)
    p.add_argument("--pck-thresholds", nargs="+", type=positive_float, default=[5., 10., 20., 50.])
    p.set_defaults(func=_execute)

    p = sub.add_parser("compare", help="Compare benchmarks with identical evaluation protocols")
    p.add_argument("--summaries", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_execute)

    p = sub.add_parser("package", help="Archive generated results with checksums, excluding weights")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_execute)
