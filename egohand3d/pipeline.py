"""One inference per frame; all exports share the same predictions."""
from __future__ import annotations

import csv
import io
import platform
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .jobs import JobStore, now, run_lock
from .mano_codec import ManoDecoder, encode_hand
from .media import iter_frames, scan_media
from .quality import inspect_hand
from .storage import atomic_text, file_hash, file_identity, load_json, save_json
from .visualization import draw_record_on_image

ROOT = Path(__file__).resolve().parents[1]


def code_identity() -> dict:
    paths = sorted((ROOT / "egohand3d").glob("*.py"))
    paths += sorted((ROOT / "wilor").rglob("*.py"))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def write_obj(path: Path, vertices, faces, mirrored: bool):
    faces = np.asarray(faces)
    if mirrored:
        faces = faces[:, [0, 2, 1]]
    lines = ["# EgoHand3D camera coordinates, meters"]
    lines += ["v %.9g %.9g %.9g" % tuple(v) for v in vertices]
    lines += ["f %d %d %d" % tuple(f + 1) for f in faces]
    atomic_text(path, "\n".join(lines) + "\n")


def export_frame(directory: Path, record: dict, image, faces, overlays: bool, meshes: bool) -> dict:
    sample_id = record["sample_id"]
    files = []
    path = directory / "records" / f"{sample_id}.json"
    save_json(path, record)
    files.append(path)
    for kind, keys in {
        "2d": ["joints_2d_xy", "joints_2d_scores"],
        "3d": ["joints_3d_root", "joints_3d_camera", "cam_t_full"],
        "mano": ["mano"],
    }.items():
        payload = {key: record[key] for key in ["sample_id", "source_path", "sequence_id", "frame_index", "timestamp_s"]}
        payload["hands"] = [{key: hand[key] for key in ["hand_index", "hand_side", "bbox_xyxy", "bbox_score", *keys]}
                            for hand in record["hands"]]
        output = directory / kind / f"{sample_id}.json"
        save_json(output, payload)
        files.append(output)
    if meshes:
        for hand in record["hands"]:
            output = directory / "meshes" / f"{sample_id}_{hand['hand_index']}.obj"
            write_obj(output, hand["vertices_3d_camera"], faces, hand["hand_side"] == "left")
            files.append(output)
    if overlays:
        output = directory / "overlays" / f"{sample_id}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        # Complete-file hashes prevent a partial image being reused after interruption.
        overlay = draw_record_on_image(image, record, score_threshold=0.01)
        if not cv2.imwrite(str(output), overlay):
            raise OSError(f"Cannot write {output}")
        files.append(output)
    return {str(path.relative_to(directory)): file_hash(path) for path in files}


def write_run_summary(directory: Path, store: JobStore, invocation: dict) -> dict:
    rows = store.records()
    successful = [r for r in rows if r["state"] == "completed"]
    hand_count = 0
    detected = 0
    flags = Counter()
    roundtrip_failed = 0
    records_index = []
    for row in successful:
        record = load_json(directory / row["record"])
        hand_count += len(record["hands"])
        detected += bool(record["hands"])
        for hand in record["hands"]:
            flags.update(hand["quality"]["flags"])
            roundtrip_failed += not hand["roundtrip"]["passed"]
        records_index.append({key: record[key] for key in ["sample_id", "source_path", "sequence_id", "timestamp_s", "frame_index"]}
                            | {"record": row["record"], "hands": len(record["hands"])})
    summary = {**store.summary(), "schema": "egohand3d.run/1", "frames_completed": len(successful),
               "frames_with_hands": detected, "hands": hand_count,
               "detection_coverage": detected / len(successful) if successful else None,
               "coverage_note": "Fraction of processed frames with predictions; not detection recall without GT",
               "quality_flags": dict(flags), "roundtrip_failures": roundtrip_failed,
               "mean_processing_seconds": float(np.mean([r["seconds"] for r in successful])) if successful else None,
               "latest_invocation": invocation,
               "failures": [{"sample_id": r["id"], "error": r["error"]} for r in rows if r["state"] == "failed"]}
    save_json(directory / "summary.json", summary)
    save_json(directory / "index.json", {"schema": "egohand3d.index/1", "records": records_index})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["sample_id", "source_path", "sequence_id", "timestamp_s", "frame_index", "record", "hands"])
    writer.writeheader()
    writer.writerows(records_index)
    atomic_text(directory / "index.csv", buffer.getvalue())
    return summary


def process(args) -> dict:
    directory = Path(args.out).resolve()
    manifest = scan_media(Path(args.input), args.recursive, args.hash_inputs)
    if not manifest["files"]:
        raise ValueError("No supported input media")
    if args.sequence_fps and any(f["kind"] != "image" for f in manifest["files"]):
        raise ValueError("--sequence-fps is for a directory of images only")
    assets = {key: file_identity(Path(getattr(args, key)), args.hash_assets)
              for key in ["checkpoint", "cfg", "detector"]}
    for name in ["MANO_RIGHT.pkl", "mano_mean_params.npz"]:
        assets[name] = file_identity(ROOT / "mano_data" / name, args.hash_assets)
    config = {"pipeline_version": 1, "manifest": manifest, "assets": assets, "code": code_identity(),
              "options": {key: getattr(args, key) for key in ["stride", "limit", "sequence_fps", "device",
              "detector_conf", "rescale_factor", "batch_size", "no_mesh", "no_overlay"]}}
    with run_lock(directory):
        store = JobStore(directory / "tasks.sqlite")
        try:
            store.initialize(config, args.resume)
            save_json(directory / "manifest.json", manifest)
            save_json(directory / "run_config.json", config)
            return _process(args, directory, store, manifest)
        finally:
            store.close()


def _process(args, directory: Path, store: JobStore, manifest: dict) -> dict:
    runtime = None
    started = time.perf_counter()
    invocation = {"started": now(), "processed": 0, "reused": 0, "python": sys.version, "platform": platform.platform()}
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for metadata, image in iter_frames(manifest, args.stride, args.limit, args.sequence_fps):
            if store.get("cancel_requested", False):
                store.set("state", "cancelled")
                break
            sample_id = metadata["sample_id"]
            if store.completed(sample_id, directory):
                invocation["reused"] += 1
                continue
            for attempt in range(args.retries + 1):
                store.begin(sample_id, metadata["source_path"])
                try:
                    if image is None:
                        raise ValueError(metadata.get("decode_error", "Unreadable image/video"))
                    if runtime is None:
                        from .inference import infer_hands_from_image, load_runtime
                        # External WiLoR resolves MANO relative to cwd; CLI establishes ROOT.
                        runtime = load_runtime(args.checkpoint, args.cfg, args.detector, args.device)
                        import torch
                        torch.set_float32_matmul_precision("highest")
                        torch.backends.cudnn.allow_tf32 = False
                        decoder = ManoDecoder(runtime.model.mano)
                        import torch
                        invocation["runtime"] = {"torch": torch.__version__, "device": str(runtime.device),
                                                 "cuda": torch.version.cuda, "float32_matmul_precision": torch.get_float32_matmul_precision()}
                    tick = time.perf_counter()
                    hands = infer_hands_from_image(image, runtime, detector_conf=args.detector_conf,
                        rescale_factor=args.rescale_factor, batch_size=args.batch_size,
                        include_vertices=True, include_mano=True)
                    for hand in hands:
                        hand["mano"] = encode_hand(hand)
                        hand["roundtrip"] = decoder.check(hand["mano"], hand)
                        hand["quality"] = inspect_hand(hand)
                    record = {"schema": "egohand3d.frame/1", **metadata, "width": int(image.shape[1]),
                              "height": int(image.shape[0]), "hands": hands,
                              "timing": {"inference_and_validation_s": time.perf_counter() - tick}}
                    artifacts = export_frame(directory, record, image, runtime.model.mano.faces,
                                             not args.no_overlay, not args.no_mesh)
                    elapsed = time.perf_counter() - tick
                    store.finish(sample_id, elapsed, f"records/{sample_id}.json", artifacts)
                    invocation["processed"] += 1
                    print(f"[process] {sample_id}: {len(hands)} hands, {elapsed:.3f}s", flush=True)
                    break
                except Exception as exc:
                    store.fail(sample_id, f"{type(exc).__name__}: {exc}")
                    print(f"[failed] {sample_id} attempt {attempt+1}: {exc}", flush=True)
                    if attempt == args.retries and args.fail_fast:
                        raise
        else:
            counts = store.summary()["counts"]
            store.set("state", "completed_with_errors" if counts.get("failed", 0) else "completed")
    except KeyboardInterrupt:
        store.set("state", "cancelled")
        print("[cancelled] Completed artifacts are retained; use --resume.", flush=True)
    except Exception:
        store.set("state", "failed")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        invocation["wall_seconds"] = time.perf_counter() - started
        store.set("updated", now())
        summary = write_run_summary(directory, store, invocation)
        from .reports import build_run_report
        build_run_report(directory)
    return summary
