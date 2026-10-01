import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest

from egohand3d.jobs import JobStore, inspect_jobs, run_lock
from egohand3d.media import iter_frames, scan_media
from egohand3d.reports import compare_benchmarks, package_results
from egohand3d.storage import file_hash, load_json, save_json


def test_atomic_write_rejects_nonfinite_and_preserves_previous_file(tmp_path):
    path = tmp_path / "record.json"
    save_json(path, {"ok": 1})
    with pytest.raises(ValueError):
        save_json(path, {"value": float("nan")})
    assert load_json(path) == {"ok": 1}


def test_resume_checks_artifact_hashes_and_changed_config(tmp_path):
    store = JobStore(tmp_path / "tasks.sqlite")
    try:
        store.initialize({"inputs": [1]}, False)
        save_json(tmp_path / "one.json", {"hands": []})
        store.begin("one", "source.png")
        store.finish("one", 0.5, "one.json", {"one.json": file_hash(tmp_path / "one.json")})
        assert store.completed("one", tmp_path)
        store.initialize({"inputs": [1]}, True)
        with pytest.raises(ValueError):
            store.initialize({"inputs": [2]}, True)
        save_json(tmp_path / "one.json", {"corrupted": True})
        assert not store.completed("one", tmp_path)
        inspect_jobs(tmp_path, cancel=True)
        assert store.get("cancel_requested") is True
    finally:
        store.close()


def test_run_lock_prevents_concurrent_writers(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with run_lock(tmp_path):
                pass


def test_scan_flags_bad_inputs_preserves_duplicate_names_and_detects_mutation(tmp_path):
    for sub in ["a", "b"]:
        folder = tmp_path / sub
        folder.mkdir()
        cv2.imwrite(str(folder / "same.png"), np.zeros((16, 24, 3), np.uint8))
    (tmp_path / "bad.jpg").write_text("broken")
    manifest = scan_media(tmp_path, recursive=True, hash_inputs=True)
    assert manifest["summary"]["readable"] == 2
    assert manifest["summary"]["unreadable"] == 1
    assert len({r["id"] for r in manifest["files"]}) == 3
    frames = list(iter_frames(manifest, sequence_fps=25))
    assert sum(image is None for _, image in frames) == 1
    (tmp_path / "a/same.png").write_bytes(b"changed")
    with pytest.raises(ValueError):
        list(iter_frames(manifest))


def test_video_stride_and_timestamps(tmp_path):
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (24, 16))
    assert writer.isOpened()
    for value in range(6):
        writer.write(np.full((16, 24, 3), value * 20, np.uint8))
    writer.release()
    frames = list(iter_frames(scan_media(path), stride=2, limit=2))
    assert [m["frame_index"] for m, _ in frames] == [0, 2]
    assert [m["timestamp_s"] for m, _ in frames] == pytest.approx([0., 0.2])


def test_archives_exclude_weights_and_checksums_cover_files(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    save_json(run / "summary.json", {"result": 1})
    (run / "weights.ckpt").write_bytes(b"not for export")
    out = tmp_path / "result.zip"
    package_results(run, out)
    with zipfile.ZipFile(out) as archive:
        assert "weights.ckpt" not in archive.namelist()
        manifest = json.loads(archive.read("ARTIFACT_MANIFEST.json"))
        assert manifest["summary.json"]["sha256"] == file_hash(run / "summary.json")


def test_comparison_rejects_different_ground_truth(tmp_path):
    payload = {"schema": "egohand3d.benchmark/1", "ground_truth_sha256": "a",
               "protocol": {}, "evaluator_sha256": "code", "pck_px": {}}
    save_json(tmp_path / "a.json", payload)
    save_json(tmp_path / "b.json", {**payload, "ground_truth_sha256": "b"})
    with pytest.raises(ValueError, match="ground_truth"):
        compare_benchmarks([tmp_path / "a.json", tmp_path / "b.json"], tmp_path / "out")
