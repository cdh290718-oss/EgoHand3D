from argparse import Namespace
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from egohand3d import pipeline
from egohand3d.storage import load_json


def test_single_inference_exports_and_resume_repairs_corrupt_artifact(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for index in range(2):
        cv2.imwrite(str(inputs / f"frame{index}.png"), np.zeros((16, 24, 3), np.uint8))
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in ["model.ckpt", "cfg.yaml", "detector.pt"]:
        (assets / name).write_text("test asset metadata only")
    mano = tmp_path / "mano_data"
    mano.mkdir()
    for name in ["MANO_RIGHT.pkl", "mano_mean_params.npz"]:
        (mano / name).write_text("test asset metadata only")
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    calls = []
    loads = []
    def load(*args):
        loads.append(args)
        return SimpleNamespace(model=SimpleNamespace(mano=SimpleNamespace(faces=[])), device="cpu")
    def infer(*args, **kwargs):
        calls.append(kwargs)
        return []
    monkeypatch.setitem(sys.modules, "egohand3d.inference", SimpleNamespace(load_runtime=load, infer_hands_from_image=infer))
    args = Namespace(input=str(inputs), out=str(tmp_path / "out"), recursive=False, hash_inputs=True,
                     hash_assets=True, checkpoint=str(assets / "model.ckpt"), cfg=str(assets / "cfg.yaml"),
                     detector=str(assets / "detector.pt"), stride=1, limit=0, sequence_fps=None, device="cpu",
                     detector_conf=0.3, rescale_factor=2., batch_size=2, no_mesh=True, no_overlay=True,
                     resume=False, retries=0, fail_fast=True)
    first = pipeline.process(args)
    assert first["frames_completed"] == 2
    assert len(calls) == 2
    assert len(loads) == 1
    out = tmp_path / "out"
    assert len(list((out / "2d").glob("*.json"))) == 2
    assert len(list((out / "3d").glob("*.json"))) == 2
    assert len(list((out / "mano").glob("*.json"))) == 2
    args.resume = True
    second = pipeline.process(args)
    assert second["latest_invocation"]["reused"] == 2
    assert len(loads) == 1  # A complete resume must not load large model assets.
    corrupt = next((out / "mano").glob("*.json"))
    corrupt.write_text("interrupted write")
    third = pipeline.process(args)
    assert third["latest_invocation"]["processed"] == 1
    assert len(calls) == 3
    assert load_json(corrupt)["hands"] == []
    args.detector_conf = 0.4
    with pytest.raises(ValueError, match="changed"):
        pipeline.process(args)
