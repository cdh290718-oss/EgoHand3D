"""Render 50 saved fine-tuned predictions and compose text-free result figures."""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "outputs/validation_20261001/hoi4d_finetuned"
OUT = HERE / "mesh_overlay_50"
FACES_SOURCE = ROOT / "outputs/workflow_smoke_02/meshes/873519995f6debef9aa8_0.obj"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detail_crop(points, boxes, width, height):
    lo, hi = points.min(axis=0), points.max(axis=0)
    for box in boxes:
        lo, hi = np.minimum(lo, box[:2]), np.maximum(hi, box[2:])
    padding = max(40, float(max(hi - lo)) * 0.15)
    lo = np.maximum([0, 0], np.floor(lo - padding))
    hi = np.minimum([width, height], np.ceil(hi + padding))
    # Expand the context to 4:3 where possible; do not cut off any saved mesh.
    crop_w, crop_h = hi - lo
    crop_w, crop_h = max(crop_w, crop_h * 4 / 3), max(crop_h, crop_w * 3 / 4)
    crop_w, crop_h = min(width, int(np.ceil(crop_w))), min(height, int(np.ceil(crop_h)))
    center = (lo + hi) / 2
    x0 = max(0, min(width - crop_w, int(np.floor(center[0] - crop_w / 2))))
    y0 = max(0, min(height - crop_h, int(np.floor(center[1] - crop_h / 2))))
    crop = [x0, y0, x0 + crop_w, y0 + crop_h]
    assert crop[0] <= lo[0] and crop[1] <= lo[1] and crop[2] >= hi[0] and crop[3] >= hi[1]
    return crop


def main():
    OUT.mkdir(exist_ok=True)
    image_dir = OUT / "overlays"
    image_dir.mkdir(exist_ok=True)
    entries = json.loads((RUN / "index.json").read_text())["records"]
    valid = []
    for i, entry in enumerate(entries):
        record = json.loads((RUN / entry["record"]).read_text())
        if record["hands"] and all(len(h.get("vertices_3d_camera", [])) == 778 for h in record["hands"]):
            assert Path(record["source_path"]).exists()
            valid.append(i)
    retained = {0, 21, 41, 60, 81}
    assert retained <= set(valid)
    rest = [i for i in valid if i not in retained]
    selected = sorted(retained | {rest[int(i)] for i in np.linspace(0, len(rest) - 1, 45).round()})
    assert len(selected) == 50
    faces = np.array([
        [int(v.split("/")[0]) - 1 for v in line.split()[1:]]
        for line in FACES_SOURCE.read_text().splitlines() if line.startswith("f ")
    ], dtype=np.int32)
    assert faces.shape == (1538, 3) and faces.max() == 777
    renderer = pyrender.OffscreenRenderer(1920, 1080)
    results = []
    try:
        for order, index in enumerate(selected):
            record_path = RUN / entries[index]["record"]
            record = json.loads(record_path.read_text())
            source_path = Path(record["source_path"])
            source = cv2.imread(str(source_path))
            assert source is not None and source.shape[:2] == (1080, 1920)
            height, width = source.shape[:2]
            intrinsics = np.array(record["hands"][0]["mano"]["intrinsics"])
            scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.28] * 3)
            projected, errors = [], []
            for hand in record["hands"]:
                assert np.allclose(intrinsics, hand["mano"]["intrinsics"])
                vertices = np.array(hand["vertices_3d_camera"], dtype=np.float64)
                assert vertices.shape == (778, 3) and np.isfinite(vertices).all() and (vertices[:, 2] > 0).all()
                uvz = vertices @ intrinsics.T
                projected.append(uvz[:, :2] / uvz[:, 2:])
                j = np.array(hand["joints_3d_camera"]) @ intrinsics.T
                error = float(np.max(np.abs(j[:, :2] / j[:, 2:] - hand["joints_2d_xy"])))
                assert error < 0.1
                errors.append(error)
                topology = faces if hand["is_right"] else faces[:, [0, 2, 1]]
                mesh = trimesh.Trimesh(vertices=vertices * [1, -1, -1], faces=topology, process=False)
                color = (0.14, 0.52, 0.82, 1.0) if hand["is_right"] else (0.95, 0.44, 0.16, 1.0)
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0, roughnessFactor=0.8, baseColorFactor=color, doubleSided=True
                )
                scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
            scene.add(pyrender.IntrinsicsCamera(
                fx=intrinsics[0, 0], fy=intrinsics[1, 1], cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                znear=0.01, zfar=1000.0
            ), pose=np.eye(4))
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=np.eye(4))
            rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            mask = depth > 0
            assert int(mask.sum()) > 100
            overlay = source.copy()
            overlay[mask] = rgba[:, :, :3][:, :, ::-1][mask]
            path = image_dir / f"sample_{index:04d}_mesh_overlay.jpg"
            assert cv2.imwrite(str(path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
            crop = detail_crop(np.concatenate(projected), [h["bbox_xyxy"] for h in record["hands"]], width, height)
            results.append({
                "order": order, "index": index, "sample_id": record["sample_id"],
                "source_image": str(source_path), "source_image_sha256": digest(source_path),
                "record": str(record_path), "record_sha256": digest(record_path),
                "file": str(path.relative_to(OUT)), "sha256": digest(path),
                "mesh_pixels": int(mask.sum()), "hand_sides": [h["hand_side"] for h in record["hands"]],
                "projection_max_abs_errors_px": errors, "detail_crop_xyxy": crop,
            })
            print(f"Rendered {order + 1}/50: sample {index:04d}", flush=True)
    finally:
        renderer.delete()
    montage_outputs = []
    for detail in [True, False]:
        tile_w, tile_h = (640, 480) if detail else (640, 360)
        canvas = Image.new("RGB", (tile_w * 5, tile_h * 10), (255, 255, 255))
        for i, result in enumerate(results):
            with Image.open(OUT / result["file"]) as source:
                tile = source.convert("RGB")
                if detail:
                    tile = tile.crop(tuple(result["detail_crop_xyxy"]))
                tile = ImageOps.contain(tile, (tile_w, tile_h), Image.Resampling.LANCZOS)
                x = (i % 5) * tile_w + (tile_w - tile.width) // 2
                y = (i // 5) * tile_h + (tile_h - tile.height) // 2
                canvas.paste(tile, (x, y))
        stem = "mesh_overlay_50_no_text" if detail else "mesh_overlay_50_full_frame_no_text"
        for ext in ["jpg", "png"]:
            path = OUT / f"{stem}.{ext}"
            options = {"quality": 95, "subsampling": 0} if ext == "jpg" else {}
            canvas.save(path, dpi=(300, 300), **options)
            montage_outputs.append({"file": path.name, "size_px": list(canvas.size), "sha256": digest(path)})
            print(f"Saved {path.name}: {canvas.size}", flush=True)
    manifest = {
        "count": 50, "model": "EgoHand3D first-person fine-tuned weights",
        "training_performed": False, "inference_performed": False,
        "source_run": str(RUN), "run_config_sha256": digest(RUN / "run_config.json"),
        "selection": "Retain five earlier fine-tuned samples; add 45 uniformly spaced records from remaining valid saved predictions. No GT/error selection.",
        "layout": {"rows": 10, "columns": 5, "text": False, "titles": False, "borders": False},
        "rendering": "Unmodified saved camera-space vertices, original topology, opaque mesh, no scene-occlusion model",
        "faces_source": str(FACES_SOURCE), "faces_source_sha256": digest(FACES_SOURCE),
        "script_sha256": digest(__file__), "results": results, "montages": montage_outputs,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "index.html").write_text(
        '<!doctype html><html><meta charset="utf-8"><title>3D Mesh Overlay</title>'
        '<style>html,body{margin:0;background:white}img{display:block;width:100%;height:auto}</style>'
        '<a href="mesh_overlay_50_no_text.png"><img src="mesh_overlay_50_no_text.jpg"></a></html>'
    )


if __name__ == "__main__":
    main()
