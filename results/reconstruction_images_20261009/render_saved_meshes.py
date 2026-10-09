"""Render existing EgoHand3D camera-space predictions; no model inference."""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import hashlib
import html
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyrender
import trimesh

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FACES_SOURCE = ROOT / "outputs/workflow_smoke_02/meshes/873519995f6debef9aa8_0.obj"
FACES = np.array([
    [int(v.split("/")[0]) - 1 for v in line.split()[1:]]
    for line in FACES_SOURCE.read_text().splitlines() if line.startswith("f ")
], dtype=np.int32)
assert FACES.shape == (1538, 3) and FACES.min() == 0 and FACES.max() == 777


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_image(path, image):
    assert cv2.imwrite(str(path), image), str(path)


def fit_panel(image, width=600, height=560):
    scale = min(width / image.shape[1], height / image.shape[0])
    image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    panel = np.full((height, width, 3), 245, np.uint8)
    y, x = (height - image.shape[0]) // 2, (width - image.shape[1]) // 2
    panel[y:y + image.shape[0], x:x + image.shape[1]] = image
    return panel


def render_record(run_dir, entry, prefix, model_label):
    record_path = run_dir / entry["record"]
    record = json.loads(record_path.read_text())
    image_path = Path(record["source_path"])
    source = cv2.imread(str(image_path))
    assert source is not None, image_path
    height, width = source.shape[:2]
    assert (record["width"], record["height"]) == (width, height)
    assert record["hands"]
    intrinsics = np.array(record["hands"][0]["mano"]["intrinsics"])
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.28] * 3)
    projected = []
    projection_errors = []
    for hand in record["hands"]:
        assert np.allclose(intrinsics, hand["mano"]["intrinsics"])
        vertices = np.array(hand["vertices_3d_camera"], dtype=np.float64)
        assert vertices.shape == (778, 3) and np.isfinite(vertices).all()
        assert (vertices[:, 2] > 0).all()
        uvz = vertices @ intrinsics.T
        projected.append(uvz[:, :2] / uvz[:, 2:])
        joints = np.array(hand["joints_3d_camera"], dtype=np.float64)
        joint_uvz = joints @ intrinsics.T
        error = float(np.max(np.abs(joint_uvz[:, :2] / joint_uvz[:, 2:] - np.array(hand["joints_2d_xy"]))))
        assert error < 0.1, error
        projection_errors.append(error)
        # Stored vertices already include camera translation and left-hand mirroring.
        # Convert x-right/y-down/z-forward to OpenGL x-right/y-up/z-backward.
        vertices_gl = vertices * [1, -1, -1]
        faces = FACES if hand["is_right"] else FACES[:, [0, 2, 1]]
        mesh = trimesh.Trimesh(vertices=vertices_gl, faces=faces, process=False)
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
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    mask = depth > 0
    assert int(mask.sum()) > 100, "No visible mesh pixels"
    color_bgr = rgba[:, :, :3][:, :, ::-1]
    overlay = source.copy()
    overlay[mask] = color_bgr[mask]
    mesh_only = np.full_like(source, 255)
    mesh_only[mask] = color_bgr[mask]
    files = {
        "input": f"{prefix}_input.jpg",
        "overlay": f"{prefix}_mesh_overlay.jpg",
        "mesh_only": f"{prefix}_mesh_only.png",
        "comparison": f"{prefix}_comparison.jpg",
    }
    shutil.copy2(image_path, OUT / files["input"])
    save_image(OUT / files["overlay"], overlay)
    save_image(OUT / files["mesh_only"], mesh_only)
    uv = np.concatenate(projected)
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    for hand in record["hands"]:
        bbox = np.array(hand["bbox_xyxy"])
        lo, hi = np.minimum(lo, bbox[:2]), np.maximum(hi, bbox[2:])
    padding = max(40, float(max(hi - lo)) * 0.15)
    x0, y0 = np.maximum(0, np.floor(lo - padding)).astype(int)
    x1, y1 = np.minimum([width, height], np.ceil(hi + padding)).astype(int)
    assert x1 > x0 and y1 > y0
    pair = np.hstack([fit_panel(source[y0:y1, x0:x1]), fit_panel(overlay[y0:y1, x0:x1])])
    title = np.full((82, 1200, 3), 255, np.uint8)
    cv2.putText(title, f"{model_label} | {prefix}", (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(title, "Original image (crop)", (18, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(title, "3D mesh overlay (same crop)", (618, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 1, cv2.LINE_AA)
    save_image(OUT / files["comparison"], np.vstack([title, pair]))
    result = {
        "sample_id": record["sample_id"], "model_label": model_label,
        "source_image": str(image_path), "source_image_sha256": sha256(image_path),
        "prediction_record": str(record_path), "prediction_record_sha256": sha256(record_path),
        "run_config": str(run_dir / "run_config.json"), "run_config_sha256": sha256(run_dir / "run_config.json"),
        "hand_sides": [h["hand_side"] for h in record["hands"]],
        "mesh_pixels": int(mask.sum()), "projection_max_abs_errors_px": projection_errors,
        "comparison_crop_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "files": files, "output_sha256": {k: sha256(OUT / v) for k, v in files.items()},
    }
    print(prefix, result["hand_sides"], "mesh pixels", result["mesh_pixels"], flush=True)
    return result


def main():
    run_dir = ROOT / "outputs/validation_20261001/hoi4d_finetuned"
    entries = json.loads((run_dir / "index.json").read_text())["records"]
    results = []
    # Fixed index intervals; if empty, take the next record with a saved detection.
    # Selection does not use prediction accuracy or ground truth.
    selected = set()
    for start in [0, 20, 40, 60, 80]:
        index = next(i for i in range(start, len(entries)) if entries[i]["hands"] and i not in selected)
        selected.add(index)
        results.append(render_record(run_dir, entries[index], f"finetuned_sample_{index:04d}", "EgoHand3D fine-tuned"))
    smoke = ROOT / "outputs/workflow_smoke_02"
    entries = json.loads((smoke / "index.json").read_text())["records"]
    entry = next(e for e in entries if e["sample_id"] == "873519995f6debef9aa8")
    results.append(render_record(smoke, entry, "initial_two_hands", "WiLoR initial weights"))
    manifest = {
        "model_inference_performed": False,
        "geometry": "Unmodified saved camera-space vertices, 778 vertices / 1538 original faces per hand",
        "rendering": "Opaque shaded mesh; right=blue, left=orange; no scene/object occlusion model",
        "selection": "First detected record at or after indices 0,20,40,60,80, plus existing two-hand smoke sample",
        "faces_source": str(FACES_SOURCE), "faces_source_sha256": sha256(FACES_SOURCE),
        "script_sha256": sha256(__file__), "results": results,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    cards = []
    for r in results:
        f = r["files"]
        cards.append(f'<section><h2>{html.escape(r["model_label"])} — {html.escape(f["comparison"].replace("_comparison.jpg", ""))}</h2>'
                     f'<a href="{f["comparison"]}"><img src="{f["comparison"]}"></a>'
                     f'<p><a href="{f["overlay"]}">完整网格叠加图</a> · <a href="{f["mesh_only"]}">白底三维手部</a> · <a href="{f["input"]}">原图</a></p></section>')
    (OUT / "index.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>手部三维重建结果</title>'
        '<style>body{font:17px sans-serif;max-width:1250px;margin:32px auto;padding:0 20px;color:#243447}img{width:100%}section{margin:35px 0}a{color:#146ac2}</style>'
        '<h1>手部三维重建结果图片</h1><p>直接渲染此前保存的三维顶点，没有重新运行模型推理。左边为原图裁剪，右边为相同区域的三维网格叠加图。蓝色为右手，橙色为左手。</p>'
        '<p>前五组来自第一视角微调模型；最后一组为初始权重的双手流程样例。叠加图未模拟物体对手部的遮挡。</p>' + ''.join(cards) + '</html>')
    (OUT / "README.txt").write_text(
        "手部三维重建结果图片\n\n打开 index.html 查看。\n"
        "finetuned_sample_*：已有 HOI4D 第一视角微调模型预测。\n"
        "initial_two_hands_*：已有初始 WiLoR 权重双手流程样例。\n"
        "*_comparison.jpg：原图与网格叠加图的同区域放大对照。\n"
        "*_mesh_overlay.jpg：原始图像尺寸的三维网格叠加图。\n"
        "*_mesh_only.png：相同相机视角的白底三维手部图。\n"
        "*_input.jpg：未经修改的输入图像。\n\n"
        "本次没有重新推理，也未修改保存的三维顶点。仅改变网格显示颜色和光照。\n"
        "蓝色为右手，橙色为左手。叠加图未处理场景物体遮挡。\n"
        "样例按固定索引选取，遇到无检测记录顺延；未按误差挑选。\n"
        "来源路径与校验值见 manifest.json。复现：egohand3d 环境下运行 python render_saved_meshes.py。\n"
    )


if __name__ == "__main__":
    main()
