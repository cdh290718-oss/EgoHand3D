from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egohand3d.io_utils import collect_image_paths, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate EgoHand3D runtime files and Python dependencies.")
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained_models/wilor_final.ckpt"))
    parser.add_argument("--cfg", type=Path, default=Path("pretrained_models/model_config.yaml"))
    parser.add_argument("--detector", type=Path, default=Path("pretrained_models/detector.pt"))
    parser.add_argument("--mano_right", type=Path, default=Path("mano_data/MANO_RIGHT.pkl"))
    parser.add_argument("--mano_left", type=Path, default=Path("mano_data/MANO_LEFT.pkl"))
    parser.add_argument("--image_dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/validate_setup.json"))
    parser.add_argument("--load_models", action="store_true", help="Actually load WiLoR and detector; slower")
    return parser.parse_args()


def check_import(module_name: str) -> dict:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"module": module_name, "ok": False, "error": str(exc)}
    version = getattr(module, "__version__", None)
    return {"module": module_name, "ok": True, "version": version}


def check_path(path: Path, label: str) -> dict:
    return {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "is_symlink": path.is_symlink(),
    }


def torch_status() -> dict:
    try:
        import torch
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }


def load_model_status(args: argparse.Namespace) -> dict:
    if not args.load_models:
        return {"requested": False}
    try:
        from egohand3d.inference import load_runtime
        runtime = load_runtime(str(args.checkpoint), str(args.cfg), str(args.detector), device_name="auto")
        return {
            "requested": True,
            "ok": True,
            "device": str(runtime.device),
            "model_class": runtime.model.__class__.__name__,
            "detector_class": runtime.detector.__class__.__name__,
        }
    except Exception as exc:
        return {"requested": True, "ok": False, "error": str(exc)}


def image_dir_status(image_dir: Path | None) -> dict:
    if image_dir is None:
        return {"provided": False}
    if not image_dir.exists():
        return {"provided": True, "exists": False, "path": str(image_dir)}
    paths = collect_image_paths(image_dir, None)
    return {
        "provided": True,
        "exists": True,
        "path": str(image_dir),
        "num_images": len(paths),
        "first_images": [str(p) for p in paths[:10]],
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# EgoHand3D Setup Validation",
        "",
        "## Python",
        "",
        f"- Executable: `{report['python']['executable']}`",
        f"- Version: `{report['python']['version']}`",
        f"- Platform: `{report['python']['platform']}`",
        "",
        "## Imports",
        "",
        "| Module | OK | Version/Error |",
        "| --- | --- | --- |",
    ]
    for item in report["imports"]:
        detail = item.get("version") if item["ok"] else item.get("error")
        lines.append(f"| {item['module']} | {item['ok']} | {detail or ''} |")
    lines.extend(["", "## Required Files", "", "| Label | Exists | Path |", "| --- | --- | --- |"])
    for item in report["paths"]:
        lines.append(f"| {item['label']} | {item['exists']} | `{item['path']}` |")
    lines.extend(["", "## Torch", ""])
    for key, value in report["torch"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Model Loading", ""])
    for key, value in report["model_loading"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    imports = [
        "cv2",
        "numpy",
        "torch",
        "pytorch_lightning",
        "hydra",
        "ultralytics",
        "trimesh",
        "pyrender",
        "webdataset",
    ]
    report = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "imports": [check_import(name) for name in imports],
        "paths": [
            check_path(args.checkpoint, "WiLoR checkpoint"),
            check_path(args.cfg, "WiLoR config"),
            check_path(args.detector, "YOLO detector"),
            check_path(args.mano_right, "MANO right"),
            check_path(args.mano_left, "MANO left"),
        ],
        "torch": torch_status(),
        "image_dir": image_dir_status(args.image_dir),
        "model_loading": load_model_status(args),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    write_markdown(args.out.with_suffix(".md"), report)
    failures = []
    failures.extend([item["module"] for item in report["imports"] if not item["ok"]])
    failures.extend([item["label"] for item in report["paths"] if not item["exists"]])
    if report["model_loading"].get("requested") and not report["model_loading"].get("ok"):
        failures.append("model_loading")
    print(f"[validate] wrote {args.out}")
    if failures:
        print("[validate] failures:", ", ".join(failures))
        raise SystemExit(1)
    print("[validate] OK")


if __name__ == "__main__":
    main()
