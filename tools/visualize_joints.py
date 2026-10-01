from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from egohand3d.io_utils import collect_image_paths, read_json
from egohand3d.visualization import draw_record_on_image, make_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw EgoHand3D 2D joints from JSON files onto images.")
    parser.add_argument("--img_folder", type=Path, required=True)
    parser.add_argument("--json_dir", type=Path, required=True)
    parser.add_argument("--out_folder", type=Path, default=Path("outputs/joint_overlays"))
    parser.add_argument("--file_type", nargs="+", default=["*.jpg", "*.png", "*.jpeg"])
    parser.add_argument("--json_suffix", default="_kpts2d.json", help="Example: _kpts2d.json or _joints3d.json")
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--draw_indices", action="store_true")
    parser.add_argument("--no_boxes", action="store_true")
    parser.add_argument("--contact_sheet", action="store_true")
    return parser.parse_args()


def candidate_json_paths(json_dir: Path, image_stem: str, suffix: str) -> list[Path]:
    candidates = [
        json_dir / f"{image_stem}{suffix}",
        json_dir / f"{image_stem}_joints3d.json",
        json_dir / f"{image_stem}_kpts2d.json",
        json_dir / f"{image_stem}.json",
    ]
    return [path for path in candidates if path.exists()]


def main() -> None:
    args = parse_args()
    args.out_folder.mkdir(parents=True, exist_ok=True)
    image_paths = collect_image_paths(args.img_folder, args.file_type)
    written = []
    missed = []

    for image_path in image_paths:
        json_paths = candidate_json_paths(args.json_dir, image_path.stem, args.json_suffix)
        if not json_paths:
            missed.append(image_path.name)
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            missed.append(image_path.name)
            continue
        record = read_json(json_paths[0])
        overlay = draw_record_on_image(
            image,
            record,
            score_threshold=args.score_threshold,
            draw_boxes=not args.no_boxes,
            draw_indices=args.draw_indices,
        )
        out_path = args.out_folder / f"{image_path.stem}_overlay.jpg"
        cv2.imwrite(str(out_path), overlay)
        written.append(out_path)
        print(f"[overlay] {image_path.name} <- {json_paths[0].name}")

    if args.contact_sheet and written:
        make_contact_sheet(written, args.out_folder / "contact_sheet.jpg")
    if missed:
        (args.out_folder / "missed.txt").write_text("\n".join(missed) + "\n", encoding="utf-8")
    print(f"[done] overlays={len(written)} missed={len(missed)} out={args.out_folder}")


if __name__ == "__main__":
    main()
