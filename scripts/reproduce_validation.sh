#!/usr/bin/env bash
set -euo pipefail

# Run from a prepared EgoHand3D checkout; no training is started by this script.
if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 HOI4D_SAMPLE_DIR BASE_CKPT BASE_CFG FINE_CKPT FINE_CFG [VIDEO]" >&2
  exit 2
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
sample_dir="$(realpath "$1")"
base_ckpt="$(realpath "$2")"
base_cfg="$(realpath "$3")"
fine_ckpt="$(realpath "$4")"
fine_cfg="$(realpath "$5")"
result_dir="${EGOHAND3D_RESULT_DIR:-outputs/reproduction}"
detector="${WILOR_DETECTOR:-pretrained_models/detector.pt}"
python_bin="${EGOHAND3D_PYTHON:-python}"
if [[ -e "$result_dir" ]]; then
  echo "Choose a fresh EGOHAND3D_RESULT_DIR; existing results: $result_dir" >&2
  exit 2
fi
"$python_bin" -m egohand3d.workflow scan --input "$sample_dir/images" --out "$result_dir/preflight.json"
"$python_bin" -m egohand3d.workflow process --input "$sample_dir/images" \
  --checkpoint "$base_ckpt" --cfg "$base_cfg" --detector "$detector" \
  --out "$result_dir/hoi4d_baseline" --no-mesh
"$python_bin" -m egohand3d.workflow process --input "$sample_dir/images" \
  --checkpoint "$fine_ckpt" --cfg "$fine_cfg" --detector "$detector" \
  --out "$result_dir/hoi4d_finetuned" --no-mesh
"$python_bin" -m egohand3d.workflow prepare-gt --run "$result_dir/hoi4d_baseline" \
  --sample-dir "$sample_dir" --out "$result_dir/hoi4d_gt.json"
for variant in baseline finetuned; do
  "$python_bin" -m egohand3d.workflow benchmark --run "$result_dir/hoi4d_$variant" \
    --ground-truth "$result_dir/hoi4d_gt.json" --out "$result_dir/eval_$variant"
done
"$python_bin" -m egohand3d.workflow compare \
  --summaries "$result_dir/eval_baseline/summary.json" "$result_dir/eval_finetuned/summary.json" \
  --out "$result_dir/comparison"
if [[ $# -eq 6 ]]; then
  video_path="$(realpath "$6")"
  "$python_bin" -m egohand3d.workflow process --input "$video_path" \
    --checkpoint "$base_ckpt" --cfg "$base_cfg" --detector "$detector" \
    --out "$result_dir/video" --stride 2 --limit 32 --no-mesh
  "$python_bin" -m egohand3d.workflow sequence --run "$result_dir/video" \
    --out "$result_dir/video_sequence" --fill-gaps
  "$python_bin" -m egohand3d.workflow render-sequence --run "$result_dir/video_sequence"
fi
printf 'Results: %s/comparison/report.md\n' "$result_dir"
