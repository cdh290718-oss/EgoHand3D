# Public Sample Outputs

This directory contains small sample artifacts for every EgoHand3D command shown
in the main README.

The image artifacts use first-person frames and overlays from
`../examples/egocentric_sequence`. Some compact numeric files are schema-valid
illustrative values intended to show output formats and file layout. Full model
outputs require the external runtime assets described in the main README.

## Command To Output Map

| README command | Output directory | Representative files |
| --- | --- | --- |
| `python egohand3d_cli.py validate --image_dir /path/to/images --out outputs/validate_setup.json` | `validate/` | `validate_setup.json`, `validate_setup.md` |
| `python egohand3d_cli.py reconstruct --img_folder /path/to/images --out_folder outputs/demo_mesh` | `reconstruct/` | `sample_frame_0.obj`, `sample_frame_overlay.png`, `summary.json` |
| `python egohand3d_cli.py export-2d --img_folder /path/to/images --out_folder outputs/2d_joints` | `2d_joints/` | `sample_frame_kpts2d.json`, `summary.json`, `records.jsonl` |
| `python egohand3d_cli.py export-3d --img_folder /path/to/images --out_folder outputs/3d_joints` | `3d_joints/` | `sample_frame_joints3d.json`, `summary.json`, `records.jsonl` |
| `python egohand3d_cli.py export-mano --img_folder /path/to/images --out_folder outputs/mano_params` | `mano_params/` | `sample_frame_mano.json`, `summary.json` |
| `python egohand3d_cli.py video --video /path/to/video.mp4 --out_folder outputs/video_hands --output_video outputs/video_hands/overlay.mp4` | `video_hands/` | `jsons/frame_000000.json`, `overlay_frames/frame_000000.png`, `summary.json` |
| `python egohand3d_cli.py evaluate --sample_dir /path/to/hoi4d_samples --pred_dir outputs/2d_joints --out_dir outputs/evaluate` | `evaluate/` | `summary.json`, `summary.csv`, `details.csv`, `per_joint_summary.json`, `overlays/sample_frame_eval_overlay.png` |
| `python egohand3d_cli.py report --result_dir outputs --image_dir /path/to/images --out_dir outputs/report --recursive` | `report/` | `report.md`, `report.json`, `json_files.csv` |
| `python egohand3d_cli.py manifest --image_dir /path/to/images --pred_dir outputs/2d_joints --out outputs/manifest.json` | `manifest/` | `manifest.json`, `manifest.csv`, `manifest.md` |
| `python egohand3d_cli.py merge --joints2d_dir outputs/2d_joints --joints3d_dir outputs/3d_joints --mano_dir outputs/mano_params --mesh_dir outputs/demo_mesh --out outputs/merged_exports.json` | `merged_exports/` | `merged_exports.json`, `merged_exports.csv`, `merged_exports.md` |
| `python egohand3d_cli.py visualize --img_folder /path/to/images --json_dir outputs/2d_joints --out_folder outputs/visualize` | `visualize/` | `sample_frame_overlay.png`, `summary.json` |
| `bash scripts/train_hoi4d_clean_4gpu.sh` | `training/` | `smoke_train_log.txt`, `expected_training_outputs.json` |
| `bash scripts/train_hoi4d_clean_egodex_2x4gpu_volc.sh` | `training/` | `smoke_train_log.txt`, `expected_training_outputs.json` |
| `python train.py exp_name=smoke_hoi4d_clean_1gpu ... GENERAL.TOTAL_STEPS=20` | `training/` | `smoke_train_log.txt`, `expected_training_outputs.json` |

## Included Demo Input

- `assets/sample_frame.png`: first-person input frame used by the example
  artifacts.
- `checksums.csv`: SHA-256 checksums for all sample files.

For real first-person example frames and matching outputs, see
`../examples/egocentric_sequence`.
