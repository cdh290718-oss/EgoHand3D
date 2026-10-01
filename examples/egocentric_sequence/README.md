# Egocentric Sequence Example

This directory contains a small first-person example package copied from the
local WiLoR workspace at `../WiLoR/20260604_203918_008`.

The package is intentionally lightweight. It includes three `cam0` frames and
their matching outputs, but it does not include the original video, model
weights, MANO files, or full trajectory arrays.

## Files

| Content | Path |
| --- | --- |
| Input frames | `inputs/cam0/cam0_frame_000000.jpg`, `cam0_frame_000100.jpg`, `cam0_frame_000200.jpg` |
| 2D keypoint JSON | `outputs/2d_joints/cam0/` |
| Raw hand detections | `outputs/detections/cam0/` |
| 3D trajectory samples | `outputs/3d_trajectory/cam0_frame_*_trajectory.json` |
| Camera parameters | `outputs/3d_trajectory/camera_params_used.json` |
| Diagnostic overlays | `outputs/overlays/cam0/` |
| Mesh overlay previews | `outputs/mesh_previews/cam0/` |
| File index | `manifest.json`, `manifest.csv` |
| File checksums | `checksums.csv` |

## Example Commands

These commands show how the same kind of outputs can be regenerated when the
runtime assets are available:

```bash
python egohand3d_cli.py export-2d \
  --img_folder examples/egocentric_sequence/inputs/cam0 \
  --out_folder outputs/egocentric_2d
```

```bash
python egohand3d_cli.py visualize \
  --img_folder examples/egocentric_sequence/inputs/cam0 \
  --json_dir examples/egocentric_sequence/outputs/2d_joints/cam0 \
  --out_folder outputs/egocentric_visualize
```

```bash
python egohand3d_cli.py manifest \
  --image_dir examples/egocentric_sequence/inputs/cam0 \
  --pred_dir examples/egocentric_sequence/outputs/2d_joints/cam0 \
  --out outputs/egocentric_manifest.json
```

## Publication Note

These are real first-person example frames. Keep this directory public only if
you intend this sequence to be visible on GitHub.
