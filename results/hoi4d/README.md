# HOI4D Results and Weight Manifest

This directory collects compact HOI4D-related result summaries copied from `../WiLoR` for the GitHub repository.

## Included In Git

- `summaries/`: compact JSON/CSV/TXT/LOG result summaries from `../WiLoR/eval_outputs`, `../WiLoR/hoi4d_samples`, and `../WiLoR/hoi4d_samples2`.
- `summary_file_manifest.csv`: source-to-repository mapping for the copied result summaries.
- `checkpoint_manifest.csv`: local checkpoint manifest discovered under `../WiLoR/logs/train/runs` and `../WiLoR/pretrained_models`.

## Not Stored Directly In Git

Large model files are intentionally not committed to normal Git history. GitHub blocks normal repository files larger than 100 MiB, and the discovered HOI4D checkpoints are around 7.2 GiB each. Upload them through GitHub Releases as split assets, or host them on Hugging Face / object storage and link them here.

## Discovery Summary

- Total checkpoint/model files discovered: `43`
- HOI4D-related checkpoint files discovered: `33`
- Compact result summary files copied into this repository: `88`
- Largest discovered checkpoint/model file: `../WiLoR/logs/train/runs/train_hoi4d_clean_egofishhands_quarter_4gpu_from_wilor_60epoch_bs120_v1/checkpoints/last.ckpt` (`7.164` GiB)

## Recommended Weight Release Method

Use GitHub Release assets with split files below 2 GiB, for example:

```bash
mkdir -p release_weights
split -b 1900M ../WiLoR/logs/train/runs/<run>/checkpoints/<checkpoint>.ckpt release_weights/<checkpoint>.ckpt.part-
sha256sum ../WiLoR/logs/train/runs/<run>/checkpoints/<checkpoint>.ckpt > release_weights/<checkpoint>.ckpt.sha256
```

Users can reconstruct the checkpoint with:

```bash
cat <checkpoint>.ckpt.part-* > <checkpoint>.ckpt
sha256sum -c <checkpoint>.ckpt.sha256
```

## Candidate HOI4D Checkpoints

See `checkpoint_manifest.csv` for the full list.
