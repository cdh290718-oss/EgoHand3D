#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate.sh

python - <<'PY'
from pathlib import Path
import torch
import pytorch_lightning as pl
import ultralytics

required = [
    "pretrained_models/wilor_final.ckpt",
    "pretrained_models/model_config.yaml",
    "pretrained_models/detector.pt",
    "mano_data/MANO_RIGHT.pkl",
    "mano_data/MANO_LEFT.pkl",
    "wilor/configs_hydra/train.yaml",
]

print("python environment OK")
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("pytorch_lightning", pl.__version__)
print("ultralytics", ultralytics.__version__)

missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))
print("required files OK")
PY

python train.py --cfg job \
  data=hoi4d_clean \
  experiment=wilor_vit_refinenet \
  trainer=gpu \
  trainer.devices=1 \
  GENERAL.TOTAL_STEPS=1 >/tmp/egohand3d_hydra_check.txt
echo "hydra config OK"

if [ -d "${WILOR_TRAINING_DATA}" ]; then
python - <<'PY'
from pathlib import Path
import os
import hydra
from wilor.configs import dataset_config
from wilor.datasets import WiLoRDataModule

root = Path.cwd()
os.environ["PROJECT_ROOT"] = str(root)
os.environ.setdefault("WILOR_TRAINING_DATA", str(root / "data/hamer_training_data"))
with hydra.initialize_config_dir(config_dir=str(root / "wilor/configs_hydra"), version_base="1.2"):
    cfg = hydra.compose(
        config_name="train",
        overrides=[
            "data=hoi4d_clean",
            "experiment=wilor_vit_refinenet",
            "trainer=gpu",
            "trainer.devices=1",
            "TRAIN.BATCH_SIZE=1",
            "TRAIN.NUM_TRAIN_SAMPLES=1",
            "GENERAL.NUM_WORKERS=0",
            "GENERAL.TOTAL_STEPS=1",
        ],
    )
dm = WiLoRDataModule(cfg, dataset_config())
dm.setup("fit")
batch = next(iter(dm.train_dataloader()["img"]))
print("train dataloader OK:", tuple(batch["img"].shape))
PY
else
  echo "skip train dataloader check: WILOR_TRAINING_DATA does not exist (${WILOR_TRAINING_DATA})"
fi

if [ -d "examples/images" ]; then
  python export_2d_joints.py \
    --img_folder examples/images \
    --out_folder outputs/check_setup_2d \
    --file_type test1.jpg \
    --checkpoint pretrained_models/wilor_final.ckpt \
    --cfg pretrained_models/model_config.yaml \
    --detector pretrained_models/detector.pt >/tmp/egohand3d_infer_check.txt
  echo "2D inference OK: outputs/check_setup_2d"
else
  echo "skip 2D inference check: examples/images is not included in public releases"
fi
