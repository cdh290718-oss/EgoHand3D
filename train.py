from typing import Optional, Tuple
import os
import signal
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
os.environ.setdefault("PROJECT_ROOT", str(root))

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from torch.distributed.elastic.multiprocessing.errors import record
from yacs.config import CfgNode

from wilor.configs import dataset_config
from wilor.datasets import WiLoRDataModule
from wilor.models.wilor import WiLoR
from wilor.utils.misc import log_hyperparameters, task_wrapper
from wilor.utils.pylogger import get_pylogger


signal.signal(signal.SIGUSR1, signal.SIG_DFL)

log = get_pylogger(__name__)


@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str) -> None:
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, "model_config.yaml"))
    with open(os.path.join(rootdir, "dataset_config.yaml"), "w", encoding="utf-8") as f:
        f.write(dataset_cfg.dump())


def configure_training_data_env(cfg: DictConfig) -> None:
    data_root = Path(str(OmegaConf.select(cfg, "paths.training_data_dir", default="../hamer/hamer_training_data")))
    if not data_root.is_absolute():
        data_root = (root / data_root).resolve()
    os.environ.setdefault("WILOR_TRAINING_DATA", str(data_root))
    log.info(f"WILOR_TRAINING_DATA={os.environ['WILOR_TRAINING_DATA']}")


def resolve_project_path(path_value: str) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def load_pretrained_weights_if_needed(model: WiLoR, cfg: DictConfig, ckpt_path: Optional[str]) -> None:
    if ckpt_path is not None:
        return

    pretrained_ckpt = OmegaConf.select(cfg, "MODEL.PRETRAINED_CKPT", default=None)
    if not pretrained_ckpt:
        return

    pretrained_path = resolve_project_path(pretrained_ckpt)
    if not pretrained_path.exists():
        raise FileNotFoundError(f"MODEL.PRETRAINED_CKPT not found: {pretrained_path}")

    log.info(f"Loading full WiLoR pretrained weights from {pretrained_path}")
    checkpoint = torch.load(str(pretrained_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    log.info(f"Loaded pretrained weights with {len(missing)} missing and {len(unexpected)} unexpected keys.")
    if missing:
        log.info(f"Missing keys sample: {missing[:20]}")
    if unexpected:
        log.info(f"Unexpected keys sample: {unexpected[:20]}")


@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    configure_training_data_env(cfg)

    dataset_cfg = dataset_config()
    save_configs(cfg, dataset_cfg, cfg.paths.output_dir)

    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path is None and cfg.GENERAL.get("RESUME", True):
        last_ckpt = Path(cfg.paths.output_dir) / "checkpoints" / "last.ckpt"
        if last_ckpt.exists():
            ckpt_path = str(last_ckpt)

    datamodule = WiLoRDataModule(cfg, dataset_cfg)
    model = WiLoR(cfg, init_renderer=OmegaConf.select(cfg, "EXTRA.INIT_RENDERER", default=False))
    load_pretrained_weights_if_needed(model, cfg, ckpt_path)

    logger = TensorBoardLogger(
        os.path.join(cfg.paths.output_dir, "tensorboard"),
        name="",
        version="",
        default_hp_metric=False,
    )
    loggers = [logger]

    best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, "checkpoints"),
        filename="step={step:06d}-val_loss={val_loss:.5f}",
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,
        save_last=False,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )
    last_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, "checkpoints"),
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS,
        save_last=True,
        save_top_k=0,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [best_checkpoint_callback, last_checkpoint_callback, lr_monitor]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
        plugins=(SLURMEnvironment(requeue_signal=signal.SIGUSR2) if (cfg.get("launcher", None) is not None) else None),
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
    log.info("Fitting done")


@record
@hydra.main(version_base="1.2", config_path=str(root / "wilor/configs_hydra"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)


if __name__ == "__main__":
    main()
