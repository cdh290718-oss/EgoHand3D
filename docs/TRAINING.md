# 第一视角适配训练记录

本轮整理现有训练资料，未启动新训练。EgoHand3D 的模型路线是沿用 WiLoR 网络，
使用第一视角数据进行训练/微调。应用流程和网络训练应分别说明。

## 对应本轮微调评估的已有实验

- run：`train_hoi4d_clean_renderegomano_half_4gpu_from_wilor_60epoch_bs120_v1`
- 使用权重：`step=063312-val_loss=0.87984.ckpt`
- 初始权重：`wilor_final.ckpt`
- 优化设置：学习率 1e-5、weight decay 1e-4、每进程 batch size 120、4 GPU DDP。
- 配置的最大步数：86376；本轮评估选用 step 63312 的已有 checkpoint。
- 训练精度配置为 16；本轮推理与 MANO 闭环检查使用完整 FP32。
- seed 配置为空，deterministic=false；不能承诺逐比特复现训练轨迹。

| 数据分组 | 训练采样权重/配置条目 |
|---|---:|
| HOI4D-CLEAN-TRAIN | 391003 |
| RENDEREGOMANO-HALF-TRAIN | 300000 |
| HOI4D-CLEAN-VAL | 20328 |
| RENDEREGOMANO-VAL | 11000 |

完整配置在 [hoi4d_renderegomano_half](../configs/training/hoi4d_renderegomano_half/)。
`*.original.yaml` 是实验原件；其中绝对路径仅保留为记录。
`train.yaml` 将初始权重路径改为环境变量，其他参数沿用该 run。
`dataset_config.yaml` 保存实际 shard 命名、数量与数据集条目。

## 数据前置条件

需要已转换成兼容 WiLoR/WebDataset 格式的本地数据，设置 `WILOR_TRAINING_DATA` 指向数据根。
原始数据集、完整转换流水线与许可受限资产不包含在本仓库；
记录中的训练数量不能替代对样本来源和训练/测试划分的检查。

与此实验关联的本地 Hydra 配置另存于
`configs/training/local_runtime_hydra/`。这些文件是依赖配置归档，不计入新增交存源码。

## 使用已准备好的兼容训练环境

```bash
export WILOR_TRAINING_DATA=/path/to/hamer_training_data
export WILOR_INITIAL_CKPT=/path/to/wilor_final.ckpt
export PROJECT_ROOT="$PWD"
torchrun --nproc_per_node=4 --master_port=29500 train.py \
  --config-path configs/training/hoi4d_renderegomano_half --config-name train \
  hydra.run.dir=logs/train/reproduction_hoi4d_renderegomano_half
```

这条命令会启动较长的 4 GPU 训练，只应在数据与资源准备好后手动运行。
归档配置和环境依赖已保存；本次发布没有重新跑完训练来检验跨机器复现。
`train.py` 与本地 WiLoR 的旧脚本相同，保留来源说明并排除于新增应用层交存稿。

历史其他权重/结果见 [results/hoi4d](../results/hoi4d/)。历史评估可能使用 GT 裁剪和不同匹配方法，
应读取每个摘要的设置，不把历史数字直接并入本轮 100 张图检测后评估。
