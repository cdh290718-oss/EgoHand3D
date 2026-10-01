# 运行与复现说明

## 已验证环境

本轮在 Linux、Python 3.10、PyTorch 2.7.0+cu128、NVIDIA A100 80GB 上完成真实推理。
完整实际包版本见 `dependencies/environment_versions.json`；`requirements.txt` 是安装依赖清单，
不是所有平台通用的严格锁文件。GPU 版 PyTorch 应与本机 CUDA/驱动兼容。

```bash
conda env create -f environment.yml
conda activate egohand3d
python -m pip install -r requirements.txt
```

应用单元测试可直接安装 `requirements-dev.txt` 后执行，不需要模型数据。

## 外部运行时与模型

本仓库使用已经扩展过的本地 WiLoR 运行时，其源码指纹在
`dependencies/runtime_manifest.json`。官方参考仓库为 https://github.com/rolpotamias/WiLoR 。
本次不打包重建网络、检测器权重或 MANO 数据。

在取得有权使用的兼容运行时和模型后，建立以下链接（路径换成自己的实际目录）：

```bash
ln -s /path/to/compatible/WiLoR/wilor wilor
ln -s /path/to/compatible/WiLoR/pretrained_models pretrained_models
ln -s /path/to/compatible/WiLoR/mano_data mano_data
```

如果目标路径已存在，请先检查已有目录，不要覆盖。所需模型身份见
`dependencies/model_assets.json`。微调权重文件约 7.7 GB，初始 WiLoR 约 2.56 GB；没有嵌入 Git。

最低运行时接口包括 `wilor.models.load_wilor`、`wilor.datasets.vitdet_dataset.ViTDetDataset`、
`wilor.models.mano_wrapper.MANO`、`wilor.utils.camera.cam_crop_to_full`、
`wilor.utils.yolo_loader.load_yolo_detector` 与 `wilor.utils.recursive_to`。
训练还需要本地训练版 `WiLoR`/`WiLoRDataModule` 和 `configs_hydra`。
官方仓库缺少其中部分本地辅助模块；**目前没有证明仅凭官方干净克隆就可复现本轮全部训练与推理**。
当前发布保存完整应用流程、配置和结果，运行模型部分仍需这些外部条件。

## 重新运行两模型比较

准备好原始 HOI4D 样例目录，其中包含 `images/` 和对应的可信本地标注 pickle；
不要用 `sample_outputs/` 中的示意 JSON 充当真实 GT。

```bash
bash scripts/reproduce_validation.sh \
  /path/to/hoi4d_samples2 \
  /path/to/wilor_final.ckpt /path/to/initial_model_config.yaml \
  /path/to/finetuned.ckpt /path/to/finetuned_model_config.yaml \
  /path/to/cam0.mp4
```

末尾视频可省略。默认输出 `outputs/reproduction`，已存在则拒绝覆盖；
可设置 `EGOHAND3D_RESULT_DIR`、`EGOHAND3D_PYTHON`、`WILOR_DETECTOR`。
脚本运行真实推理、GT 转换、两次评估及对比，不启动新的训练。
输入路径/文件元数据参与样本 ID，新位置应重新生成该批次的 GT，不能混用原始归档的 ID。

## 仅复核归档结果

将 `artifacts/` 的四个 ZIP 解压到同一目录，得到原始 `outputs/` 结构。
每个原始 run 保留 `run_config.json`、`manifest.json`、`index.json`、逐帧 JSON、导出及任务数据库。
路径记录保持当时原样，便于核查实验；迁移后不要把旧数据库当成可直接恢复的新任务。
SQLite 为一致性备份；锁文件、WAL/SHM 等瞬时文件不参与归档。

```bash
python -m egohand3d.workflow benchmark \
  --run /path/to/unpacked/outputs/validation_20261001/hoi4d_baseline \
  --ground-truth /path/to/unpacked/outputs/validation_20261001/hoi4d_gt.json \
  --out outputs/recheck_baseline
```

此步骤读取保存的预测重新计算指标，不加载 GPU 模型。微调模型同理。
比较必须使用相同 GT、相同评价代码及相同协议。

## 结果解释

本轮 100 张图片未核实训练/测试去重，仅作固定样例验证。
本轮二维误差、历史 GT 裁剪三维误差和视频一致性误差衡量不同问题，不能放到同一口径下比较。
单目三维结果采用 WiLoR 假设焦距；MANO 重建一致并不证明绝对尺度准确。
视频插值明确标记，精度评估剔除插值手。时序平滑降低运动速度也不自动等于更准确。
