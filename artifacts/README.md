# 当前全部结果归档

四个 ZIP 都可单独解压，将它们解压到**同一个目录**即可还原完整 `outputs/` 结构。
它们是按文件分组的独立 ZIP，不是 `.z01` 类型的压缩分卷；每个文件仅出现在一个包中。

| 下载 | 大小 | 结果文件数 |
|---|---:|---:|
| [EgoHand3D_results_20261001_part01.zip](EgoHand3D_results_20261001_part01.zip) | 38.70 MiB | 507 |
| [EgoHand3D_results_20261001_part02.zip](EgoHand3D_results_20261001_part02.zip) | 38.60 MiB | 483 |
| [EgoHand3D_results_20261001_part03.zip](EgoHand3D_results_20261001_part03.zip) | 21.43 MiB | 247 |
| [EgoHand3D_results_20261001_part04.zip](EgoHand3D_results_20261001_part04.zip) | 5.07 MiB | 85 |

总计 1322 个结果文件，解压前约 104 MiB、原始内容约 165.54 MiB。
包含本轮所有原始预测、2D/3D/MANO 导出、网格、叠加图、视频、固定 GT、评估、配置、输入清单、
SQLite 一致性备份，以及先前的小型格式/环境检查输出。
保留 `workflow_smoke_01` 中 TF32 造成一致性检查失败的初次运行，不能把它计入通过结果；
修正后的真实图片验证是 `workflow_smoke_02`。

不包含权重、MANO 模型、原始数据集或完整原始视频。
历史摘要另在 `results/hoi4d/`，旧格式示例另在 `sample_outputs/`。
归档保持原始实验路径和样本 ID；迁移后重新推理应创建新 run。

```bash
cd artifacts
sha256sum -c SHA256SUMS
mkdir -p ../unpacked_results
for archive in EgoHand3D_results_20261001_part*.zip; do
  unzip "$archive" -d ../unpacked_results
done
```

[results_archive_manifest.json](results_archive_manifest.json) 记录每个文件的 SHA-256、大小及所属 ZIP。
每个包内也有相应清单。解压后直接打开
`outputs/validation_20261001/comparison/report.html` 查看原始 HTML 报告。
