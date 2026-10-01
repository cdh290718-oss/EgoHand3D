# 命令行扩展验证记录：2026-10-01

本次完成应用功能开发及真实推理，没有新增训练、没有创建交互界面；本文件随流程和结果归档整理发布。
详细操作方式见 [HEADLESS_WORKFLOW.md](HEADLESS_WORKFLOW.md)。

## 软件验证

- 15 项自动测试全部通过，包括旋转转换、临界旋转插值、漏检分母、候选匹配、轨迹 ID、坏文件、
  视频抽帧、并发写入保护、配置变化拒绝复用、损坏结果修复和归档校验。
- 真实第一视角样例：3 张图片，8 条检测手记录，MANO 关节/网格/投影闭环全部通过。
- 独立 CPU MANO 解码器重新加载一帧中的左右手，也通过同一检查。
- 真实视频：`../WiLoR/20260604_203918_008/cam0.mp4`，stride=2，处理前 32 个采样帧。
  共 88 条检测手记录，全部通过 MANO 一致性检查；序列关联产生 4 个轨迹 ID，
  `--fill-gaps` 额外生成 7 条明确标注的插值记录。
- 视频批次最大网格重建差异约 3.16e-8 米，最大重投影差异约 1.18e-4 像素。
- 恢复已完成的 100 帧任务：processed=0、reused=100，无模型加载，帧遍历与校验约 1.13 秒
  （不含启动、输入预检和最终报告生成）。

首轮冒烟测试发现环境默认 TF32 导致批量推理和单手 MANO 解码出现约 0.1 mm 差异。
随后将新流程设置为完整 FP32；保留原有 1e-5 m 和 0.05 px 的验证阈值，未通过放宽阈值解决问题。
初次失败输出位于 `outputs/workflow_smoke_01`，修正后的样例位于 `outputs/workflow_smoke_02`。

闭环一致性只说明参数导出、重建和投影没有明显不一致，不等于姿态估计真实误差。
视频缺少 GT，轨迹平滑和插值仅作为功能演示，不能据此宣称精度提高。

## 同一批 HOI4D 图像的重新评估

输入为现有 `../WiLoR/hoi4d_samples2/images` 的 100 张图片，使用该样例包对应的真实 `kps2D` 标注。
未验证它与历史训练数据完全不重叠，因此本记录是固定样例验证，不作为独立测试集泛化结论。

两个重建权重均使用同一检测器、conf=0.3、rescale_factor=2.0、相同逐帧输入和完整 FP32。
模型参数：

- 初始模型：`../WiLoR/pretrained_models/wilor_final.ckpt`。
- 已有微调模型：
  `../WiLoR/logs/train/runs/train_hoi4d_clean_renderegomano_half_4gpu_from_wilor_60epoch_bs120_v1/checkpoints/step=063312-val_loss=0.87984.ckpt`，
  使用该 run 的 `model_config.yaml`。

评估规则：同左右手类别、框 IoU≥0.1、一对一匹配；GT 框由 21 个标注点范围外扩 10% 得到。
没有利用关节误差选择预测。平均误差分母为匹配成功的有效关节；全 GT PCK 分母为全部 2100 个标注关节。
标注仅覆盖每张图的目标手，未匹配预测不直接视为假阳性，不报告完整检测精确率。

|指标|初始 WiLoR|已有微调权重|
|---|---:|---:|
|处理图片|100|100|
|有预测的图片|86|86|
|全部预测手数量|94|94|
|匹配的标注手|85/100|85/100|
|漏配标注手|15|15|
|匹配关节平均二维误差|17.0416 px|8.2124 px|
|匹配关节中位误差|12.9248 px|5.5659 px|
|全 GT PCK@10 px，漏检计错|32.00%|63.76%|
|全 GT PCK@20 px，漏检计错|59.00%|78.38%|
|MANO 闭环失败|0/94|0/94|

这些指标由新流程实际重新运行得到，不是从旧摘要或 `sample_outputs` 的示意数值复制。
精度差异来自两个已有权重，不归因于新增的任务管理、导出、质量检查等应用模块。
这次真实标注评估只有二维指标，不能把软件对三维指标的支持等同于已经验证了真实三维精度。

固定 GT 文件 SHA-256：
`d6352d913e18ded4c61acaab838fc64a5825fc0ba2cfc13fefe6cd79ae90f7b2`。

## 随仓库保存的结果入口

- [两模型对比报告](../results/validation_20261001/comparison/report.html)
- [初始模型评估](../results/validation_20261001/eval_baseline/report.html)
- [微调模型评估](../results/validation_20261001/eval_finetuned/report.html)
- [视频时序处理报告](../results/validation_20261001/video_sequence/report.html)
- [原始/处理后并排视频](../results/validation_20261001/video_sequence/f191bb737fc68685d387_comparison.mp4)
- [独立 CPU MANO 检查](../results/validation_20261001/cpu_mano_check.json)

可浏览报告位于 `results/validation_20261001/`，全部原始 `outputs/` 另存为 [四个结果归档](../artifacts/README.md)。
每次推理的 `run_config.json` 留存实际资产、输入和代码身份，`tasks.sqlite` 备份留存任务状态与输出 SHA-256。
