# 源程序交存整理稿

软件名称沿用项目已有材料：**第一视角手部三维重建适配与评估软件 V1.0（EgoHand3D）**。
本稿范围是新增的命令行流程应用模块；WiLoR 核心、MANO、检测器、原有推理适配器是其依赖。
申请表、软件说明书与源码所描述的申请范围应一致，不能将本清单称为整个 WiLoR 衍生项目的全部源码。

## 交存文件

- [源程序交存 PDF](source/EgoHand3D_V1.0_source_submission.pdf)
- [可编辑 Word](source/EgoHand3D_V1.0_source_submission.docx)
- [完整源码文本](source/EgoHand3D_V1.0_source_full.txt)
- [源文件与 SHA-256 清单](source/source_manifest.json)
- [页码/原文件/原行号对应表](source/source_line_map.csv)
- [与本地 WiLoR 的代码核对记录](source_overlap_audit.json)

当前范围为 13 个文件、1522 个原始物理行、1359 个非空源码行，排成 **28 页**。
每页 50 个原始非空源码行，最后一页 9 行；长行仅在展示时换行。
仅省略全空白行，保留注释、函数和每个文件的全部有效内容，不拼接不连续函数，不用重复内容凑页。
正文侧栏是“文件编号:原行号”，对应清单中的文件顺序；最后保留命令行程序入口结束部分。
PDF 是已检查的固定分页版本；Word 在不同字体/软件下可能重新分页，转 PDF 后应再次核对。

依据 [国家版权局《计算机软件著作权登记办法》第十条](https://www.ncac.gov.cn/xxfb/flfg/bmgz/202410/t20241015_869486.html)，
通常交存源程序与一种文档的前后各连续 30 页，不足 60 页则提交全部；
通常源程序每页不少于 50 行。这里的单位是“页”，不是前后 30 行。
本次不足 60 页，因此 submission 与 full 是同一份完整文件，不额外制造 60 页版本。

## 纳入与排除

纳入文件逐项记录于 [source_scope.json](source_scope.json)：存储、媒体预检、任务管理、
MANO 编解码、质量检查、统一推理编排、时序关联、GT 适配、指标、报告、离线视频、命令注册与入口。
这些功能仍调用外部/既有接口；不因函数名或文件位置不同就声称核心网络为本项目原创。

排除 `wilor/` 全部核心及训练数据处理实现、模型/检测器/MANO 资产、训练配置、测试、输出数据、
文档生成工具和所有来源未核实的旧应用代码。
尤其排除与本地 WiLoR 完全相同的 `train.py`、`export_2d_joints.py`、
`tools/detect_video_hands.py`、`tools/evaluate_hoi4d_wilor_2djoints.py`，
以及近似改写的 `detect_and_reconstruct.py`。

旧 `egohand3d/inference.py` 中存在复用的投影等辅助函数，也作为既有适配依赖排除。
`io_utils.py`、`visualization.py`、`__init__.py` 同样不在新增范围。
其完整实现仍保留于软件仓库，便于运行和审查，不以排除交存冒充消除了依赖。

新增代码的文件清单和相似性检查只能说明本次整理范围及发现的重复项，不能单独证明权属或保证登记结果。
现有脚本混合来源尚未核实的事实仍保留于来源说明；身份、合作开发、许可等证明材料没有代填。
这次没有向登记机构提交申请。

## 重新生成

```bash
python -m pip install -r requirements-docs.txt
python tools/build_source_submission.py
```

需要 Linux DejaVu Sans Mono 和中文字体；可通过 `--mono-font`、`--chinese-font` 指定。
源码发生任何改动后，应重新生成清单、页码和 PDF，不再沿用旧的 SHA-256。
