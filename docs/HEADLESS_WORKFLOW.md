# EgoHand3D 命令行处理流程

本流程提供图片/视频预检、统一推理、MANO 参数校验、可恢复任务、视频后处理、评估和静态报告。
不包含交互界面、Web 服务或自动训练。旧命令保留，新流程以 `process` 为入口。

## 环境与快速开始

在项目根目录执行，使用已有 `egohand3d` Conda 环境：

```bash
source scripts/activate.sh
python egohand3d_cli.py scan --input examples/egocentric_sequence/inputs/cam0 --out outputs/preflight.json
python egohand3d_cli.py process --input examples/egocentric_sequence/inputs/cam0 --out outputs/my_run
python egohand3d_cli.py jobs --run outputs/my_run
```

推理沿用外部 WiLoR 后端，需要 `wilor/`、WiLoR/检测器权重、MANO_RIGHT.pkl 和 mano_mean_params.npz。
新应用模块依赖 NumPy、OpenCV、SciPy；静态图表使用 Matplotlib。没有增加界面框架。

`scan` 不加载模型，检查文件可解码性、图片大小、视频帧数/帧率。文件自然排序，支持
`--recursive` 与 `--hash-inputs`。无法读取的媒体单独标记；非媒体文件计入忽略数量。
目录中同名图片使用包含完整源路径的样本标识，不会互相覆盖。

`process` 每帧执行一次检测及重建，共享结果生成：

```text
manifest.json          输入预检与身份
run_config.json        配置、模型/输入身份、代码 SHA-256
tasks.sqlite           任务状态、尝试次数、错误、输出校验和
index.json / index.csv 输入与逐帧结果的对应清单
records/               原始预测、标准 MANO、重建校验、质量标记
2d/                    二维关节
3d/                    三维关节及相机平移
mano/                  标准 MANO 参数包
meshes/                相机坐标中的 OBJ 网格（米）
overlays/              二维骨架叠加图
summary.json           汇总与最近一次运行耗时
report.md / report.html 静态报告，不启动服务器
```

完整记录始终保存网格顶点，便于校验；`--no-mesh` 仅跳过 OBJ，`--no-overlay` 跳过叠加图。
`--limit N` 限制本次选定的总帧数，`--stride N` 对视频抽帧。图像目录不自动被当作连续视频。
确为等间隔帧序列时可指定 `--sequence-fps`；不要给来自不同场景的图片伪造连续时间戳。
视频时间戳由容器帧率计算，当前假定恒定帧率；变帧率视频应先规范时间轴。

## 恢复、重试和取消

```bash
# 参数必须与创建任务时一致
python egohand3d_cli.py process --input examples/egocentric_sequence/inputs/cam0 --out outputs/my_run --resume
python egohand3d_cli.py jobs --run outputs/my_run --cancel
```

只有输出校验和全部正确的帧才会复用；文件丢失或损坏会重算该帧。
完全完成的任务恢复时不加载大型模型。代码、输入身份、权重身份或处理选项变化时拒绝复用，
应选择新的输出目录。默认模型身份使用真实路径、大小和纳秒修改时间；
需要权重内容校验和时在首次运行加入 `--hash-assets`，恢复时保留该选项。

默认每个失败任务再重试一次，可用 `--retries` 调整。`--fail-fast` 遇到最终失败就退出；
否则继续处理其他帧，最后返回非零退出码。Ctrl+C、SIGTERM 和 `jobs --cancel` 可取消任务；
取消会保留完成的帧。取消请求在帧与帧之间检查，不能即时中断正在运行的 GPU 内核。
同一输出目录使用进程锁，防止两个推理进程同时写入。

质量标记包括越界关节、非正深度、退化骨长、低检测分数和 MANO 一致性失败。
检测分数不等于独立的逐关节置信度；没有 GT 时，“含预测帧比例”也不等于检测召回率。

## MANO 格式与闭环验证

标准格式为 `egohand3d.mano/1`，同时提供：

- `pose_rotmat`: 16×3×3，第一项是全局旋转，其余 15 项是手部局部旋转。
- `pose_axis_angle`: 16×3，弧度；验证时检查它与旋转矩阵表示同一旋转。
- `betas`: 10 个形状系数。
- `translation_camera`: 三维平移，米；`intrinsics`: 3×3 投影内参。
- `hand_side`、`model_hand`、`mirror_model_x`、单位、坐标约定和图像大小。

重要约定：WiLoR 以 RIGHT MANO 模型表示两只手，左手在解码后镜像局部几何的 x，
然后加相机平移。这不是可直接传给 LEFT MANO 模型的左手姿态参数。
镜像网格导出时同步翻转三角面绕序。三维坐标轴为 x 向右、y 向下、z 向前。
旧字段 `joints_3d_root` 指 MANO 原点下的几何，未保证腕关节严格为零；做根对齐评估时显式减腕关节。

当前内参来自 WiLoR 假设焦距，标准文件明确标注来源；不是相机标定结果，也不保证单目绝对尺度准确。

```bash
python egohand3d_cli.py mano-check --input outputs/my_run/records/实际样本ID.json --out outputs/mano_check.json
```

对完整帧执行导出参数→MANO→关节/网格→二维投影，与原预测比较。
默认最大几何偏差阈值 1e-5 米，最大投影偏差 0.05 像素。
推理和解码使用完整 FP32 矩阵运算，避免环境默认 TF32 导致批量/单手解码差异。
该检查证明序列化与重建一致，不证明预测接近真实手姿。

## 视频关联、平滑和显式补全

```bash
python egohand3d_cli.py process --input /path/to/video.mp4 --out outputs/video --stride 2 --limit 100 --no-mesh
python egohand3d_cli.py sequence --run outputs/video --out outputs/video_processed --max-gap 3 --smoothing-seconds 0.08
python egohand3d_cli.py render-sequence --run outputs/video_processed
```

关联采用左右手约束、框重叠和中心距离的一对一匹配。它是应用基线，遮挡、快速运动、多人交叉时
仍可能换 ID。旋转沿 SO(3) 最短路径平滑，形状/平移线性平滑，按时间差调整滤波权重。
原始 `mano` 和几何保持在记录中，处理结果存于 `processed_mano`、`processed`。

默认不补全。加入 `--fill-gaps` 才在同 ID 的两个检测之间插值不超过 `--max-gap` 个处理帧；
补全标为 `observation=interpolated`，记录两端来源与插值权重。不存在向无限未来外推。
补全不计入检测召回或姿态精度统计。关联跨缺失保留并不意味着识别出了真实身份。

静态输出包括轨迹图和原始/处理后并排 MP4，不需要交互界面。
MP4 导出要求等间隔帧；不等间隔结果可查看 JSON 和轨迹图。
平均速度只是运动描述，平滑可能引入滞后；有标注时应同时验证精度。

## 固定标注评估与对比

支持二维像素误差、像素 PCK、三维相机坐标 MPJPE、腕根对齐 MPJPE、相似变换对齐 PA-MPJPE。
三维输入必须明确以米表示，报告转换为毫米。匹配用同侧框 IoU，一对一分配，
不会用 GT 关节误差挑选最佳预测。均值仅在匹配成功的有效关节上计算，必须同时报告漏检。
全 GT PCK 将漏检记为不正确，避免只看成功样本。

固定 GT JSON 示例结构（坐标数组应填真实 21 关节，不使用示意数值做实验）：

```text
schema: egohand3d.ground_truth/1
joint_order: OpenPose21
length_unit: m
camera_axes: x_right_y_down_z_forward
labels_complete: true 或 false
frames:
  sample_id: 与 manifest/index 一致
  hands:
    hand_side: left 或 right
    bbox_xyxy: [x1,y1,x2,y2]
    joints_2d_xy: 21×2（可选）
    joints_3d_camera: 21×3（可选）
    valid: 21 个布尔值（可选，默认全部有效）
```

现有可信本地 HOI4D 数据可转换为二维 GT：

```bash
python egohand3d_cli.py process --input ../WiLoR/hoi4d_samples2/images --out outputs/hoi4d_baseline --no-mesh
python egohand3d_cli.py prepare-gt --run outputs/hoi4d_baseline --sample-dir ../WiLoR/hoi4d_samples2 --out outputs/hoi4d_gt.json
python egohand3d_cli.py benchmark --run outputs/hoi4d_baseline --ground-truth outputs/hoi4d_gt.json --out outputs/hoi4d_eval
```

HOI4D 适配器读取该数据集的本地 pickle，不用于不可信文件。
匹配框由 GT 二维关节范围外扩 10% 得到，报告保留该规则。
样例标注仅覆盖目标手，其他可见手可能未标注，因此该适配器不报告完整检测精确率。
它不直接转换 HOI4D 的三维 MANO 标注，避免混淆不同左右手、均值姿态和坐标约定。

另一个权重运行同一输入后，复用同一 GT 文件评估，再比较：

```bash
python egohand3d_cli.py compare --summaries outputs/hoi4d_eval/summary.json outputs/other_eval/summary.json --out outputs/comparison
python egohand3d_cli.py package --run outputs/hoi4d_eval --out outputs/hoi4d_eval.zip
```

`compare` 要求 GT 文件、评估实现和匹配规则一致，拒绝混合口径。
`package` 打包本地生成物及逐文件 SHA-256；不加入模型权重、MANO 数据、SQLite 和缓存，不执行上传。

## 测试与来源

```bash
python -m pytest -q tests
```

测试覆盖旋转临界情况、左右手反射约定、漏检分母、框匹配、轨迹关联、文件损坏恢复、输入变化、
视频抽帧、并发锁和校验和归档。测试使用的合成数据只用于软件逻辑验证，不混入真实精度报告。
新增功能是应用工程实现，继续依赖 WiLoR/MANO；不声称新开发了核心重建网络。
旧应用脚本来源仍需核实，具体边界见 THIRD_PARTY_NOTICES.md。
