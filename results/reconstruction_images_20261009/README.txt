手部三维重建结果图片

打开 index.html 查看。
finetuned_sample_*：已有 HOI4D 第一视角微调模型预测。
initial_two_hands_*：已有初始 WiLoR 权重双手流程样例。
*_comparison.jpg：原图与网格叠加图的同区域放大对照。
*_mesh_overlay.jpg：原始图像尺寸的三维网格叠加图。
*_mesh_only.png：相同相机视角的白底三维手部图。
*_input.jpg：未经修改的输入图像。

本次没有重新推理，也未修改保存的三维顶点。仅改变网格显示颜色和光照。
蓝色为右手，橙色为左手。叠加图未处理场景物体遮挡。
样例按固定索引选取，遇到无检测记录顺延；未按误差挑选。
来源路径与校验值见 manifest.json。复现：egohand3d 环境下运行 python render_saved_meshes.py。
