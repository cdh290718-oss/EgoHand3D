from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import json
from typing import Dict, Optional

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from wilor.utils.yolo_loader import load_yolo_detector
LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)  # 渲染时使用的网格颜色

def main():
    parser = argparse.ArgumentParser(description='WiLoR demo code')
    # 输入/输出与运行参数
    parser.add_argument('--img_folder', type=str, default='./', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=True, help='If set, save meshes to disk also')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'], help='List of file extensions to consider')
    parser.add_argument('--checkpoint', type=str, default='./pretrained_models/wilor_final.ckpt', help='WiLoR checkpoint path')
    parser.add_argument('--cfg', type=str, default='./pretrained_models/model_config.yaml', help='WiLoR model config path')
    parser.add_argument('--detector', type=str, default='./pretrained_models/detector.pt', help='YOLO hand detector path')

    args = parser.parse_args()

    # 加载模型与检测器权重
    model, model_cfg = load_wilor(checkpoint_path=args.checkpoint, cfg_path=args.cfg)
    detector = load_yolo_detector(args.detector)
    # 初始化渲染器（renderer 用于正视图，renderer_side 目前未使用）
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    renderer_side = Renderer(model_cfg, faces=model.mano.faces)
    
    # 选择设备并将模型移动到对应设备
    device   = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model    = model.to(device)
    detector = detector.to(device)
    model.eval()

    # 输出目录不存在则创建
    os.makedirs(args.out_folder, exist_ok=True)

    # 按扩展名收集输入图片
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    # 遍历文件夹内所有图片
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        # 运行手部检测器（YOLO）获取候选框
        detections = detector(img_cv2, conf = 0.3, verbose=False)[0]
        bboxes    = []
        is_right  = []
        for det in detections: 
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())
        
        # 没有检测到手则跳过
        if len(bboxes) == 0:
            continue
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        # 构建 WiLoR 所需的裁剪数据集
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        # 收集每只手的输出用于后续渲染
        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints= []
        all_kpts  = []
        
        for batch in dataloader: 
            batch = recursive_to(batch, device)
    
            with torch.no_grad():
                out = model(batch) 
                
            # 根据左右手翻转相机的 x 方向
            multiplier    = (2*batch['right']-1)
            pred_cam      = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center    = batch["box_center"].float()
            box_size      = batch["box_size"].float()
            img_size      = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            # 将相机平移恢复到整图坐标系
            pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            
            # 单手后处理与可选的网格导出
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # 从路径中获取文件名
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                
                verts  = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                
                is_right    = batch['right'][n].cpu().numpy()
                # 对左手进行 x 方向镜像以保持一致坐标系
                verts[:,0]  = (2*is_right-1)*verts[:,0]
                joints[:,0] = (2*is_right-1)*joints[:,0]
                cam_t = pred_cam_t_full[n]
                # 将顶点投影到 2D 用于可视化或下游使用
                kpts_2d = project_full_img(verts, cam_t, scaled_focal_length, img_size[n])
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_joints.append(joints)
                all_kpts.append(kpts_2d)
                
                
                # 如需保存，则将网格写入磁盘
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_PURPLE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{n}.obj'))

        # 渲染并叠加到原图
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # 叠加渲染结果
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # 添加 alpha 通道
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}.jpg'), 255*input_img_overlay[:, :, ::-1])

def project_full_img(points, cam_trans, focal_length, img_res): 
    """将 3D 点投影到整图像素坐标系。"""
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3) 
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:] 
    
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]

if __name__ == '__main__':
    main()
