import os
import torch
import cv2
import numpy as np
import functools
from pyquaternion import Quaternion
from torch.utils.data import DataLoader

# ==============================================================================
# 配置区
# ==============================================================================
CKPT_PATH = "./outputs/det/CRN_r50_256x704_128x128_4key/lightning_logs/version_68/checkpoints/epoch=23-step=4320.ckpt"
OUTPUT_DIR = "./work_dirs/debug_gt_vs_pred"
SCORE_THR = 0.35 # 只显示置信度高的预测框

os.makedirs(OUTPUT_DIR, exist_ok=True)

from exps.det.CRN_r50_256x704_128x128_4key import CRNLightningModel
from datasets.nusc_det_dataset import collate_fn
from mmdet3d.core import LiDARInstance3DBoxes

# ------------------------------------------------------------------------------
# 🚨 核心函数：使用你最新修正的 [W, L, H] 顺序和 Yaw 旋转逻辑生成顶点
# ------------------------------------------------------------------------------
def get_3d_box_corners(size, translation, yaw_angle):
    w, l, h = size # [W, L, H]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
    corners = np.vstack([x_corners, y_corners, z_corners])
    
    rot_quat = [float(np.cos(yaw_angle / 2)), 0.0, 0.0, float(np.sin(yaw_angle / 2))]
    rot_matrix = Quaternion(rot_quat).rotation_matrix
    corners = np.dot(rot_matrix, corners)
    
    corners[0, :] += translation[0]
    corners[1, :] += translation[1]
    corners[2, :] += translation[2]
    return corners

def draw_box(img, corners_2d, color):
    """辅助函数：画 3D 框的线"""
    # 底盘
    for line in [[0,1], [1,2], [2,3], [3,0]]:
        cv2.line(img, tuple(corners_2d[line[0]]), tuple(corners_2d[line[1]]), color, 2)
    # 立柱和顶棚
    for line in [[4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]:
        cv2.line(img, tuple(corners_2d[line[0]]), tuple(corners_2d[line[1]]), color, 1)

def run_comparison():
    print("\n" + "="*70)
    print("🚀 启动真值与预测值同台对质脚本 (第0帧)...")
    print("="*70 + "\n")

    model = CRNLightningModel()
    checkpoint = torch.load(CKPT_PATH, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda().eval()

    # 1. 获取干净的数据集 (is_train=False)
    original_loader = model.train_dataloader()
    dataset = original_loader.dataset
    dataset.is_train = False

    custom_collate = functools.partial(
        collate_fn, is_return_image=True, 
        is_return_depth=model.return_depth, is_return_radar_pv=model.return_radar_pv
    )
    seq_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    # 2. 只拿第 0 帧进行极限剖析
    batch_idx = 0
    batch = next(iter(seq_loader))
    (sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, depth_labels, pts_pv) = batch

    sweep_imgs_cuda = sweep_imgs.cuda()
    pts_pv_cuda = pts_pv.cuda()
    mats_cuda = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in mats.items()}

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # 前向传播 is_train=True (模型内部逻辑)
            outputs = model.model(sweep_imgs_cuda, mats_cuda, sweep_ptss=pts_pv_cuda, is_train=True)
            preds = outputs[0]
        img_meta = dataset.infos[0].copy()
        img_meta['box_type_3d'] = LiDARInstance3DBoxes
        img_metas = [img_meta]
        bbox_results = model.model.get_bboxes(preds, img_metas)

    # 3. 提取预测和真值
    pred_boxes = bbox_results[0][0].tensor.cpu().numpy()
    pred_scores = bbox_results[0][1].cpu().numpy()
    gt_boxes_data = gt_boxes_3d[0].cpu().numpy() # 这里的 GT 也是 [x, y, z, l, w, h, yaw]

    # 4. 准备画图
    info = dataset.infos[batch_idx]
    cam_info = info['cam_infos']['CAM_FRONT']
    img_path = os.path.join("./data/my_formatted_data", cam_info['filename'])
    img = cv2.imread(img_path)
    K = np.array(cam_info['calibrated_sensor']['camera_intrinsic'])
    
    # 建立标准的 Ego 到 Camera 投影矩阵
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(cam_info['calibrated_sensor']['rotation']).rotation_matrix
    cam2ego[:3, 3] = np.array(cam_info['calibrated_sensor']['translation'])
    ego2cam = np.linalg.inv(cam2ego)

    print("📊 [数值对齐对质面板]：")
    print(f"--- 🟢 这一帧里共有 {len(gt_boxes_data)} 个【真值 GT 框】 (青色) ---")
    for idx, box in enumerate(gt_boxes_data):
        # GT 的 size 顺序在 MMDetection3D 内部已经被标准化为 [L, W, H]
        # 但为了和你修正后的函数兼容，这里我们传入时依然当做 [W, L, H] 来传，看画出来对不对
        size = [box[3], box[4], box[5]] 
        translation = [box[0], box[1], box[2]]
        yaw = box[6]
        print(f" GT  #{idx}: xyz=[{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}] | size=[{box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f}] | yaw={box[6]:.2f}")
        
        corners_ego = get_3d_box_corners(size, translation, yaw)
        corners_cam = ego2cam @ np.vstack([corners_ego, np.ones((1, 8))])
        if np.any(corners_cam[2, :] <= 0.1): continue
        c_2d = K @ corners_cam[:3, :]
        c_2d = (c_2d[:2, :] / c_2d[2, :]).astype(int).T
        draw_box(img, c_2d, (255, 255, 0)) # 青色

    print(f"\n--- 🔴 过滤后模型吐出的【预测 PRED 框】 (红色) ---")
    p_idx = 0
    for i in range(len(pred_boxes)):
        if pred_scores[i] < SCORE_THR: continue
        box = pred_boxes[i]
        # 预测框的 size 顺序明确是 [W, L, H]
        size = [box[3], box[4], box[5]]
        translation = [box[0], box[1], box[2]]
        yaw = box[6]
        print(f" Pred #{p_idx}: xyz=[{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}] | size=[{box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f}] | yaw={box[6]:.2f} | Score={pred_scores[i]:.2f}")
        
        corners_ego = get_3d_box_corners(size, translation, yaw)
        corners_cam = ego2cam @ np.vstack([corners_ego, np.ones((1, 8))])
        if np.any(corners_cam[2, :] <= 0.1): continue
        c_2d = K @ corners_cam[:3, :]
        c_2d = (c_2d[:2, :] / c_2d[2, :]).astype(int).T
        draw_box(img, c_2d, (0, 0, 255)) # 红色
        p_idx += 1

    save_path = os.path.join(OUTPUT_DIR, "gt_vs_pred_final_contrast.jpg")
    cv2.imwrite(save_path, img)
    print("\n" + "="*70)
    print(f"📸 终极质检图已生成 ➡️ {save_path}")
    print("请查看终端里的 [数值对齐对质面板]，并把打印结果和生成的图片发出来！")
    print("="*70)

if __name__ == "__main__":
    run_comparison()