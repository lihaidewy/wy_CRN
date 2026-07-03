import os
import torch
import cv2
import numpy as np
import functools
from pyquaternion import Quaternion
from torch.utils.data import DataLoader

# ==============================================================================
# 1. 基础配置
# ==============================================================================
CKPT_PATH = "outputs/det/CRN_r50_256x704_128x128_4key/lightning_logs/version_74/checkpoints/epoch=23-step=3456.ckpt"
OUTPUT_DIR = "./work_dirs/model_predict_results3"
SCORE_THR = 0.0        
NUM_FRAMES_TO_VIS = 180  

os.makedirs(OUTPUT_DIR, exist_ok=True)

from exps.det.CRN_r50_256x704_128x128_4key import CRNLightningModel
from datasets.nusc_det_dataset import collate_fn  
from mmdet3d.core import LiDARInstance3DBoxes

# ==============================================================================
# 2. 核心三维几何函数 (严格解耦 GT 和 Pred，采用各自的物理基准)
# ==============================================================================

def get_gt_box_corners(size, translation, rotation_quat):
    """(100% 正确) 用于青色真值框的生成，直接读取自标注文件"""
    l, w, h = size
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2] 
    corners = np.vstack([x_corners, y_corners, z_corners])

    rot_matrix = Quaternion(rotation_quat).rotation_matrix
    corners = np.dot(rot_matrix, corners)
    corners[0, :] += translation[0]
    corners[1, :] += translation[1]
    corners[2, :] += translation[2]
    return corners

def get_pred_box_corners(box):
    """
     终极修复：同步 MMDet3D 官方 LiDARInstance3DBoxes 底层标定标准
    """
    x, y, z, w, l, h, yaw = box[:7]
    
    # 1. 核心修复一：z 是底面中心，z_corners 必须从 0 到 h（彻底解决车辆陷入地里的问题）
    # 2. 核心修复二：轴向对齐，MMDet3D 标准中 w 沿 local X 轴，l 沿 local Y 轴
    x_corners = [-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]
    y_corners = [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2]
    z_corners = [0, 0, 0, 0, h, h, h, h]
    corners = np.vstack([x_corners, y_corners, z_corners])
    
    # 3. 核心修复三：航向角校准。模型预测的 yaw 是在雷达系下的，需施加 -pi/2 的相位补偿转正
    yaw_calibrated = yaw - (np.pi / 2.0)
    
    rot_mat = np.array([
        [np.cos(yaw_calibrated), -np.sin(yaw_calibrated), 0],
        [np.sin(yaw_calibrated),  np.cos(yaw_calibrated), 0],
        [0,                      0,                      1]
    ])
    corners = np.dot(rot_mat, corners)
    
    # 4. 平移绝对物理中心
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    return corners

def draw_3d_box(img, corners_2d, color):
    """安全版绘图函数，过滤齐次坐标，杜绝 OpenCV Bad Argument 报错"""
    pts = corners_2d[:, :2].astype(int)
    lines = [[0,1], [1,2], [2,3], [3,0], 
             [4,5], [5,6], [6,7], [7,4], 
             [0,4], [1,5], [2,6], [3,7]]
    for line in lines:
        pt1 = tuple(pts[line[0]])
        pt2 = tuple(pts[line[1]])
        cv2.line(img, pt1, pt2, color, 2)

# ==============================================================================
# 3. 推理主循环 (含详细数值对质打印)
# ==============================================================================
def run_pure_inference():
    print(">>> 正在初始化模型并加载权重...")
    model = CRNLightningModel()
    checkpoint = torch.load(CKPT_PATH, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda().eval()  
    
    # dataset = model.train_dataloader().dataset
    dataset = model.val_dataloader().dataset
    dataset.is_train = False 
    
    custom_collate = functools.partial(
        collate_fn, 
        is_return_image=True, 
        is_return_depth=model.return_depth, 
        is_return_radar_pv=model.return_radar_pv
    )
    seq_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    
    print("✅ 空间基准重构对齐就绪，启动核心解算...\n")
    
    for batch_idx, batch in enumerate(seq_loader):
        if batch_idx >= NUM_FRAMES_TO_VIS: break
            
        (sweep_imgs, mats, _, _, _, _, _, pts_pv) = batch
        
        sweep_imgs_cuda = sweep_imgs.cuda().half()
        pts_pv_cuda = pts_pv.cuda().float()
        mats_cuda = {k: (v.cuda().float() if isinstance(v, torch.Tensor) else v) for k, v in mats.items()}
                
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model.model(sweep_imgs_cuda, mats_cuda, sweep_ptss=pts_pv_cuda, is_train=True)
                preds = outputs[0]
                
            img_meta = dataset.infos[batch_idx].copy()
            img_meta['box_type_3d'] = LiDARInstance3DBoxes
            bbox_results = model.model.get_bboxes(preds, [img_meta])
            
        pred_boxes3d = bbox_results[0][0].tensor.cpu().numpy()  
        pred_scores = bbox_results[0][1].cpu().numpy()
        
        # =========== 外参矩阵提取 (完美的绝对物理外参) ===========
        info = dataset.infos[batch_idx]
        cam_info = info['cam_infos']['CAM_FRONT']
        img = cv2.imread(os.path.join("./data/my_formatted_data", cam_info['filename']))
        K = np.array(cam_info['calibrated_sensor']['camera_intrinsic'])
        
        quat = Quaternion(cam_info['calibrated_sensor']['rotation'])
        trans = np.array(cam_info['calibrated_sensor']['translation'])
        cam2world = np.eye(4)
        cam2world[:3, :3] = quat.rotation_matrix
        cam2world[:3, 3] = trans
        ego2cam = np.linalg.inv(cam2world)  
        
        print(f"\n====================== 🛰️ Frame #{batch_idx} 数值对质面板 ======================")

        # ----------------  绘制真值 (青色) ----------------
        print(f"🟢 [真值 GT 列表] - 共有 {len(info['ann_infos'])} 个标注物体:")
        for g_idx, ann in enumerate(info['ann_infos']):
            size = ann['size']       
            trans_gt = ann['translation'] 
            
            print(f"  GT  #{g_idx:<2} | 类别: {ann['category_name'][:10]:<10} | "
                  f"xyz=[{trans_gt[0]:7.2f}, {trans_gt[1]:7.2f}, {trans_gt[2]:7.2f}] | "
                  f"size=[{size[0]:5.2f}, {size[1]:5.2f}, {size[2]:5.2f}]")
            
            corners_world = get_gt_box_corners(ann['size'], ann['translation'], ann['rotation'])
            corners_world_homo = np.vstack([corners_world, np.ones((1, 8))])
            corners_cam = ego2cam @ corners_world_homo
            if np.any(corners_cam[2, :] <= 0.1): continue
            
            corners_2d = (K @ corners_cam[:3, :] / corners_cam[2, :]).T
            draw_3d_box(img, corners_2d, (255, 255, 0)) 

        # ----------------  绘制预测 (红色) ----------------
        print(f"\n🔴 [模型预测 PRED 列表] (得分 >= {SCORE_THR}):")
        p_idx = 0
        for i in range(len(pred_boxes3d)):
            if pred_scores[i] < SCORE_THR: continue
            box = pred_boxes3d[i]
            
            x, y, z, w, l, h, yaw = box[:7]
            print(f"  Pred #{p_idx:<2} | 得分: {pred_scores[i]:6.2%}     | "
                  f"xyz=[{x:7.2f}, {y:7.2f}, {z:7.2f}] | "
                  f"size=[{w:5.2f}, {l:5.2f}, {h:5.2f}] | yaw={yaw:6.2f}")
            p_idx += 1
            
            # 调用全新对齐的顶点解算
            corners_world = get_pred_box_corners(box)
            
            corners_world_homo = np.vstack([corners_world, np.ones((1, 8))])
            corners_cam = ego2cam @ corners_world_homo
            if np.any(corners_cam[2, :] <= 0.1): continue
            
            corners_2d = (K @ corners_cam[:3, :] / corners_cam[2, :]).T
            draw_3d_box(img, corners_2d, (0, 0, 255)) 
            cv2.putText(img, f"P{p_idx-1}", tuple(corners_2d[0][:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        save_path = os.path.join(OUTPUT_DIR, f"predict_frame_{batch_idx:03d}.jpg")
        cv2.imwrite(save_path, img)
        print(f"\n📸 结果已生成 -> {save_path}\n" + "="*80)

if __name__ == "__main__":
    run_pure_inference()