import pickle
import cv2
import numpy as np
import os
from pyquaternion import Quaternion

# ==========================================
# 1. 路径与配置
# ==========================================
PKL_PATH = "./data/my_formatted_data/nuscenes_infos_train.pkl"
DATA_ROOT = "./data/my_formatted_data"
OUTPUT_DIR = "./work_dirs/batch_vis_results"  # 批量生成的图片存放在这里

# 你想连续抽查多少帧？(建议先抽 20 帧看看)
NUM_FRAMES_TO_CHECK = 20  

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_3d_box_corners(size, translation, rotation_quat):
    """生成 3D 框的 8 个顶点"""
    l, w, h = size
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2] 
    # z_corners = [0, 0, 0, 0, h, h, h, h] 
    corners = np.vstack([x_corners, y_corners, z_corners])

    rot_matrix = Quaternion(rotation_quat).rotation_matrix
    corners = np.dot(rot_matrix, corners)
    corners[0, :] += translation[0]
    corners[1, :] += translation[1]
    corners[2, :] += translation[2]
    
    return corners

def verify_batch_frames():
    print(f">>> 启动批量 PKL 质检，准备绘制前 {NUM_FRAMES_TO_CHECK} 帧...\n")
    
    if not os.path.exists(PKL_PATH):
        print(f"❌ 找不到 PKL 文件: {PKL_PATH}")
        return

    with open(PKL_PATH, 'rb') as f:
        infos = pickle.load(f)
    
    total_frames = len(infos)
    print(f"✅ 成功读取 PKL，共包含 {total_frames} 帧数据。")
    
    check_count = min(NUM_FRAMES_TO_CHECK, total_frames)
    
    for i in range(check_count):
        info = infos[i]
        cam_info = info['cam_infos']['CAM_FRONT']
        img_path = os.path.join(DATA_ROOT, cam_info['filename'])
        
        if not os.path.exists(img_path):
            print(f"⚠️ 找不到图片 {img_path}，跳过第 {i} 帧。")
            continue

        img = cv2.imread(img_path)
        
        # 提取内外参
        K = np.array(cam_info['calibrated_sensor']['camera_intrinsic'])
        quat = Quaternion(cam_info['calibrated_sensor']['rotation'])
        trans = np.array(cam_info['calibrated_sensor']['translation'])
        
        cam2world = np.eye(4)
        cam2world[:3, :3] = quat.rotation_matrix
        cam2world[:3, 3] = trans
        world2cam = np.linalg.inv(cam2world)

        valid_box_count = 0
        for ann in info['ann_infos']:
            corners_world = get_3d_box_corners(ann['size'], ann['translation'], ann['rotation'])
            corners_world_homo = np.vstack([corners_world, np.ones((1, 8))])
            corners_cam = world2cam @ corners_world_homo
            
            if np.any(corners_cam[2, :] <= 0.1):
                continue
                
            corners_2d = K @ corners_cam[:3, :]
            corners_2d = corners_2d[:2, :] / corners_2d[2, :]
            corners_2d = corners_2d.astype(int).T
            
            lines = [[0,1], [1,2], [2,3], [3,0], 
                     [4,5], [5,6], [6,7], [7,4], 
                     [0,4], [1,5], [2,6], [3,7]]
            for line in lines:
                pt1, pt2 = tuple(corners_2d[line[0]]), tuple(corners_2d[line[1]])
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)
                
            cv2.putText(img, ann['category_name'], tuple(corners_2d[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            valid_box_count += 1

        # 保存图片
        frame_id_str = str(cam_info['filename']).split('/')[-1]
        save_path = os.path.join(OUTPUT_DIR, f"vis_{frame_id_str}")
        cv2.imwrite(save_path, img)
        print(f"📸 进度 [{i+1}/{check_count}]: 绘制了 {valid_box_count} 个框 -> {save_path}")

    print(f"\n🎉 批量绘制完成！请打开 {OUTPUT_DIR} 文件夹，像看幻灯片一样切换图片进行检查吧！")

if __name__ == "__main__":
    verify_batch_frames()