import pickle
import cv2
import numpy as np
import os
from pyquaternion import Quaternion

# ==========================================
# 配置路径 (请确保和你的实际路径一致)
# ==========================================
PKL_PATH = "./data/my_formatted_data/nuscenes_infos_train.pkl"
DATA_ROOT = "./data/my_formatted_data"

def get_3d_box_corners(size, translation, rotation_quat):
    """根据长宽高、中心点和四元数，生成 3D 框的 8 个顶点"""
    l, w, h = size
    # 定义 8 个顶点 (基于中心点为原点)
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2] 
    corners = np.vstack([x_corners, y_corners, z_corners])

    # 旋转
    rot_matrix = Quaternion(rotation_quat).rotation_matrix
    corners = np.dot(rot_matrix, corners)

    # 平移
    corners[0, :] += translation[0]
    corners[1, :] += translation[1]
    corners[2, :] += translation[2]
    
    return corners # (3, 8)

def verify_step4():
    print(">>> 启动第四关 PKL 终极质检...\n")
    
    if not os.path.exists(PKL_PATH):
        print(f"❌ 找不到 PKL 文件: {PKL_PATH}")
        return

    with open(PKL_PATH, 'rb') as f:
        infos = pickle.load(f)
    
    print(f"✅ 成功读取 PKL，共包含 {len(infos)} 帧数据。")
    
    # 抽取第一帧进行终极画图验证
    info = infos[0]
    cam_info = info['cam_infos']['CAM_FRONT']
    img_path = os.path.join(DATA_ROOT, cam_info['filename'])
    
    if not os.path.exists(img_path):
        print(f"❌ 找不到图片文件: {img_path}")
        return

    img = cv2.imread(img_path)
    
    # 1. 提取相机内参 K (3x3)
    K = np.array(cam_info['calibrated_sensor']['camera_intrinsic'])
    
    # 2. 提取并构建外参：从 相机 到 世界 (Cam2World)
    quat = Quaternion(cam_info['calibrated_sensor']['rotation'])
    trans = np.array(cam_info['calibrated_sensor']['translation'])
    
    # 打印出来查验我们当时强制注入的轴向四元数
    print("\n--- 🔍 查验外参矩阵注入情况 ---")
    print(f"平移 (Translation): {trans} (Z 轴应该是 +6.0 左右)")
    print(f"旋转 (Quaternion): {quat.elements} (应该是我们注入的 [0.5, -0.5, 0.5, -0.5])")

    # 构建完整的 Cam2World 4x4 矩阵
    cam2world = np.eye(4)
    cam2world[:3, :3] = quat.rotation_matrix
    cam2world[:3, 3] = trans
    
    # 求逆矩阵：得到 世界 到 相机 (World2Cam) 的变换矩阵，用于投影
    world2cam = np.linalg.inv(cam2world)

    print(f"\n--- 📸 正在绘制第一帧的 {len(info['ann_infos'])} 个 3D 标注框 ---")
    
    for ann in info['ann_infos']:
        # 获取 8 个 3D 顶点 (世界坐标系)
        corners_world = get_3d_box_corners(ann['size'], ann['translation'], ann['rotation'])
        
        # 变齐次坐标 (4, 8)
        corners_world_homo = np.vstack([corners_world, np.ones((1, 8))])
        
        # 转换到相机坐标系
        corners_cam = world2cam @ corners_world_homo
        
        # 过滤跑到相机背面的点 (Z <= 0.1)
        if np.any(corners_cam[2, :] <= 0.1):
            continue
            
        # 投影到像素平面
        corners_2d = K @ corners_cam[:3, :]
        corners_2d = corners_2d[:2, :] / corners_2d[2, :]
        corners_2d = corners_2d.astype(int).T # (8, 2)
        
        # 画 3D 框连线 (底面、顶面、立柱)
        lines = [[0,1], [1,2], [2,3], [3,0], 
                 [4,5], [5,6], [6,7], [7,4], 
                 [0,4], [1,5], [2,6], [3,7]]
        for line in lines:
            pt1, pt2 = tuple(corners_2d[line[0]]), tuple(corners_2d[line[1]])
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            
        # 标个名字
        cv2.putText(img, ann['category_name'], tuple(corners_2d[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    save_path = "check_step4_final_3dbox.jpg"
    cv2.imwrite(save_path, img)
    print(f"\n🎉 [PKL 终极验收成功] 请查看生成的图片: {save_path}")

if __name__ == "__main__":
    verify_step4()