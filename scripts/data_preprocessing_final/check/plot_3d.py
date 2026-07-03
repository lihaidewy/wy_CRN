import json
import numpy as np
import cv2

def get_3d_box_corners(size, translation, rotation_cam):
    l, w, h = size
    
    # x: 前/后, y: 左/右
    x_corners = [l/2, l/2, l/2, l/2, -l/2, -l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, w/2, -w/2, w/2, -w/2, w/2, -w/2]
    
    # --- 修改这里 ---
    # 如果标注点在地面，Z应该从 0 延伸到 h
    # 如果标注点在中心，Z才是从 -h/2 到 h/2
    # 针对你的图，改为 [0, 0, h, h, 0, 0, h, h] 这种逻辑（取决于顶点顺序）
    # 或者简单点，直接给整个 z_corners 加上 h/2 的偏移
    z_corners = [h, h, 0, 0, h, h, 0, 0] 
    
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    
    # 后面旋转和平移的代码保持不变...
    yaw = rotation_cam
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    corners_3d = np.dot(R, corners_3d)
    corners_3d[0, :] += translation[0]
    corners_3d[1, :] += translation[1]
    corners_3d[2, :] += translation[2]
    
    return corners_3d

def project_to_image_custom(corners_3d, intrinsics):
    """
    修正版：适配 X-前(深度), Y-左/右, Z-上 的投影逻辑
    """
    x = corners_3d[0, :] # 深度 (X轴朝前)
    y = corners_3d[1, :] # 左右 (Y轴)
    z = corners_3d[2, :] # 上下 (Z轴)
    
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # --- 核心修改处 ---
    # 如果之前是 cx - (...) 导致左右反了，这里改为 cx + (...)
    # 反之亦然。根据你的描述，将符号翻转：
    u = cx - (y / x * fx) 
    
    # v 方向：z朝上，图像下为正，所以用 cy - (z/x * fy) 是正确的（z越大，像素v越小，点越高）
    v = cy - (z / x * fy)
    
    return np.vstack([u, v]).T.astype(int)
    

def draw_3d_box(img, corners_2d, color=(0, 255, 0)):
    # 连线定义
    lines = [
        (0,1), (1,3), (3,2), (2,0), # 前面
        (4,5), (5,7), (7,6), (6,4), # 后面
        (0,4), (1,5), (2,6), (3,7)  # 四条侧棱
    ]
    for start, end in lines:
        cv2.line(img, tuple(corners_2d[start]), tuple(corners_2d[end]), color, 3)
    return img

# --- 执行 ---
with open('data/my_formatted_data/jsons/1.json', 'r') as f:
    data = json.load(f)

# 创建画布 (对应 cx, cy 所在的 3840x2160 分辨率)
img = cv2.imread("data/my_formatted_data/samples/CAM_FRONT/1.png")

for obj in data['objects']:
    # 获取相机坐标系下的 3D 顶点
    corners_3d = get_3d_box_corners(obj['size'], obj['translation_cam'], obj['rotation_cam'])
    
    # 只有在相机前方(x > 0)才投影
    if np.all(corners_3d[0, :] > 0):
        corners_2d = project_to_image_custom(corners_3d, data['intrinsics'])
        img = draw_3d_box(img, corners_2d)

cv2.imwrite('result.jpg', img)
print("可视化完成，请查看 result.jpg")