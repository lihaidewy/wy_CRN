import os
import json
import numpy as np
import cv2
import mmcv

# ================= 路径配置 =================
DATA_ROOT = "./data/my_formatted_data"
IMG_DIR = os.path.join(DATA_ROOT, "samples/CAM_FRONT")
PKL_PATH = os.path.join(DATA_ROOT, "nuscenes_infos_val.pkl") 
# 请务必确保此路径指向你最新的 json 文件！
JSON_PATH = "./outputs/det/CRN_r50_256x704_128x128_4key/results_nusc.json"
OUTPUT_DIR = "./outputs/custom_vis_results" 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4K 相机内参
CAMERA_INTRINSIC = np.array([
    [3325.5375505322445, 0.0, 1920.0],
    [0.0, 3325.5375505322445, 1080.0],
    [0.0, 0.0, 1.0]
])
H = 6.0 

def project_to_image(x, y, z):
    # 物理坐标系转换
    cam_x = -y  
    cam_y = H   
    cam_z = x   
    
    if cam_z <= 0.5: return None
        
    u = (CAMERA_INTRINSIC[0,0] * cam_x / cam_z) + CAMERA_INTRINSIC[0,2]
    v = (CAMERA_INTRINSIC[1,1] * cam_y / cam_z) + CAMERA_INTRINSIC[1,2]
    return int(u), int(v)

def evaluate_and_visualize():
    print(f">>> 正在加载预测结果: {JSON_PATH}")
    with open(JSON_PATH, 'r') as f:
        json_data = json.load(f)
        pred_data = json_data.get('results', json_data)

    infos = mmcv.load(PKL_PATH)
    print(f">>> 成功加载 {len(infos)} 帧信息")

    for info in infos[:20]: # 先测试前20帧，防止跑太慢
        token = info['sample_token']
        frame_idx = token.split('_')[-1]
        img_path = os.path.join(IMG_DIR, f"{int(frame_idx)}.png")
        
        if not os.path.exists(img_path): continue
        img = cv2.imread(img_path)
        
        # 1. 画真实目标 (GT) - 绿圈
        for ann in info['ann_infos']:
            pos = project_to_image(*ann['translation'])
            if pos:
                cv2.circle(img, pos, 20, (0, 255, 0), 3) 

        # 2. 画预测目标 (Pred) - 红点
        raw_preds = pred_data.get(token, [])
        if len(raw_preds) > 0:
            print(f"帧 {frame_idx} 共有 {len(raw_preds)} 个预测，首个坐标: {raw_preds[0]['translation']}")
        
        # 暴力显示：画出所有预测，即使点在画面外(虽然看不见)，也要画在 img 变量里
        for pred in raw_preds:
            pos = project_to_image(*pred['translation'])
            if pos:
                # 画出所有点，半径设为 10
                cv2.circle(img, pos, 10, (0, 0, 255), -1) 
            else:
                print(f"帧 {frame_idx} 有预测点投影失败: {pred['translation']}")

        out_path = os.path.join(OUTPUT_DIR, f"vis_{frame_idx}.jpg")
        cv2.imwrite(out_path, img)
        print(f"已保存: {out_path}")

    print("✅ 可视化完成！请检查 output/custom_vis_results")

if __name__ == "__main__":
    evaluate_and_visualize()