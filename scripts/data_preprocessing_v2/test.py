import mmcv
import json
import os
import numpy as np

def verify_pkl_vs_json():
    PKL_PATH = "./data/my_formatted_data/nuscenes_infos_train.pkl"
    JSON_DIR = "./data/my_formatted_data/jsons"
    
    infos = mmcv.load(PKL_PATH)
    print(f"--- 正在检查 PKL，共包含 {len(infos)} 帧 ---")
    
    # 抽取前 5 帧进行比对
    for i in range(min(5, len(infos))):
        info = infos[i]
        sample_token = info['sample_token']
        frame_idx = int(sample_token.split('_')[-1])
        
        json_path = os.path.join(JSON_DIR, f"{frame_idx}.json")
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
        print(f"\n>>> 验证第 {i+1} 帧 (frame_{frame_idx})")
        # 简单比对第一个物体
        pkl_obj = info['ann_infos'][0]
        json_obj = raw_data['objects'][0]
        
        print(f"  [PKL] translation: {pkl_obj['translation']}")
        print(f"  [JSON] translation: {json_obj['translation']}")
        
        # 误差校验
        diff = np.abs(np.array(pkl_obj['translation']) - np.array(json_obj['translation']))
        if np.all(diff < 1e-4):
            print("  ✅ 坐标对齐成功！")
        else:
            print(f"  ❌ 坐标存在偏差: {diff}")

if __name__ == "__main__":
    verify_pkl_vs_json()