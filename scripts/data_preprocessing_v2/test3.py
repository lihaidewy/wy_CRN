import os
import numpy as np
import mmcv

DATA_ROOT = "./data/my_formatted_data"
RADAR_BEV_DIR = os.path.join(DATA_ROOT, "radar_bev_filter")
DEPTH_GT_DIR = os.path.join(DATA_ROOT, "depth_gt")
PKL_PATH = os.path.join(DATA_ROOT, "nuscenes_infos_train.pkl")

def verify_data():
    print("==================================================")
    print("          🚀 训练前终极数据体检 (X-Ray Check)      ")
    print("==================================================\n")

    # ==========================================
    # 1. 检查雷达特征 (radar_bev_filter)
    # ==========================================
    print("🔎 [1/3] 检查雷达特征 (radar_bev_filter/*.bin)...")
    radar_files = [f for f in os.listdir(RADAR_BEV_DIR) if f.endswith('.bin')]
    if not radar_files:
        print("❌ 错误：雷达目录为空！请运行 generate_derivatives.py")
        return
    
    test_radar_file = os.path.join(RADAR_BEV_DIR, radar_files[0])
    radar_data = np.fromfile(test_radar_file, dtype=np.float32).reshape(-1, 7)
    
    x_min, x_max = radar_data[:, 0].min(), radar_data[:, 0].max()
    y_min, y_max = radar_data[:, 1].min(), radar_data[:, 1].max()
    
    # 规则：X 是深度 (应为正数)，Y 是横向 (应在 0 左右正负分布)
    x_pass = (x_min >= -10.0) and (x_max <= 200.0)  # 允许极小误差
    y_pass = (y_min >= -100.0) and (y_max <= 100.0) and (y_min < 0) and (y_max > 0)
    
    print(f"   ▶ 测试文件: {radar_files[0]} (点数: {len(radar_data)})")
    print(f"   ▶ X轴 (深度) 范围: [{x_min:.2f}, {x_max:.2f}] 米")
    print(f"   ▶ Y轴 (横向) 范围: [{y_min:.2f}, {y_max:.2f}] 米")
    
    if x_pass and y_pass:
        print("   ✅ 状态：完美！X/Y 坐标轴映射正确。")
    else:
        print("   ❌ 状态：异常！X 或 Y 的范围严重偏离路侧场景，请检查！")

    # ==========================================
    # 2. 检查深度真值 (depth_gt)
    # ==========================================
    print("\n🔎 [2/3] 检查深度真值 (depth_gt/*.png.bin)...")
    depth_files = [f for f in os.listdir(DEPTH_GT_DIR) if f.endswith('.bin')]
    if not depth_files:
        print("❌ 错误：深度目录为空！请运行 build_real_depth_gt.py")
        return
    
    test_depth_file = os.path.join(DEPTH_GT_DIR, depth_files[0])
    depth_data = np.fromfile(test_depth_file, dtype=np.float32).reshape(-1, 3)
    
    u_min, u_max = depth_data[:, 0].min(), depth_data[:, 0].max()
    v_min, v_max = depth_data[:, 1].min(), depth_data[:, 1].max()
    z_min, z_max = depth_data[:, 2].min(), depth_data[:, 2].max()
    
    u_pass = (u_min >= 0) and (u_max <= 3840)
    v_pass = (v_min >= 0) and (v_max <= 2160)
    z_pass = z_min > 0
    
    print(f"   ▶ 测试文件: {depth_files[0]} (点数: {len(depth_data)})")
    print(f"   ▶ U (像素X): [{u_min:.1f}, {u_max:.1f}] (应在 0~3840)")
    print(f"   ▶ V (像素Y): [{v_min:.1f}, {v_max:.1f}] (应在 0~2160)")
    print(f"   ▶ Z (物理深度): [{z_min:.2f}, {z_max:.2f}] 米 (必须 > 0)")
    
    if u_pass and v_pass and z_pass:
        print("   ✅ 状态：完美！透视投影无越界，深度值为正。")
    else:
        print("   ❌ 状态：异常！存在越界像素或负数深度，请检查！")

    # ==========================================
    # 3. 检查标注文件 (PKL)
    # ==========================================
    print("\n🔎 [3/3] 检查训练标注文件 (PKL)...")
    infos = mmcv.load(PKL_PATH)
    gt_checked = False
    
    for info in infos:
        if len(info['ann_infos']) > 0:
            ann = info['ann_infos'][0]
            gt_x, gt_y, gt_z = ann['translation']
            
            print(f"   ▶ 测试目标类别: {ann['category_name']}")
            print(f"   ▶ GT 中心坐标: X(深度)={gt_x:.2f}, Y(横向)={gt_y:.2f}, Z(高度)={gt_z:.2f}")
            
            if gt_x > 0 and abs(gt_y) < 100:
                print("   ✅ 状态：完美！Ground Truth 坐标轴系正确。")
            else:
                print("   ❌ 状态：异常！GT 坐标似乎依然是反转的！")
            gt_checked = True
            break
            
    if not gt_checked:
        print("   ⚠️ 警告：前几帧没有发现任何 Ground Truth 目标，请确认数据是否过少。")

    print("\n==================================================")
    print("如果在上述检查中看到了三个 ✅，你的数据已经无懈可击！")
    print("==================================================")

if __name__ == "__main__":
    verify_data()