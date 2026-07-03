import numpy as np
import os

# ==========================================
# 请替换为你刚刚测试过的同一帧的索引号
# ==========================================
frame_idx = 1
bev_file_path = f"./data/my_formatted_data/radar_bev_filter/radar_{frame_idx:06d}.pcd.bin" 
pv_file_path = f"./data/my_formatted_data/radar_pv_filter/{frame_idx}.png.bin" 

def verify_step2():
    print(">>> 启动第二关双流特征质检...\n")
    
    # ---------------------------------------------------------
    # 查验 1：BEV 特征 (鸟瞰视角)
    # ---------------------------------------------------------
    if not os.path.exists(bev_file_path):
        print(f"❌ 找不到 BEV 文件: {bev_file_path}")
    else:
        bev_points = np.fromfile(bev_file_path, dtype=np.float32).reshape(-1, 7)
        print(f"✅ [BEV 特征] 读取成功，维度: {bev_points.shape} (应为 N x 7)")
        if len(bev_points) > 0:
            print(f"   - 深度(X) 范围: [{bev_points[:, 0].min():.2f}m, {bev_points[:, 0].max():.2f}m] (应与第一关一致)")
            print(f"   - 横向(Y) 范围: [{bev_points[:, 1].min():.2f}m, {bev_points[:, 1].max():.2f}m] (应与第一关一致)")
            print(f"   - 高度(Z) 范围: [{bev_points[:, 2].min():.2f}m, {bev_points[:, 2].max():.2f}m] (应全为0)")

    print("\n" + "-"*50 + "\n")

    # ---------------------------------------------------------
    # 查验 2：PV 特征 (图像透视视角)
    # ---------------------------------------------------------
    if not os.path.exists(pv_file_path):
        print(f"❌ 找不到 PV 文件: {pv_file_path}")
    else:
        pv_points = np.fromfile(pv_file_path, dtype=np.float32).reshape(-1, 7)
        print(f"✅ [PV 特征] 读取成功，维度: {pv_points.shape} (应为 N x 7)")
        if len(pv_points) > 0:
            u_coords = pv_points[:, 0]
            v_coords = pv_points[:, 1]
            depths = pv_points[:, 2]
            
            # 过滤掉无效点 (-1) 来统计图像内的坐标
            valid_mask = (u_coords > 0) & (v_coords > 0)
            valid_u = u_coords[valid_mask]
            valid_v = v_coords[valid_mask]
            
            print(f"   - 投影深度(Z) 范围: [{depths.min():.2f}m, {depths.max():.2f}m]")
            if len(valid_u) > 0:
                print(f"   - 有效像素 U (横向): [{valid_u.min():.0f}, {valid_u.max():.0f}] (正常应在 0~3840 之间)")
                print(f"   - 有效像素 V (纵向): [{valid_v.min():.0f}, {valid_v.max():.0f}] (正常应在 0~2160 之间)")
            else:
                print("   ⚠️ 警告: 所有点都被投影到了画面外或被标记为 -1")
                
            print("\n--- 🔍 抽取前 3 个有效 PV 点查验坐标 ---")
            print(f"{'像素 U':>8} | {'像素 V':>8} | {'对应深度 Z':>10}")
            count = 0
            for i in range(len(pv_points)):
                if valid_mask[i]:
                    print(f"{u_coords[i]:8.0f} | {v_coords[i]:8.0f} | {depths[i]:10.2f}m")
                    count += 1
                    if count >= 3: break

if __name__ == "__main__":
    verify_step2()