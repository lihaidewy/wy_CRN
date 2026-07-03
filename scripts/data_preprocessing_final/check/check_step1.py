import numpy as np
import os

# ==========================================
# 请替换为你刚刚实际生成的一个 bin 文件路径
# ==========================================
bin_file_path = "./data/my_formatted_data/samples/RADAR_FRONT/radar_000001.pcd.bin" 

def verify_step1():
    if not os.path.exists(bin_file_path):
        print(f"❌ 找不到文件: {bin_file_path}，请检查 process_radar.py 是否正常运行！")
        return

    # 读取 18 通道二进制文件
    points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 18)
    num_points = points.shape[0]
    
    print(f"✅ 成功读取雷达点云，共 {num_points} 个点。")
    print(f"✅ 数据维度检查: {points.shape} (必须是 N 行 x 18 列)")
    
    if num_points > 0:
        print("\n--- 🔍 抽取前 5 个点进行【物理常识】查验 ---")
        print(f"{'深度(X)':>10} | {'横向(Y)':>10} | {'高度(Z)':>10} | {'前向速度(vx)':>12} | {'侧向速度(vy)':>12}")
        print("-" * 70)
        for i in range(min(5, num_points)):
            x = points[i, 0]
            y = points[i, 1]
            z = points[i, 2]
            vx = points[i, 6]
            vy = points[i, 7]
            print(f"{x:10.2f} | {y:10.2f} | {z:10.2f} | {vx:12.2f} | {vy:12.2f}")
        
        print("\n--- 📊 极值统计信息查验 ---")
        print(f"X (深度) 范围: [{points[:, 0].min():.2f}m, {points[:, 0].max():.2f}m]")
        print(f"Y (横向) 范围: [{points[:, 1].min():.2f}m, {points[:, 1].max():.2f}m]")
        print(f"Z (高度) 范围: [{points[:, 2].min():.2f}m, {points[:, 2].max():.2f}m]")

if __name__ == "__main__":
    verify_step1()