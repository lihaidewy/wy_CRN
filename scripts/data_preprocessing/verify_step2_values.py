import os
import numpy as np

# 🚀 对齐你的真实目录
DATA_ROOT = "./data/my_formatted_data"
FRAME_IDX = 1  # 抽检第 1 帧

print(f"==== 正在对第 {FRAME_IDX} 帧的步骤二输出进行数值质检 ====")

# ------------------------------------------------------------------------------
# 1. 质检 radar_bev_filter
# ------------------------------------------------------------------------------
bev_path = os.path.join(DATA_ROOT, f"radar_bev_filter/radar_{FRAME_IDX:06d}.pcd.bin")

if not os.path.exists(bev_path):
    print(f"[✗] 错误：找不到 BEV 文件 {bev_path}")
else:
    # 按照 7 通道强行反序列化
    bev_data = np.fromfile(bev_path, dtype=np.float32).reshape(-1, 7)
    print(f"\n[1/2] 正在检测 BEV 衍生流 (radar_bev_filter):")
    print(f"   -> 矩阵形状 (点的个数, 通道数): {bev_data.shape}")
    
    if bev_data.shape[1] == 7:
        print("   [✓] 通道数正确：严格符合 7 通道要求。")
    else:
        print("   [✗] 通道数错误：不满足 7 通道。")
        
    # 提取 Z 轴（通道索引 2）
    z_values = bev_data[:, 2]
    if np.all(z_values == 0.0):
        print("   [✓] 空间消高正确：所有点的 Z 轴已完美归 0。")
    else:
        print("   [✗] 空间消高错误：Z 轴存在非 0 值。")
        
    # 打印前 2 个点的具体数值，人工肉眼复核 [x, y, z, rcs/snr, vx, vy, sweep]
    print(f"   -> 前两个点的真实数值样本:\n{bev_data[:2]}")


# ------------------------------------------------------------------------------
# 2. 质检 radar_pv_filter
# ------------------------------------------------------------------------------
pv_path = os.path.join(DATA_ROOT, f"radar_pv_filter/image_{FRAME_IDX:06d}.jpg.bin")

if not os.path.exists(pv_path):
    print(f"[✗] 错误：找不到 PV 文件 {pv_path}")
else:
    pv_data = np.fromfile(pv_path, dtype=np.float32).reshape(-1, 7)
    print(f"\n[2/2] 正在检测 4K 图像投影 PV 衍生流 (radar_pv_filter):")
    print(f"   -> 矩阵形状 (点的个数, 通道数): {pv_data.shape}")
    
    u_coords = pv_data[:, 0]
    v_coords = pv_data[:, 1]
    depths = pv_data[:, 2]
    
    # 验证 4K 像素范围界限
    u_in_screen = np.logical_and(u_coords >= 0, u_coords <= 3840)
    v_in_screen = np.logical_and(v_coords >= 0, v_coords <= 2160)
    valid_pixel_ratio = np.sum(np.logical_and(u_in_screen, v_in_screen)) / len(u_coords) * 100
    
    print(f"   -> 像素 U 坐标范围: [{u_coords.min():.1f} ~ {u_coords.max():.1f}]")
    print(f"   -> 像素 V 坐标范围: [{v_coords.min():.1f} ~ {v_coords.max():.1f}]")
    print(f"   -> 投影点落在 4K 画面内的比例: {valid_pixel_ratio:.1f}%")
    
    if valid_pixel_ratio > 30:
        print("   [✓] 4K 内参矩阵有效：雷达点能大面积正确投射进 4K 画布内。")
    else:
        print("   [✗] 内参或几何变换可能存在偏差：大部分雷达点飞到了 4K 画面外面，请核对内参！")
        
    # 🚀 锁死 PITCH_DEG = 0 的深度闭环终极验证
    # 在 0 度俯仰角下，PV流里的深度 depth(通道2) 必须严丝合缝地等于 BEV流里的地面纵向距离 Y(通道1)
    if np.allclose(depths, bev_data[:, 1], atol=1e-3):
        print("   [✓] 物理深度闭环验证通过：深度完全等同于纵向距离 Y，100% 正确！")
    else:
        print("   [✗] 深度闭环失败：PV流的 depth 与 BEV流的 Y轴不相等，数学公式有错。")
        
    print(f"   -> 前两个点的真实数值样本 [u, v, depth, snr, vx, vy, sweep]:\n{pv_data[:2]}")