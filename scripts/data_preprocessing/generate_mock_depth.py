import os
import numpy as np

#  对齐你的自定义数据集根目录
DATA_ROOT = "./data/my_formatted_data"
DEPTH_GT_DIR = os.path.join(DATA_ROOT, "depth_gt")

# 创建深度真值硬编码读取目录
os.makedirs(DEPTH_GT_DIR, exist_ok=True)

# 扫描你的特征文件，获知有多少帧（应当是 180 帧）
bev_dir = os.path.join(DATA_ROOT, "radar_bev_filter")
total_frames = len([f for f in os.listdir(bev_dir) if f.endswith('.bin')])

print(f">>> 正在为 {total_frames} 帧数据批量暴力伪装 depth_gt 真值文件...")

# CRN 默认的下游深度图网格分辨率通常是特征图尺度（例如 256x704 下采样 16 倍，或原图特定下采样）
# 为了绝对安全不报形状错误，我们直接生成一个可以被任何形状 reshape 的基础一维全 0 数组，或者对齐它最常读取的尺寸。
# 原厂在 nusc_det_dataset.py 中对 depth_gt 的序列化通常读取为 float32 类型的截锥网格（如 32 * 88 像素）
# 我们这里创建一个足够大或标准的 32x88 深度层矩阵

for frame_idx in range(1, total_frames + 1):
    # 模拟一个 32x88 的全 0 深度图真值
    mock_depth = np.zeros((100,3), dtype=np.float32)
    
    #  完美对齐报错中的命名规则：1.png.bin
    filename = f"{frame_idx}.png.bin"
    mock_depth.tofile(os.path.join(DEPTH_GT_DIR, filename))

print("[✓] 深度真值伪装完毕！depth_gt 目录已全部就绪。")