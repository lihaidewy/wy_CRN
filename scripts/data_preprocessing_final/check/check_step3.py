import numpy as np
import cv2
import os
import glob

# ==========================================
# 基础配置
# ==========================================
# 用户指定的输出文件夹 (会自动创建)
OUTPUT_DIR = os.path.join("work_dirs", "check_step3_results")

# 数据集路径 (请确保路径与你实际的存放位置一致)
DEPTH_GT_DIR = "./data/my_formatted_data/depth_gt"
IMG_DIR = "./data/my_formatted_data/samples/CAM_FRONT"

# 你想要批量检查的连续图片数量 (设为 None 可以检查所有图片)
MAX_CHECK_NUM = 10  

def verify_step3_batch_ordered():
    print(f">>> 启动第三关 深度真值 (Depth GT) 顺序批量质检...")
    
    # 1. 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 结果将统一保存至: {OUTPUT_DIR}\n")

    # 2. 自动搜索所有的 .bin 文件
    search_pattern = os.path.join(DEPTH_GT_DIR, "*.bin")
    bin_files = glob.glob(search_pattern)
    
    if not bin_files:
        print(f"❌ 找不到任何深度真值文件，请检查路径: {DEPTH_GT_DIR}")
        return
        
    print(f"🔍 共发现 {len(bin_files)} 个深度真值文件。")
    
    # =========================================================================
    # 🌟 核心修改：强制按数字顺序升序排列
    # 原理：提取文件名 "40.png.bin" 中的 "40"，转为整数后进行排序
    # 这样可以杜绝 10 排在 2 前面的字符排序陷阱，保证严格的 0, 1, 2, 3 顺序
    # =========================================================================
    try:
        bin_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        print("✅ 已成功对所有文件进行严格的数字升序排列。")
    except ValueError:
        print("⚠️ 警告：文件名中包含非数字字符，无法进行完美数字排序，将回退为默认字符排序。")
        bin_files.sort()
    
    # 限制检查数量
    if MAX_CHECK_NUM is not None:
        bin_files = bin_files[:MAX_CHECK_NUM]
        print(f"⏳ 本次将【按严格顺序】检查前 {MAX_CHECK_NUM} 张图片...\n")

    # 3. 开启批量循环
    success_count = 0
    for bin_path in bin_files:
        # 从文件名中提取 frame_idx (例如从 "40.png.bin" 中提取出 "40")
        basename = os.path.basename(bin_path)
        frame_idx = basename.split('.')[0] 
        
        # 寻找对应的原图 (兼容 .png 和 .jpg)
        img_path = os.path.join(IMG_DIR, f"{frame_idx}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMG_DIR, f"{frame_idx}.jpg")
            
        if not os.path.exists(img_path):
            print(f"⚠️ [跳过] Frame {frame_idx:<4}: 找不到对应的原图。")
            continue

        # 读取二进制深度数据
        depth_pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 3)
        num_pts = depth_pts.shape[0]
        
        if num_pts == 0:
            print(f"⚠️ [跳过] Frame {frame_idx:<4}: 该帧深度点数量为 0。")
            continue

        u_coords = depth_pts[:, 0]
        v_coords = depth_pts[:, 1]
        depths = depth_pts[:, 2]

        print(f"  -> 深度(Z) 范围: [{depths.min():.1f}m ~ {depths.max():.1f}m]")
        print(f"  -> 像素 U (横): [{u_coords.min():.0f} ~ {u_coords.max():.0f}] (若最大值在 1280 左右，说明被缩放了！)")
        print(f"  -> 像素 V (纵): [{v_coords.min():.0f} ~ {v_coords.max():.0f}] (若最大值在 256 左右，说明被缩放了！)")

        # 视觉验证 (重绘到图片上)
        img = cv2.imread(img_path)
        for i in range(num_pts):
            u, v, z = int(u_coords[i]), int(v_coords[i]), depths[i]
            # u = 3840 - u
            # 根据深度上色：近处偏绿，远处偏红
            ratio = min(z / 160.0, 1.0)
            color = (0, int(255 * (1 - ratio)), int(255 * ratio))
            
            # 画实心圆
            cv2.circle(img, (u, v), 6, color, -1)

        # 保存结果到指定的 work_dirs 文件夹
        save_name = f"check_step3_visual_{frame_idx}.jpg"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        cv2.imwrite(save_path, img)
        
        print(f"✅ Frame {frame_idx:<4} | 投影点数: {num_pts:<5} | 深度范围: [{depths.min():>5.1f}m ~ {depths.max():>5.1f}m] -> 已保存")
        success_count += 1

    print(f"\n🎉 [批量质检完成] 成功按顺序处理了 {success_count} 张图片！")
    print(f"👉 请打开文件夹查看结果: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    verify_step3_batch_ordered()