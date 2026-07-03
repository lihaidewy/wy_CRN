import os
import mmcv
import numpy as np

# 🚀 对齐你的自定义数据集根目录
DATA_ROOT = "./data/my_formatted_data"
PKL_PATH = os.path.join(DATA_ROOT, "nuscenes_infos_train.pkl")

print("="*60)
print(f" 开始对步骤三生成的索引文件进行『全功能体检』...")
print("="*60)

# 1. 检查文件本身是否存在
if not os.path.exists(PKL_PATH):
    print(f"[✗] 严重错误：在 {PKL_PATH} 找不到 .pkl 索引文件！请先运行步骤三脚本。")
    exit(1)

# 2. 载入 .pkl 数据
try:
    infos = mmcv.load(PKL_PATH)
except Exception as e:
    print(f"[✗] 严重错误：MMCV 无法解析该 .pkl 文件，可能格式损坏。报错: {e}")
    exit(1)

print(f"[✓] 基础加载成功：成功读取索引表，总计包含 {len(infos)} 帧数据。")

if len(infos) == 0:
    print("[✗] 错误：索引表内数据为空！")
    exit(1)

# 3. 抽取第一帧进行全模态细节质检
sample_info = infos[0]
print(f"\n>>> 正在抽检第一帧 (sample_token: {sample_info.get('sample_token')}) 的深层数据...")

# ------------------------------------------------------------------------------
# 3a. 图像路径与软链接层级深度体检
# ------------------------------------------------------------------------------
cam_front = sample_info.get('cam_infos', {}).get('CAM_FRONT', {})
img_filename = cam_front.get('filename')
print(f"\n[1/4] 检查 4K 图像索引:")
print(f"   -> 索引登记的相对路径: {img_filename}")

if img_filename:
    # 结合数据集根目录，算出在 Linux 下的绝对物理路径
    full_img_path = os.path.join(DATA_ROOT, img_filename)
    if os.path.exists(full_img_path):
        print(f"   [✓] 物理断点打通：Linux 顺着软链接成功透视到了该图片文件！(分辨率: {cam_front.get('width')}x{cam_front.get('height')})")
    else:
        print(f"   [✗] 路径断裂警告：底层找不到该图片文件！实际检测路径为: {full_img_path}")
        print(f"       (请检查你的图片软链接层级，确保 ls {os.path.join(DATA_ROOT, 'samples/CAM_FRONT/')} 能直接看到 1.png)")
else:
    print("   [✗] 错误：cam_infos 中缺失 filename 字段！")

# ------------------------------------------------------------------------------
# 3b. 真实 4K 内参矩阵体检
# ------------------------------------------------------------------------------
intrinsic = cam_front.get('calibrated_sensor', {}).get('camera_intrinsic', [])
print(f"\n[2/4] 检查 4K 相机内参:")
try:
    intrinsic_np = np.array(intrinsic)
    print(f"   -> 底层读取到的内参矩阵为:\n{intrinsic_np}")
    if intrinsic_np.shape == (3, 3):
        print(f"   [✓] 形状正确：严格为 3x3 矩阵。")
        if np.isclose(intrinsic_np[0, 0], 3325.5375, atol=1e-2):
            print(f"   [✓] 数值精准度对齐：成功复核你们标定的真实焦距 fx = {intrinsic_np[0, 0]}")
        else:
            print(f"   [!] 警告：焦距数值与你们提供的 3325.5375 不一致，请确认是否写错。")
    else:
        print(f"   [✗] 形状错误：内参不是 3x3 矩阵，形状为 {intrinsic_np.shape}")
except Exception as e:
    print(f"   [✗] 解析失败：内参数据损坏或格式不对。{e}")

# ------------------------------------------------------------------------------
# 3c. 瞒天过海核心：LIDAR_TOP 替换路径体检
# ------------------------------------------------------------------------------
lidar_top = sample_info.get('lidar_infos', {}).get('LIDAR_TOP', {})
lidar_filename = lidar_top.get('filename')
print(f"\n[3/4] 检查雷达基准索引 (LIDAR_TOP 替换流):")
print(f"   -> 索引登记的相对路径: {lidar_filename}")

if lidar_filename:
    full_lidar_path = os.path.join(DATA_ROOT, lidar_filename)
    if os.path.exists(full_lidar_path):
        print(f"   [✓] 伪装文件存在：成功定位步骤二生成的 7通道 BEV 特征二进制文件！")
    else:
        print(f"   [✗] 路径断裂警告：找不到特征二进制文件！检测路径为: {full_lidar_path}")
else:
    print("   [✗] 错误：lidar_infos 中缺失 filename 字段！")

# ------------------------------------------------------------------------------
# 3d. 3D 真值标签与物理速度体检
# ------------------------------------------------------------------------------
ann_infos = sample_info.get('ann_infos', [])
print(f"\n[4/4] 检查 3D 真值标签标注:")
print(f"   -> 当前帧包含 {len(ann_infos)} 个 3D 标注框。")
if len(ann_infos) > 0:
    first_ann = ann_infos[0]
    print(f"   -> 抽检第一个 Bounding Box 类别: {first_ann.get('category_name')}")
    print(f"   -> 中心点 3D 坐标 (x, y, z): {first_ann.get('translation')}")
    
    velocity = first_ann.get('velocity')
    if velocity is not None:
        print(f"   -> 登记的绝对速度向量: {velocity}")
        if isinstance(velocity, np.ndarray) or isinstance(velocity, list):
            print(f"   [✓] 速度属性合规：成功注入带方向的运动学真值参数。")
        else:
            print(f"   [✗] 速度格式不合规：必须是 list 或 numpy 数组。")
    else:
        print(f"   [✗] 缺失速度：ann_infos 里没有 velocity 字段，训练时模型无法学习速度分支。")
else:
    print("   [!] 提示：当前 ann_infos 为空（如果是测试集这很正常；如果是训练集，请务必把你们的 3D 标注数据写进去）。")

print("\n" + "="*60)
print(" 检查完毕！如果上面全线亮起 [✓]，说明步骤三完美收官，数据已成功实现伪装！")
print("="*60)