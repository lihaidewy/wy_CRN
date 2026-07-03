import os
import mmcv
import numpy as np

# ==============================================================================
# 1. 基础配置区 (严格对齐你过关的 my_formatted_data 目录)
# ==============================================================================
DATA_ROOT = "./data/my_formatted_data"

#  锁死你们提供的 100% 真实 4K 相机内参
CAMERA_INTRINSIC = [
    [3325.5375505322445,                0.0, 1920.0],
    [                0.0, 3325.5375505322445, 1080.0],
    [                0.0,                0.0,    1.0]
]

def build_nuscenes_pkl_infos():
    # 扫描步骤二生成的特征文件，自动获知有多少帧
    bev_dir = os.path.join(DATA_ROOT, "radar_bev_filter")
    bin_files = sorted([f for f in os.listdir(bev_dir) if f.endswith('.bin')])
    total_frames = len(bin_files)
    
    print(f">>> 启动步骤三：扫描到 {total_frames} 帧特征流，开始构建 .pkl 索引总表...")
    
    infos_list = []
    
    for file_name in bin_files:
        # 提取帧号
        frame_idx = int(file_name.split('_')[1].split('.')[0])
        
        info = dict()
        info['scene_name'] = "scene-0001"
        info['scene_token'] = "scene_token_001"
        info['sample_token'] = f"sample_token_{frame_idx:06d}"
        info['timestamp'] = 1600000000000000 + frame_idx * 100000  # 16位微秒级时间戳
        
        # ----------------------------------------------------------------------
        # 2a. 伪装相机元数据 (cam_infos)
        # ----------------------------------------------------------------------
        cam_infos = dict()
        sweep_cam_info = {
            'sample_token': info['sample_token'],
            'timestamp': info['timestamp'],
            'filename': f"samples/CAM_FRONT/{frame_idx}.png", # 4K 真实图片相对路径
            'height': 2160, 'width': 3840,                             # 4K 分辨率
            'is_key_frame': True,
            'ego_pose': {
                'translation': [0.0, 0.0, 0.0], 
                'rotation': [1.0, 0.0, 0.0, 0.0]                       # 龙门架静止，位姿归0
            },
            'calibrated_sensor': {
                'translation': [0.0, 0.0, 7.0],                         # 安装高度 7 米
                'rotation': [1.0, 0.0, 0.0, 0.0],                       # PITCH_DEG = 0 四元数
                'camera_intrinsic': CAMERA_INTRINSIC                    # 精确 4K 内参
            }
        }
        cam_infos['CAM_FRONT'] = sweep_cam_info
        info['cam_infos'] = cam_infos
        
        # ----------------------------------------------------------------------
        # 2b. 伪装基准坐标系 (lidar_infos)
        # ----------------------------------------------------------------------
        lidar_infos = dict()
        sweep_lidar_info = {
            'sample_token': info['sample_token'],
            'timestamp': info['timestamp'],
            #  核心关键：直接把 LIDAR_TOP 指向你步骤二生成的离线 7通道 BEV 雷达数据！
            'filename': f"radar_bev_filter/radar_{frame_idx:06d}.pcd.bin",
            'ego_pose': {'token': f"ego_token_{frame_idx:06d}", 'translation': [0.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0]},
            'calibrated_sensor': {'token': f"calib_token_{frame_idx:06d}", 'translation': [0.0, 0.0, 7.0], 'rotation': [1.0, 0.0, 0.0, 0.0]}
        }
        lidar_infos['LIDAR_TOP'] = sweep_lidar_info
        info['lidar_infos'] = lidar_infos
        
        # 给空列表占位，短路掉多帧 sweeps 加载，防止缺失历史帧报错
        info['cam_sweeps'] = []
        info['lidar_sweeps'] = []
        
        # ----------------------------------------------------------------------
        # 2c. 放入 3D 边界框真实标注 (ann_infos) -> 训练的真值监督
        # ----------------------------------------------------------------------
        #  【训练时，请在此处写代码循环读取你们该帧的 3D 标注真实标签】
        # 标注的 translation(x,y,z) 必须也是基于龙门架地平面投影点为原点的直角坐标系
        ann_infos = []
        
        # 以下为单条虚拟 3D 标注示例（代表一辆正在朝龙门架开来的小汽车）
        mock_car_ann = {
            'category_name': 'vehicle.car',
            'token': f"ann_token_{frame_idx:06d}_1",
            'translation': [-7.8, 151.7, 0.5],          # 3D 框中心点位置 (x, y, z) 紧紧咬住你的雷达点！
            'size': [2.0, 4.8, 1.6],                     # 3D 框的尺寸：宽, 长, 高 (w, l, h)
            'rotation': [1.0, 0.0, 0.0, 0.0],            # 3D 框的朝向四元数
            'num_lidar_pts': 5, 'num_radar_pts': 5,
            'velocity': np.array([0.68, -13.15, 0.0])    #  注入真实的纵向运动物理速度绝对值！
        }
        ann_infos.append(mock_car_ann)
        info['ann_infos'] = ann_infos
        
        infos_list.append(info)
        
    # 3. 导出符合 MMDetection3D 规范的 .pkl 文件
    # CRN 通常会读取这两路径，我们直接双向复制导出
    mmcv.dump(infos_list, os.path.join(DATA_ROOT, 'nuscenes_infos_train.pkl'))
    mmcv.dump(infos_list, os.path.join(DATA_ROOT, 'nuscenes_infos_val.pkl'))
    
    print(f"[✓] 步骤三执行完毕！完美生成 {len(infos_list)} 帧数据的总索引表格。")

if __name__ == "__main__":
    build_nuscenes_pkl_infos()