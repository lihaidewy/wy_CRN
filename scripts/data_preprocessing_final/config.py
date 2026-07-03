"""
数据预处理管线 —— 集中式参数配置
==================================
所有路径、传感器参数、场景预设、批次映射统一在此文件定义。
修改一次，全局生效，杜绝参数遗漏。

重构要点:
  - 场景(Scenario): 一组固定的物理参数 (H, yaw, 朝向)
  - 批次(Offset):   每次新增数据指定一个唯一的帧号偏移
  - OFFSET_SCENARIO_MAP: offset → 场景名的映射表，新增批次时在此注册
"""

import os
import numpy as np


# ==============================================================================
#  场景物理参数预设
# ==============================================================================
class ScenarioConfig:
    """
    每个场景定义一组物理参数。新增场景只需在此添加一个字典。

    字段说明:
        name      : 场景名（命令行 --scenario 参数用）
        radar_h   : 雷达安装高度 (m)
        cam_h     : 相机安装高度 (m)
        yaw_deg   : 雷达与世界坐标系 X 轴的夹角 (度)
                    >0 表示 CCW（从上方看，向左偏）
    """

    FRONTAL = {
        'name': 'frontal',
        'radar_h': 7.0,   # 雷达安装高度
        'cam_h': 6.0,     # 相机安装高度
        'yaw_deg': 0.0,
    }

    OBLIQUE = {
        'name': 'oblique',
        'radar_h': 25.0,  # 雷达安装高度
        'cam_h': 18.0,    # 相机安装高度
        'yaw_deg': 15.0,
    }

    # 所有已注册场景
    _ALL = {
        'frontal': FRONTAL,
        'oblique': OBLIQUE,
    }

    @classmethod
    def get(cls, name):
        """根据名称获取场景配置，找不到返回 None"""
        return cls._ALL.get(name)


# ==============================================================================
#  批次→场景映射表
# ==============================================================================
# 每个 offset 范围对应一个场景。帧号 = offset_base + 原始帧号。
# 新增数据批次时只需在此添加一行映射。
#
# 格式: { offest_base: 场景名 }
# ┌──────────┬─────────┬───────────────┐
# │  offset   │ 场景     │ 说明           │
# ├──────────┼─────────┼───────────────┤
# │  0        │ frontal  │ 正视首批数据    │
# │  100000   │ frontal  │ 正视第2批      │
# │  200000   │ frontal  │ 正视第3批      │
# │  300000   │ frontal  │ 正视第4批      │
# │  400000   │ oblique  │ 斜视第1批      │
# │  500000   │ oblique  │ 斜视第2批(预留) │
# └──────────┴─────────┴───────────────┘
OFFSET_SCENARIO_MAP = {
    0:       'frontal',
    100000:  'frontal',
    200000:  'frontal',
    300000:  'frontal',
    400000:  'oblique',
    500000:  'oblique',
    600000:  'oblique',
    700000:  'oblique',
    800000:  'frontal',
    900000:  'oblique',   # ← 新增场景二数据批次
}


def get_scenario_by_frame_idx(frame_idx):
    """
    根据帧号查找对应场景。
    规则：找到 ≤ frame_idx 的最大 offset，返回对应场景名。
    返回值: (scenario_name, offset_base) 或 (None, None)
    """
    matched_offset = -1
    matched_scenario = None
    for offset, scenario_name in OFFSET_SCENARIO_MAP.items():
        if offset <= frame_idx and offset > matched_offset:
            matched_offset = offset
            matched_scenario = scenario_name
    if matched_scenario is None:
        return None, None
    return matched_scenario, matched_offset


# ==============================================================================
#  路径配置
# ==============================================================================
class PathConfig:
    """数据输入/输出路径"""
    DATA_ROOT = "./data/my_formatted_data"

    # ---- 新批次临时数据源 ----
    # oblique 700000 批次 (WSL路径)
    NEW_IMG_DIR = "/mnt/e/数据/add_cam"
    NEW_JSON_DIR = "/mnt/e/数据/addS1/addS1/S1_3dbox"

    # ---- 主数据目录下的子目录（相对于 DATA_ROOT）----
    SAMPLES_RADAR_FRONT = "samples/RADAR_FRONT"
    SAMPLES_CAM_FRONT = "samples/CAM_FRONT"
    RADAR_BEV_FILTER = "radar_bev_filter"
    RADAR_PV_FILTER = "radar_pv_filter"
    DEPTH_GT = "depth_gt"
    JSONS = "jsons"

    # ---- 产物文件名 ----
    TRAIN_PKL = "nuscenes_infos_train.pkl"
    VAL_PKL = "nuscenes_infos_val.pkl"

    # ---- 分场景独立训练产物 ----
    FRONTAL_TRAIN_PKL = "nuscenes_infos_frontal_train.pkl"
    FRONTAL_VAL_PKL = "nuscenes_infos_frontal_val.pkl"
    OBLIQUE_TRAIN_PKL = "nuscenes_infos_oblique_train.pkl"
    OBLIQUE_VAL_PKL = "nuscenes_infos_oblique_val.pkl"

    @classmethod
    def get_abs_path(cls, rel_path):
        """将相对 DATA_ROOT 的路径转为绝对路径"""
        return os.path.join(cls.DATA_ROOT, rel_path)


# ==============================================================================
#  传感器固定参数（不随场景变化）
# ==============================================================================
class SensorConfig:
    """相机固定参数 — 内参和图像尺寸，通常不随场景变化"""

    # ---- 相机内参矩阵 ----
    CAMERA_INTRINSIC = np.array([
        [3325.5375505322445,                0.0, 1920.0],
        [               0.0, 3325.5375505322445, 1080.0],
        [               0.0,                0.0,    1.0],
    ])

    # ---- 图像分辨率 ----
    IMAGE_WIDTH = 3840
    IMAGE_HEIGHT = 2160

    # ---- 雷达点云参数 ----
    RADAR_MIN_DEPTH_CAM = 0.5   # 相机坐标系下最小深度（过滤镜头后方点）


# ==============================================================================
#  管线行为配置
# ==============================================================================
class PipelineConfig:
    """控制管线运行行为"""

    # ---- 训练/验证集划分 ----
    # "all_train"     : 全部作为训练集
    # "sequential"    : 按比例顺序切分（全局）
    # "range_based"   : 按帧号范围过滤
    # "per_scenario"  : 每个场景内固定帧数验证（用于分场景独立训练）
    # "train_test"    : 按 offset 严格划分训练/测试（用于泛化验证）
    TRAIN_VAL_SPLIT = "train_test"
    SPLIT_RATIO = 0.8
    VAL_FRAMES_PER_SCENARIO = 100  # 每场景固定验证帧数（用于分场景训练）
    VAL_FRAME_RANGE = (200000, 400000)

    # ---- 训练/测试集 offset 划分（泛化验证用）----
    # 格式: {场景名: {'train': [offset列表], 'test': [offset列表]}}
    TRAIN_TEST_OFFSETS = {
        'frontal': {
            'train': [0, 100000, 800000],
            'test':  [200000, 300000],
        },
        'oblique': {
            'train': [400000, 700000, 900000],   # ← 新增 900000
            'test':  [500000, 600000],
        },
    }

    # ---- 调试 ----
    DEBUG_VIS_FRAME = False
    DEBUG_OUTPUT_DIR = "./work_dirs/debug_radar_proj"


# ==============================================================================
#  类别映射
# ==============================================================================
CATEGORY_MAP = {
    'car': 'vehicle.car',
    'truck': 'vehicle.truck',
    'bus': 'vehicle.bus.rigid',
    'pedestrian': 'human.pedestrian.adult',
    'bicycle': 'vehicle.bicycle',
    'motorcycle': 'vehicle.motorcycle',
}

MAP_NAME_GENERAL_TO_DETECTION = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
