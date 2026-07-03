#!/usr/bin/env python
"""
CRN 数据预处理 - 统一管线 (v2 — 多场景支持)
==============================================
将原始传感器数据转换为 CRN 训练所需的全部中间产物。
支持多个场景 (正视/斜视) 和多个数据批次。

用法:
    # 查看配置
    python pipeline.py --show-config

    # 正视场景（默认）
    python pipeline.py --scenario frontal --offset 0 --stage radar2bin

    # 斜视场景
    python pipeline.py --scenario oblique --offset 400000 --stage radar2bin

    # 新增数据批次（正视第5批，offset=600000）
    python pipeline.py --scenario frontal --offset 600000 --stage radar2bin

    # 一键全管线（处理完雷达后）
    python pipeline.py --scenario frontal --stage derivatives
    python pipeline.py --scenario oblique --stage derivatives
    python pipeline.py --scenario frontal --stage depth_gt
    python pipeline.py --scenario oblique --stage depth_gt
    python pipeline.py --stage build_infos        # 自动合并所有场景
"""

import os
import re
import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import (
    PathConfig, SensorConfig, PipelineConfig,
    ScenarioConfig, OFFSET_SCENARIO_MAP, get_scenario_by_frame_idx,
    CATEGORY_MAP,
)


# ══════════════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════════════
def _yaw_quaternion(yaw_deg):
    """偏航角(度) -> 绕Z轴旋转四元数 [w, x, y, z]"""
    half = np.deg2rad(yaw_deg) / 2
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]


def _quat_multiply(q1, q2):
    """四元数乘法 q1 * q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]


def _yaw_to_quaternion(yaw):
    """偏航角（弧度）-> 四元数 [w, x, y, z]"""
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def _mat_to_quaternion(R):
    """3x3 旋转矩阵 -> 四元数 [w, x, y, z]"""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [w, x, y, z]


def _invert_extrinsics(E_4x4):
    """
    world->camera 外参矩阵 -> camera->world (即 sensor2ego).
    E = [R | t], 求逆 = [R^T | -R^T @ t].
    返回: (translation_3d, rotation_quat_4d)
    """
    R = np.array(E_4x4)[:3, :3]
    t = np.array(E_4x4)[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    quat = _mat_to_quaternion(R_inv)
    return t_inv.tolist(), quat


def _quat_to_R(q):
    """四元数 [w,x,y,z] -> 3x3 旋转矩阵"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def _get_frame_camera_extrinsics(frame_idx):
    """
    根据帧号获取相机外参 (translation, rotation_quat)。
    优先读取 JSON 文件中的 extrinsics_matrix_4x4，否则用场景参数计算。
    """
    json_path = os.path.join(PathConfig.DATA_ROOT, PathConfig.JSONS, f"{frame_idx}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return _build_camera_extrinsics(frame_idx, data)
    else:
        # 无 JSON，回退到场景参数
        sc_name, _ = get_scenario_by_frame_idx(frame_idx)
        sc = ScenarioConfig.get(sc_name) if sc_name else None
        cam_h = sc['cam_h'] if sc else 6.0
        yaw_deg = sc['yaw_deg'] if sc else 0.0
        translation = [0.0, 0.0, cam_h]
        base_rotation = [0.5, -0.5, 0.5, -0.5]
        if yaw_deg != 0.0:
            yaw_quat = _yaw_quaternion(yaw_deg)
            rotation = _quat_multiply(yaw_quat, base_rotation)
        else:
            rotation = base_rotation
        return translation, rotation


def _get_radar_xy_from_json(frame_idx):
    """
    读取 JSON 获取相机的水平安装位置 [X, Y]。
    假设雷达和相机安装在同一水平位置。
    如果 JSON 不存在，回退到 [0.0, 0.0]。
    """
    json_path = os.path.join(PathConfig.DATA_ROOT, PathConfig.JSONS, f"{frame_idx}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'extrinsics_matrix_4x4' in data:
            E = np.array(data['extrinsics_matrix_4x4'])
            # E 是 world->camera，求逆得到 camera->world 的 translation（相机世界位置）
            R = E[:3, :3]
            t = E[:3, 3]
            R_inv = R.T
            t_inv = -R_inv @ t
            return [float(t_inv[0]), float(t_inv[1])]
    # 无 JSON 或缺外参，回退到道路中心
    return [0.0, 0.0]


def _world_to_cam(points_world, translation, rotation_quat):
    """
    将世界坐标系点转换到相机坐标系。
    Args:
        points_world: (N, 3) 或 (3, N)
        translation:  camera->world 的平移 (世界坐标系下相机位置)
        rotation_quat: camera->world 的旋转四元数 [w,x,y,z]
    Returns:
        同 shape 的相机坐标系点
    """
    is_1d = False
    if points_world.ndim == 1:
        points_world = points_world.reshape(3, 1)
        is_1d = True

    # 先平移：P_local = P_world - t
    pts_local = points_world.copy()
    if pts_local.shape[0] == 3:
        pts_local = pts_local - np.array(translation).reshape(3, 1)
    else:
        pts_local = pts_local.T - np.array(translation).reshape(3, 1)

    # 再旋转：rotation_quat 是 camera->world，其逆是世界->camera
    # 单位四元数的逆 = 共轭
    R = _quat_to_R(rotation_quat).T  # 转置 = 逆旋转
    pts_cam = R @ pts_local

    if is_1d:
        pts_cam = pts_cam.flatten()
    return pts_cam


def _rotate_radar_to_world(x_forward, y_left, yaw_deg):
    """
    将雷达本地坐标旋转到世界坐标系。
    雷达本地: X->雷达前方, Y->雷达左侧
    世界:      X->道路前方, Y->道路左侧
    yaw_deg > 0: 雷达向左偏（CCW，从上方看）
    """
    if yaw_deg == 0.0:
        return x_forward, y_left
    yaw = np.deg2rad(yaw_deg)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    x_world = x_forward * cos_yaw - y_left * sin_yaw
    y_world = x_forward * sin_yaw + y_left * cos_yaw
    return x_world, y_world


def _convert_csv_frame_to_bin(df_frame, frame_idx, radar_h, yaw_deg, radar_xy=(0.0, 0.0)):
    """
    将单帧雷达 CSV -> nuScenes 18 通道二进制。

    处理流程:
      极坐标(θ, r) -> 雷达本地笛卡尔(X_radar, Y_left)
                    -> yaw旋转 -> 世界坐标(X_world, Y_world)
                    -> 平移到雷达安装位置

    Args:
        df_frame: DataFrame (Angle, Range, Speed, SNR, ID)
        frame_idx: 帧号（已叠加 OFFSET）
        radar_h:   雷达安装高度 (m)
        yaw_deg:   雷达与世界X轴夹角 (度)
        radar_xy:  雷达水平安装位置 [X, Y] (世界坐标系，默认 [0,0])
    """
    a_deg = df_frame["Angle"].values
    r_val = df_frame["Range"].values
    v_radial = df_frame["Speed"].values
    snr_val = df_frame["SNR"].values
    obj_id = df_frame["ID"].values

    a_rad = np.deg2rad(a_deg)

    # 高度盲区过滤（使用雷达真实高度）
    depth_sq = (r_val * np.cos(a_rad)) ** 2 - radar_h ** 2
    valid_mask = depth_sq > 0.1

    a_valid = a_rad[valid_mask]
    r_valid = r_val[valid_mask]
    v_radial = v_radial[valid_mask]
    snr_valid = snr_val[valid_mask]
    id_valid = obj_id[valid_mask]

    # 雷达本地坐标：前方X，左侧Y（Angle正值=左侧，标准极坐标）
    x_radar = np.sqrt(depth_sq[valid_mask])
    y_radar = r_valid * np.sin(a_valid)

    # 速度分解（雷达本地）
    vx_radar = v_radial * np.cos(a_valid)
    vy_radar = v_radial * np.sin(a_valid)

    # 雷达本地 -> 世界坐标（yaw旋转 + 平移到安装位置）
    x_world, y_world = _rotate_radar_to_world(x_radar, y_radar, yaw_deg)
    vx_world, vy_world = _rotate_radar_to_world(vx_radar, vy_radar, yaw_deg)

    # 平移到雷达真实水平安装位置（相机同位置，仅高度不同）
    x_world += radar_xy[0]
    y_world += radar_xy[1]

    num_points = len(x_world)
    radar_18ch = np.zeros((num_points, 18), dtype=np.float32)
    radar_18ch[:, 0] = x_world          # [0] 世界X (道路前方)
    radar_18ch[:, 1] = y_world          # [1] 世界Y (道路左侧)
    radar_18ch[:, 2] = 0.0              # [2] Z贴地
    radar_18ch[:, 4] = id_valid         # [4] Object_ID
    radar_18ch[:, 5] = snr_valid        # [5] SNR
    radar_18ch[:, 6] = vx_world         # [6] 世界vx
    radar_18ch[:, 7] = vy_world         # [7] 世界vy
    radar_18ch[:, 8] = vx_world         # [8] vx_comp
    radar_18ch[:, 9] = vy_world         # [9] vy_comp

    bin_filename = f"samples/RADAR_FRONT/radar_{frame_idx:06d}.pcd.bin"
    output_path = os.path.join(PathConfig.DATA_ROOT, bin_filename)
    radar_18ch.tofile(output_path)
    return bin_filename, num_points


# ══════════════════════════════════════════════════════════════════════════════
#  阶段1: 雷达 CSV -> 18ch 二进制
# ══════════════════════════════════════════════════════════════════════════════
def stage_radar_csv_to_bin(scenario_name='frontal', csv_path=None, offset=None):
    """
    原始雷达 CSV -> nuScenes 标准 18ch .bin 文件。

    Args:
        scenario_name: 场景名（决定 H 和 yaw）
        csv_path:      CSV 源路径（None 则用命令行 --csv 参数）
        offset:        帧号偏移量
    """
    scenario = ScenarioConfig.get(scenario_name)
    if scenario is None:
        print(f"[ERR] 未知场景 '{scenario_name}'，可用: {list(ScenarioConfig._ALL.keys())}")
        return

    radar_h = scenario['radar_h']
    yaw_deg = scenario['yaw_deg']

    samples_dir = os.path.join(PathConfig.DATA_ROOT, PathConfig.SAMPLES_RADAR_FRONT)
    os.makedirs(samples_dir, exist_ok=True)

    if csv_path is None:
        print(f"[ERR] 请通过 --csv 参数指定雷达 CSV 路径")
        return
    if not os.path.exists(csv_path):
        print(f"[ERR] 找不到雷达 CSV 文件 {csv_path}")
        return
    if offset is None:
        print(f"[[ERR]] 错误：请通过 --offset 参数指定帧号偏移量")
        return

    print(f">>> [阶段1] 雷达 CSV -> 18ch 二进制")
    print(f"    场景:     {scenario_name}")
    print(f"    雷达高度: {radar_h}m")
    print(f"    Yaw偏角:  {yaw_deg}°")
    print(f"    CSV源:    {csv_path}")
    print(f"    OFFSET:   {offset}")
    print("-" * 65)

    df_all = pd.read_csv(csv_path)
    grouped = df_all.groupby("Frame")

    for frame_idx, df_frame in grouped:
        new_frame_idx = int(frame_idx) + offset
        # 读取对应 JSON 获取雷达/相机的水平安装位置
        radar_xy = _get_radar_xy_from_json(new_frame_idx)
        rel_path, num_pts = _convert_csv_frame_to_bin(
            df_frame, new_frame_idx, radar_h, yaw_deg, radar_xy=radar_xy)
        print(f"  [[OK]] 帧 {new_frame_idx} ({num_pts} 点) radar_xy={radar_xy} -> {rel_path}")

    print(f"\n[[OK]] 阶段1 完成！共 {len(grouped)} 帧。")


# ══════════════════════════════════════════════════════════════════════════════
#  阶段1.5: 合并新批次图片 + JSON
# ══════════════════════════════════════════════════════════════════════════════
def stage_merge_new_data(offset=None):
    """
    将临时目录中的图片和 JSON 标注复制到主数据目录，带 OFFSET 帧号重命名。
    """
    if offset is None:
        print("[[ERR]] 错误：请通过 --offset 指定帧号偏移量")
        return

    print(f">>> [阶段1.5] 合并新批次数据（OFFSET={offset}）")
    print("-" * 65)

    main_img_dir = os.path.join(PathConfig.DATA_ROOT, PathConfig.SAMPLES_CAM_FRONT)
    main_json_dir = os.path.join(PathConfig.DATA_ROOT, PathConfig.JSONS)
    os.makedirs(main_img_dir, exist_ok=True)
    os.makedirs(main_json_dir, exist_ok=True)

    def _migrate(src_dir, dst_dir, ext):
        if not os.path.exists(src_dir):
            print(f"  [!] 源目录 {src_dir} 不存在，跳过。")
            return 0
        count = 0
        for filename in sorted(os.listdir(src_dir)):
            if filename.endswith(ext):
                old_idx = int(filename.split('.')[0])
                new_idx = old_idx + offset
                new_filename = f"{new_idx}{ext}"
                shutil.copy2(os.path.join(src_dir, filename),
                             os.path.join(dst_dir, new_filename))
                count += 1
        print(f"  [[OK]] {ext}: {count} 个 -> {dst_dir}")
        return count

    _migrate(PathConfig.NEW_IMG_DIR, main_img_dir, ".png")
    _migrate(PathConfig.NEW_JSON_DIR, main_json_dir, ".json")

    print(f"\n[[OK]] 阶段1.5 完成！")


# ══════════════════════════════════════════════════════════════════════════════
#  阶段2: 18ch -> BEV(7ch) + PV(7ch)
# ══════════════════════════════════════════════════════════════════════════════
def stage_generate_derivatives():
    """
    从 18ch 雷达点云生成双流特征（BEV + PV）。
    自动根据帧号判断场景，每帧使用对应的 H 参数。
    """
    print(f">>> [阶段2] 18ch -> BEV(7ch) + PV(7ch)")
    print(f"    自动按帧号识别场景，使用对应安装高度")
    print("-" * 65)

    bev_dir = os.path.join(PathConfig.DATA_ROOT, PathConfig.RADAR_BEV_FILTER)
    pv_dir = os.path.join(PathConfig.DATA_ROOT, PathConfig.RADAR_PV_FILTER)
    os.makedirs(bev_dir, exist_ok=True)
    os.makedirs(pv_dir, exist_ok=True)

    base_radar_dir = os.path.join(PathConfig.DATA_ROOT, PathConfig.SAMPLES_RADAR_FRONT)
    if not os.path.exists(base_radar_dir):
        print(f"[[ERR]] 错误：找不到 {base_radar_dir}")
        return

    bin_files = sorted([f for f in os.listdir(base_radar_dir) if f.endswith('.bin')])
    print(f"    扫描到 {len(bin_files)} 帧雷达数据")

    K = SensorConfig.CAMERA_INTRINSIC

    for file_name in bin_files:
        frame_idx = int(file_name.split('_')[1].split('.')[0])
        base_bin_path = os.path.join(base_radar_dir, file_name)
        points_18ch = np.fromfile(base_bin_path, dtype=np.float32).reshape(-1, 18)
        num_pts = points_18ch.shape[0]

        # ---- BEV 7通道 ----
        radar_bev = np.zeros((num_pts, 7), dtype=np.float32)
        radar_bev[:, 0] = points_18ch[:, 0]          # X
        radar_bev[:, 1] = points_18ch[:, 1]          # Y
        radar_bev[:, 2] = points_18ch[:, 2]          # Z
        radar_bev[:, 3] = points_18ch[:, 5]          # RCS
        radar_bev[:, 4:6] = points_18ch[:, 8:10]     # vx, vy
        radar_bev[:, 6] = 0.0                         # sweep_idx

        bev_filename = f"radar_{frame_idx:06d}.pcd.bin"
        radar_bev.tofile(os.path.join(bev_dir, bev_filename))

        # ---- PV 7通道（透视投影，使用完整相机外参）----
        trans, rot_quat = _get_frame_camera_extrinsics(frame_idx)
        radar_pv = np.zeros((num_pts, 7), dtype=np.float32)
        for i in range(num_pts):
            pt_w = radar_bev[i, :3]
            pt_cam = _world_to_cam(pt_w, trans, rot_quat)
            cam_x, cam_y, cam_z = pt_cam[0], pt_cam[1], pt_cam[2]

            if cam_z > SensorConfig.RADAR_MIN_DEPTH_CAM:
                u = (K[0, 0] * cam_x / cam_z) + K[0, 2]
                v = (K[1, 1] * cam_y / cam_z) + K[1, 2]
            else:
                u, v = -1, -1

            radar_pv[i, 0] = u
            radar_pv[i, 1] = v
            radar_pv[i, 2] = cam_z

        radar_pv[:, 3] = radar_bev[:, 3]
        radar_pv[:, 4:6] = radar_bev[:, 4:6]
        radar_pv[:, 6] = 0.0

        pv_filename = f"{frame_idx}.png.bin"
        radar_pv.tofile(os.path.join(pv_dir, pv_filename))

    print(f"\n[[OK]] 阶段2 完成！")


# ══════════════════════════════════════════════════════════════════════════════
#  阶段3: BEV -> 深度真值
# ══════════════════════════════════════════════════════════════════════════════
def stage_build_depth_gt():
    """
    雷达 BEV 点云 -> 相机平面投影，生成稀疏深度真值。
    自动根据帧号判断场景，每帧使用对应的 H 参数。
    """
    print(f">>> [阶段3] BEV -> 深度真值 (Depth GT)")
    print(f"    自动按帧号识别场景，使用对应安装高度")
    print("-" * 65)
    print("-" * 65)

    bev_dir = os.path.join(PathConfig.DATA_ROOT, PathConfig.RADAR_BEV_FILTER)
    depth_gt_dir = os.path.join(PathConfig.DATA_ROOT, PathConfig.DEPTH_GT)
    image_dir = os.path.join(PathConfig.DATA_ROOT, PathConfig.SAMPLES_CAM_FRONT)
    os.makedirs(depth_gt_dir, exist_ok=True)
    os.makedirs(PipelineConfig.DEBUG_OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(bev_dir):
        print(f"[[ERR]] 错误：找不到 BEV 目录 {bev_dir}，请先运行阶段2。")
        return

    bin_files = sorted([f for f in os.listdir(bev_dir) if f.endswith('.bin')])
    print(f"    扫描到 {len(bin_files)} 帧")

    K = SensorConfig.CAMERA_INTRINSIC
    W_img = SensorConfig.IMAGE_WIDTH
    H_img = SensorConfig.IMAGE_HEIGHT
    first_frame = True
    test_img = None

    print(f"{'Frame':<8} | {'Original':<10} | {'Kept':<10} | {'Rate':<10}")
    print("-" * 65)

    for file_name in bin_files:
        numbers = re.findall(r'\d+', file_name)
        if not numbers:
            continue
        frame_idx = int(numbers[-1])

        radar_points = np.fromfile(
            os.path.join(bev_dir, file_name), dtype=np.float32).reshape(-1, 7)
        total_pts = len(radar_points)
        if total_pts == 0:
            continue

        # ---- 使用完整相机外参进行 world -> camera 变换 ----
        trans, rot_quat = _get_frame_camera_extrinsics(frame_idx)
        pts_world = radar_points[:, :3].T  # (3, N)
        pts_cam = _world_to_cam(pts_world, trans, rot_quat)  # (3, N)
        cam_x, cam_y, cam_z = pts_cam[0, :], pts_cam[1, :], pts_cam[2, :]

        if first_frame and len(cam_z) > 0:
            print(f"  [探针] Frame {frame_idx}: cam_Z={cam_z[:3]} cam_X={cam_x[:3]} cam_Y={cam_y[:3]}")
            first_frame = False

        valid = cam_z > SensorConfig.RADAR_MIN_DEPTH_CAM
        cam_z, cam_x, cam_y = cam_z[valid], cam_x[valid], cam_y[valid]
        if len(cam_z) == 0:
            print(f"{frame_idx:<8} | {total_pts:<10} | 0          | 0.00%")
            continue

        u = (K[0, 0] * cam_x / cam_z) + K[0, 2]
        v = (K[1, 1] * cam_y / cam_z) + K[1, 2]

        in_view = (u >= 0) & (u <= W_img) & (v >= 0) & (v <= H_img)
        final_u = u[in_view].astype(int)
        final_v = v[in_view].astype(int)
        final_z = cam_z[in_view]

        final_pts = len(final_z)
        rate = (final_pts / total_pts * 100) if total_pts > 0 else 0
        print(f"{frame_idx:<8} | {total_pts:<10} | {final_pts:<10} | {rate:<10.2f}%")

        depth_gt = np.stack([final_u, final_v, final_z], axis=1).astype(np.float32)
        depth_gt.tofile(os.path.join(depth_gt_dir, f"{frame_idx}.png.bin"))

        if PipelineConfig.DEBUG_VIS_FRAME and final_pts > 0 and test_img is None:
            img_path = os.path.join(image_dir, f"{frame_idx}.png")
            if os.path.exists(img_path):
                import cv2
                test_img = cv2.imread(img_path)
                for i in range(final_pts):
                    color = (0,
                             int(255 * (1 - min(final_z[i] / 160, 1))),
                             int(255 * min(final_z[i] / 160, 1)))
                    cv2.circle(test_img, (final_u[i], final_v[i]), 3, color, -1)
                out_path = os.path.join(PipelineConfig.DEBUG_OUTPUT_DIR,
                                        f"radar_proj_{frame_idx}.jpg")
                cv2.imwrite(out_path, test_img)
                print(f"  [验证图] {out_path}")

    print(f"\n[[OK]] 阶段3 完成！")


# ══════════════════════════════════════════════════════════════════════════════
#  阶段4: JSON -> nuscenes info PKL（多场景自动合并）
# ══════════════════════════════════════════════════════════════════════════════
def _build_camera_extrinsics(frame_idx, data):
    """
    为单帧确定 calibrated_sensor 的外参。

    优先级:
      1. JSON 中有 extrinsics_matrix_4x4 -> 求逆得到 camera->world
      2. 否则 -> 从场景物理参数计算（正视场景的默认路径）
    """
    # --- 根据场景参数计算正确的 rotation（作为基准）---
    sc_name, sc_offset = get_scenario_by_frame_idx(frame_idx)
    scenario = ScenarioConfig.get(sc_name) if sc_name else None

    if scenario is None:
        cam_h = 6.0
        yaw_deg_fallback = 0.0
    else:
        cam_h = scenario['cam_h']
        yaw_deg_fallback = scenario['yaw_deg']

    base_rotation = [0.5, -0.5, 0.5, -0.5]
    if yaw_deg_fallback != 0.0:
        yaw_quat = _yaw_quaternion(yaw_deg_fallback)
        scenario_rotation = _quat_multiply(yaw_quat, base_rotation)
    else:
        scenario_rotation = base_rotation

    if 'extrinsics_matrix_4x4' in data:
        # --- 从标注外参矩阵自动提取 ---
        E = np.array(data['extrinsics_matrix_4x4'])
        translation, rotation_quat = _invert_extrinsics(E)

        # 如果提取的 rotation 接近 identity（w > 0.999），说明标注文件缺旋转
        # 此时用场景参数计算的 rotation 替代
        # 注意：15° 旋转的 w=0.9914，阈值必须高于此值才能避免误判
        if rotation_quat[0] > 0.999:
            print(f"  [WARN] Frame {frame_idx}: JSON extrinsics rotation ≈ identity, "
                  f"using scenario-calculated rotation instead.")
            return translation, scenario_rotation
        else:
            # JSON 包含额外的旋转。检查是否只是绕 Z 轴的 yaw（x≈0, y≈0）。
            # 如果是，说明 JSON 只给了 yaw 但没有 base_rotation，需要组合。
            if abs(rotation_quat[1]) < 0.01 and abs(rotation_quat[2]) < 0.01:
                combined = _quat_multiply(rotation_quat, base_rotation)
                return translation, combined
            else:
                return translation, rotation_quat
    else:
        # --- JSON 无外参矩阵，完全回退到场景参数 ---
        translation = [0.0, 0.0, cam_h]
        return translation, scenario_rotation


def stage_build_infos():
    """
    从所有 JSON 标注生成 nuscenes info PKL。
    自动根据帧号判断场景，赋予正确的相机外参。
    """
    print(f">>> [阶段4] JSON -> nuscenes infos PKL（多场景自动合并）")
    print("-" * 65)

    json_dir = os.path.join(PathConfig.DATA_ROOT, PathConfig.JSONS)
    if not os.path.exists(json_dir):
        print(f"[[ERR]] 错误：找不到 JSON 目录 {json_dir}")
        return

    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')],
                        key=lambda x: int(x.split('.')[0]))
    print(f"    发现 {len(json_files)} 帧标注")

    infos_list = []
    scenario_counts = {}  # 统计各场景帧数

    for file_name in json_files:
        frame_idx = int(file_name.split('.')[0])
        with open(os.path.join(json_dir, file_name), 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 确定场景
        sc_name, sc_offset = get_scenario_by_frame_idx(frame_idx)
        scenario_counts[sc_name or 'unknown'] = scenario_counts.get(sc_name or 'unknown', 0) + 1

        # 获取该帧的相机外参
        cam_trans, cam_rot = _build_camera_extrinsics(frame_idx, data)

        info = {
            'scene_name': f"roadside_{sc_name or 'unknown'}",
            'scene_token': f"scene_token_{sc_name or 'unknown'}",
            'sample_token': f"sample_token_{frame_idx:06d}",
            'timestamp': 1600000000000000 + frame_idx * 100000,
            'cam_infos': {},
            'lidar_infos': {},
            'ann_infos': [],
            'lidar_sweeps': [],
            'cam_sweeps': [],
        }

        # 相机内参（优先从 JSON 读取）
        intrinsics = data.get('intrinsics', {})
        camera_intrinsic = [
            [intrinsics.get('fx', float(SensorConfig.CAMERA_INTRINSIC[0, 0])),
             0.0,
             intrinsics.get('cx', float(SensorConfig.CAMERA_INTRINSIC[0, 2]))],
            [0.0,
             intrinsics.get('fy', float(SensorConfig.CAMERA_INTRINSIC[1, 1])),
             intrinsics.get('cy', float(SensorConfig.CAMERA_INTRINSIC[1, 2]))],
            [0.0, 0.0, 1.0],
        ]

        # 相机元数据（使用自动提取的外参）
        info['cam_infos']['CAM_FRONT'] = {
            'sample_token': info['sample_token'],
            'timestamp': info['timestamp'],
            'filename': f"samples/CAM_FRONT/{frame_idx}.png",
            'height': SensorConfig.IMAGE_HEIGHT,
            'width': SensorConfig.IMAGE_WIDTH,
            'is_key_frame': True,
            'ego_pose': {'translation': [0.0, 0.0, 0.0],
                         'rotation': [1.0, 0.0, 0.0, 0.0]},
            'calibrated_sensor': {
                'translation': cam_trans,
                'rotation': cam_rot,
                'camera_intrinsic': camera_intrinsic,
            },
        }

        # 雷达元数据
        info['lidar_infos']['LIDAR_TOP'] = {
            'sample_token': info['sample_token'],
            'timestamp': info['timestamp'],
            'filename': f"radar_bev_filter/radar_{frame_idx:06d}.pcd.bin",
            'ego_pose': {'translation': [0.0, 0.0, 0.0],
                         'rotation': [1.0, 0.0, 0.0, 0.0]},
            'calibrated_sensor': {'translation': [0.0, 0.0, 0.0],
                                  'rotation': [1.0, 0.0, 0.0, 0.0]},
        }

        # GT 标注
        ann_infos = []
        for obj in data.get('objects', []):
            cat_name = CATEGORY_MAP.get(obj.get('category', '').lower(), 'vehicle.car')
            vel_in = obj.get('velocity', [0.0, 0.0, 0.0])
            l, w, h = obj.get('size', [1.0, 1.0, 1.0])
            raw_trans = obj.get('translation', [0.0, 0.0, 0.0])
            raw_yaw = obj.get('rotation', 0.0)

            ann = {
                'token': obj.get('token', f"token_{np.random.randint(10000)}"),
                'category_name': cat_name,
                'translation': [raw_trans[0], raw_trans[1], raw_trans[2] + h / 2.0],
                'size': [l, w, h],
                'rotation': _yaw_to_quaternion(raw_yaw),
                'velocity': [float(vel_in[0]) if len(vel_in) > 0 else 0.0,
                             float(vel_in[1]) if len(vel_in) > 1 else 0.0,
                             float(vel_in[2]) if len(vel_in) > 2 else 0.0],
                'num_lidar_pts': 10,
                'num_radar_pts': 10,
            }
            ann_infos.append(ann)

        info['ann_infos'] = ann_infos
        infos_list.append(info)

    # ---- 场景分布统计 ----
    print(f"\n    场景分布:")
    for sc, cnt in sorted(scenario_counts.items()):
        print(f"      {sc}: {cnt} 帧")

    # ---- 训练/验证集划分 ----
    total_frames = len(infos_list)
    split_mode = PipelineConfig.TRAIN_VAL_SPLIT

    if split_mode == "sequential":
        split_idx = int(total_frames * PipelineConfig.SPLIT_RATIO)
        train_infos = infos_list[:split_idx]
        val_infos = infos_list[split_idx:]
    elif split_mode == "range_based":
        lo, hi = PipelineConfig.VAL_FRAME_RANGE
        train_infos = infos_list
        val_infos = [info for info in infos_list
                     if lo <= int(info['sample_token'].split('_')[-1]) < hi]
    elif split_mode == "per_scenario":
        # 每个场景内按帧号排序，留固定 VAL_FRAMES_PER_SCENARIO 帧作为验证
        from collections import defaultdict
        scenario_groups = defaultdict(list)
        for info in infos_list:
            sc_name, _ = get_scenario_by_frame_idx(
                int(info['sample_token'].split('_')[-1]))
            scenario_groups[sc_name or 'unknown'].append(info)

        train_infos, val_infos = [], []
        val_count = PipelineConfig.VAL_FRAMES_PER_SCENARIO
        for sc_name in sorted(scenario_groups.keys()):
            frames = sorted(scenario_groups[sc_name],
                            key=lambda x: int(x['sample_token'].split('_')[-1]))
            n_val = min(val_count, len(frames))
            train_infos.extend(frames[:-n_val])
            val_infos.extend(frames[-n_val:])
            print(f"    {sc_name}: {len(frames)} 帧 -> train={len(frames)-n_val}, val={n_val}")
    elif split_mode == "train_test":
        # 按 offset 严格划分训练/测试（用于泛化验证）
        train_infos, test_infos = [], []
        from collections import defaultdict
        offset_counts = defaultdict(lambda: {'train': 0, 'test': 0})

        for info in infos_list:
            frame_idx = int(info['sample_token'].split('_')[-1])
            sc_name, offset = get_scenario_by_frame_idx(frame_idx)
            sc_name = sc_name or 'unknown'

            split_map = PipelineConfig.TRAIN_TEST_OFFSETS.get(sc_name, {})
            if offset in split_map.get('train', []):
                train_infos.append(info)
                offset_counts[offset]['train'] += 1
            elif offset in split_map.get('test', []):
                test_infos.append(info)
                offset_counts[offset]['test'] += 1

        val_infos = test_infos  # 兼容后续代码

        print(f"\n    训练/测试集分布（按 offset）:")
        for offset in sorted(offset_counts.keys()):
            tc = offset_counts[offset]['train']
            tec = offset_counts[offset]['test']
            if tc > 0:
                print(f"      offset {offset}: train={tc}")
            if tec > 0:
                print(f"      offset {offset}: test={tec}")
    else:
        train_infos = infos_list
        val_infos = infos_list

    import pickle

    # 写全局合并 PKL
    train_pkl_path = os.path.join(PathConfig.DATA_ROOT, PathConfig.TRAIN_PKL)
    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train_infos, f)

    if split_mode == "train_test":
        test_pkl_path = os.path.join(PathConfig.DATA_ROOT, "nuscenes_infos_test.pkl")
        with open(test_pkl_path, 'wb') as f:
            pickle.dump(val_infos, f)
    else:
        val_pkl_path = os.path.join(PathConfig.DATA_ROOT, PathConfig.VAL_PKL)
        with open(val_pkl_path, 'wb') as f:
            pickle.dump(val_infos, f)

    # 写单场景 PKL
    per_scenario_pkls = {}
    for sc_name in ['frontal', 'oblique']:
        sc_train = [info for info in train_infos
                    if info['scene_name'].endswith(sc_name)]
        sc_val = [info for info in val_infos
                   if info['scene_name'].endswith(sc_name)]
        sc_train_pkl = os.path.join(PathConfig.DATA_ROOT, f"nuscenes_infos_{sc_name}_train.pkl")
        if split_mode == "train_test":
            sc_val_pkl = os.path.join(PathConfig.DATA_ROOT, f"nuscenes_infos_{sc_name}_test.pkl")
            val_label = "test"
        else:
            sc_val_pkl = os.path.join(PathConfig.DATA_ROOT, f"nuscenes_infos_{sc_name}_val.pkl")
            val_label = "val"
        with open(sc_train_pkl, 'wb') as f:
            pickle.dump(sc_train, f)
        with open(sc_val_pkl, 'wb') as f:
            pickle.dump(sc_val, f)
        per_scenario_pkls[sc_name] = (sc_train_pkl, sc_val_pkl, len(sc_train), len(sc_val))

    print(f"\n[[OK]] 阶段4 完成！")
    print(f"    总数据量: {total_frames} 帧")
    print(f"    划分模式: {split_mode}")
    print(f"    全局 — 训练集: {len(train_infos)} 帧 -> {PathConfig.TRAIN_PKL}")
    if split_mode == "train_test":
        print(f"    全局 — 测试集: {len(val_infos)} 帧 -> nuscenes_infos_test.pkl")
    else:
        print(f"    全局 — 验证集: {len(val_infos)} 帧 -> {PathConfig.VAL_PKL}")
    for sc_name, (tp, vp, nt, nv) in per_scenario_pkls.items():
        label = "test" if split_mode == "train_test" else "val"
        print(f"    {sc_name} — train={nt} -> {os.path.basename(tp)}")
        print(f"    {sc_name} — {label}={nv}   -> {os.path.basename(vp)}")


# ══════════════════════════════════════════════════════════════════════════════
#  配置展示
# ══════════════════════════════════════════════════════════════════════════════
def show_config():
    """打印当前配置"""
    print("=" * 65)
    print("CRN 数据预处理管线 — 当前配置")
    print("=" * 65)

    print(f"\n[场景预设]")
    for name in ScenarioConfig._ALL:
        s = ScenarioConfig._ALL[name]
        print(f"  {name}:  radar_h={s['radar_h']}m, cam_h={s['cam_h']}m, yaw={s['yaw_deg']}°")

    print(f"\n[批次映射 (OFFSET -> 场景)]")
    for offset in sorted(OFFSET_SCENARIO_MAP.keys()):
        print(f"  {offset:>8d}  ->  {OFFSET_SCENARIO_MAP[offset]}")

    print(f"\n[传感器]")
    print(f"  CAMERA_INTRINSIC = fx={SensorConfig.CAMERA_INTRINSIC[0,0]:.2f}, "
          f"fy={SensorConfig.CAMERA_INTRINSIC[1,1]:.2f}, "
          f"cx={SensorConfig.CAMERA_INTRINSIC[0,2]}, "
          f"cy={SensorConfig.CAMERA_INTRINSIC[1,2]}")
    print(f"  IMAGE            = {SensorConfig.IMAGE_WIDTH}x{SensorConfig.IMAGE_HEIGHT}")

    print(f"\n[路径]")
    print(f"  DATA_ROOT         = {PathConfig.DATA_ROOT}")

    print(f"\n[管线]")
    print(f"  TRAIN_VAL_SPLIT   = {PipelineConfig.TRAIN_VAL_SPLIT}")
    print(f"  DEBUG_VIS_FRAME   = {PipelineConfig.DEBUG_VIS_FRAME}")

    print(f"\n[类别映射]")
    for k, v in CATEGORY_MAP.items():
        print(f"  {k:<20} -> {v}")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="CRN 数据预处理统一管线 (v2 — 多场景)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 正视场景
  python pipeline.py --scenario frontal --offset 0 --csv data/radar_data/S1s_flag.csv --stage radar2bin

  # 斜视场景（安装高度18m，yaw=15°）
  python pipeline.py --scenario oblique --offset 400000 --csv data/radar_data/oblique_flag.csv --stage radar2bin

  # 生成衍生特征（自动识别所有场景，无需 --scenario）
  python pipeline.py --stage derivatives

  # 生成深度真值（自动识别所有场景）
  python pipeline.py --stage depth_gt

  # 构建 PKL（自动合并所有场景）
  python pipeline.py --stage build_infos

  # 查看配置
  python pipeline.py --show-config
        """,
    )
    parser.add_argument('--scenario', type=str, default='frontal',
                        choices=list(ScenarioConfig._ALL.keys()),
                        help='场景名（决定 H 和 yaw），默认: frontal')
    parser.add_argument('--stage', type=str, default=None,
                        choices=['radar2bin', 'merge_new_data', 'derivatives',
                                 'depth_gt', 'build_infos'],
                        help='要运行的阶段')
    parser.add_argument('--csv', type=str, default=None,
                        help='雷达 CSV 源路径 (radar2bin 阶段必需)')
    parser.add_argument('--offset', type=int, default=None,
                        help='帧号偏移量 (radar2bin 和 merge_new_data 阶段必需)')
    parser.add_argument('--show-config', action='store_true',
                        help='打印当前配置并退出')

    args = parser.parse_args()

    if args.show_config:
        show_config()
        return

    if args.stage is None:
        parser.print_help()
        print("\n提示: 使用 --stage 指定阶段，或 --help 查看更多选项。")
        return

    # 分发
    if args.stage == 'radar2bin':
        stage_radar_csv_to_bin(scenario_name=args.scenario,
                               csv_path=args.csv, offset=args.offset)
    elif args.stage == 'merge_new_data':
        stage_merge_new_data(offset=args.offset)
    elif args.stage == 'derivatives':
        stage_generate_derivatives()
    elif args.stage == 'depth_gt':
        stage_build_depth_gt()
    elif args.stage == 'build_infos':
        stage_build_infos()


if __name__ == "__main__":
    main()
