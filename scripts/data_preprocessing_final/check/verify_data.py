#!/usr/bin/env python
"""
CRN 数据预处理 - 统一验证脚本
==============================
同时检查雷达点云投影和 3D 标注框是否正确对齐。
支持正视和斜视两种场景。

用法:
    # 检查前 5 帧（默认）
    python scripts/data_preprocessing_final/check/verify_data.py

    # 检查指定帧
    python scripts/data_preprocessing_final/check/verify_data.py --frame 400001

    # 检查斜视场景的前 5 帧
    python scripts/data_preprocessing_final/check/verify_data.py --scenario oblique --num 5

    # 逐帧交互模式
    python scripts/data_preprocessing_final/check/verify_data.py --interactive
"""

import os
import sys
import pickle
import argparse
import numpy as np
import cv2

# ── 配置 ───────────────────────────────────────────────────────
DATA_ROOT = "./data/my_formatted_data"
PKL_PATH = os.path.join(DATA_ROOT, "nuscenes_infos_train.pkl")
IMG_DIR = os.path.join(DATA_ROOT, "samples/CAM_FRONT")
DEPTH_DIR = os.path.join(DATA_ROOT, "depth_gt")
OUTPUT_DIR = "work_dirs/verify_data"
# ────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_pkl():
    """加载 PKL，返回帧索引列表"""
    with open(PKL_PATH, 'rb') as f:
        infos = pickle.load(f)
    print(f"[OK] 加载 PKL: {len(infos)} 帧")
    return infos


def get_frame_idx(info):
    """从 info 中提取帧号"""
    return int(info['sample_token'].split('_')[-1])


def get_scenario_name(info):
    """从 scene_name 提取场景名"""
    return info.get('scene_name', '').replace('roadside_', '')


def build_cam2world_matrix(info):
    """从 info 构建 camera→world 变换矩阵"""
    calib = info['cam_infos']['CAM_FRONT']['calibrated_sensor']
    trans = np.array(calib['translation'])
    rot = calib['rotation']  # [w, x, y, z]

    # 四元数→旋转矩阵
    w, x, y, z = rot
    R = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
    ])

    cam2world = np.eye(4)
    cam2world[:3, :3] = R
    cam2world[:3, 3] = trans
    return cam2world


def get_3d_box_corners(size, translation, rotation_quat):
    """3D 框 8 个顶点 (世界坐标系)"""
    l, w, h = size
    corners = np.array([
        [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2],
        [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2],
        [-h/2, -h/2, -h/2, -h/2,  h/2,  h/2,  h/2,  h/2],
    ])

    # 四元数→旋转矩阵
    w, x, y, z = rotation_quat
    R = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
    ])

    corners = R @ corners
    corners[0, :] += translation[0]
    corners[1, :] += translation[1]
    corners[2, :] += translation[2]
    return corners  # (3, 8)


def project_box_to_image(corners_3d, world2cam, K):
    """世界坐标 3D 框顶点→像素坐标 (8, 2)"""
    corners_homo = np.vstack([corners_3d, np.ones((1, 8))])
    corners_cam = world2cam @ corners_homo
    z = corners_cam[2, :]

    if np.all(z <= 0.1):
        return None

    corners_2d = K @ corners_cam[:3, :]
    corners_2d = corners_2d[:2, :] / corners_2d[2, :]
    return corners_2d.T.astype(int)  # (8, 2)


def draw_3d_box(img, corners_2d, label, color=(0, 255, 0)):
    """在图片上绘制 3D 框12条边"""
    lines = [
        (0,1),(1,2),(2,3),(3,0),     # 底面
        (4,5),(5,6),(6,7),(7,4),     # 顶面
        (0,4),(1,5),(2,6),(3,7),     # 立柱
    ]
    for a, b in lines:
        p1 = tuple(corners_2d[a])
        p2 = tuple(corners_2d[b])
        cv2.line(img, p1, p2, color, 2)
    # 类别标签
    cv2.putText(img, label, tuple(corners_2d[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def load_depth_gt(frame_idx):
    """加载深度真值，返回 (u, v, depth)"""
    dp = os.path.join(DEPTH_DIR, f"{frame_idx}.png.bin")
    if not os.path.exists(dp):
        return None
    pts = np.fromfile(dp, dtype=np.float32).reshape(-1, 3)
    return pts


def verify_frame(info, draw_radar=True, draw_boxes=True, save=True):
    """验证单帧：雷达投影 + 3D 标注框"""
    frame_idx = get_frame_idx(info)
    scenario = get_scenario_name(info)

    # 加载图片
    img_path = os.path.join(IMG_DIR, f"{frame_idx}.png")
    if not os.path.exists(img_path):
        # 尝试 jpg
        img_path = os.path.join(IMG_DIR, f"{frame_idx}.jpg")
    if not os.path.exists(img_path):
        print(f"  [SKIP] 找不到图片 {frame_idx}.png")
        return None

    img = cv2.imread(img_path)
    if img is None:
        print(f"  [SKIP] 无法读取图片 {img_path}")
        return None

    H_img, W_img = img.shape[:2]
    overlay = img.copy()

    # ── 相机参数 ──
    calib = info['cam_infos']['CAM_FRONT']['calibrated_sensor']
    K = np.array(calib['camera_intrinsic'])
    cam2world = build_cam2world_matrix(info)
    world2cam = np.linalg.inv(cam2world)

    cam_trans = calib['translation']
    cam_rot = calib['rotation']

    print(f"\n{'='*60}")
    print(f"Frame {frame_idx}  |  场景: {scenario}")
    print(f"  相机外参: trans={[f'{v:.2f}' for v in cam_trans]}")
    print(f"            rot=[{cam_rot[0]:.4f}, {cam_rot[1]:.4f}, {cam_rot[2]:.4f}, {cam_rot[3]:.4f}]")
    print(f"  标注数量: {len(info['ann_infos'])}")
    print(f"  图片尺寸: {W_img}x{H_img}")

    # ── 1. 雷达点云投影 ──
    radar_stats = None
    if draw_radar:
        depth_pts = load_depth_gt(frame_idx)
        if depth_pts is not None and len(depth_pts) > 0:
            u_pts = depth_pts[:, 0].astype(int)
            v_pts = depth_pts[:, 1].astype(int)
            z_pts = depth_pts[:, 2]

            z_min, z_max = z_pts.min(), z_pts.max()
            z_mean = z_pts.mean()
            in_view = (u_pts >= 0) & (u_pts < W_img) & (v_pts >= 0) & (v_pts < H_img)
            in_view_pct = in_view.sum() / len(in_view) * 100

            radar_stats = {
                'num_pts': len(depth_pts),
                'depth_range': (z_min, z_max),
                'depth_mean': z_mean,
                'in_view_pct': in_view_pct,
            }
            print(f"  雷达点数: {len(depth_pts)}  |  深度范围: [{z_min:.1f}, {z_max:.1f}]m")
            print(f"  视野内: {in_view.sum()}/{len(depth_pts)} ({in_view_pct:.1f}%)")

            for i in range(len(z_pts)):
                if not in_view[i]:
                    continue
                ratio = min(z_pts[i] / 160.0, 1.0)
                color = (0, int(255 * (1 - ratio)), int(255 * ratio))
                cv2.circle(overlay, (u_pts[i], v_pts[i]), 4, color, -1)
        else:
            print(f"  [WARN] 无雷达深度数据")

    # ── 2. 3D 标注框投影 ──
    box_count = 0
    if draw_boxes:
        for ann in info['ann_infos']:
            corners_w = get_3d_box_corners(ann['size'], ann['translation'], ann['rotation'])
            corners_2d = project_box_to_image(corners_w, world2cam, K)
            if corners_2d is not None:
                # 检查是否在视野内
                if np.all(corners_2d[:, 0] >= 0) and np.all(corners_2d[:, 0] < W_img) and \
                   np.all(corners_2d[:, 1] >= 0) and np.all(corners_2d[:, 1] < H_img):
                    color = (0, 255, 0)
                else:
                    color = (0, 165, 255)  # 橙色=部分在视野外
                draw_3d_box(overlay, corners_2d, ann['category_name'], color)
                box_count += 1
        print(f"  已绘制: {box_count} 个 3D 框")

    # ── 混合 ──
    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

    # ── 写入统计信息到图片 ──
    y_offset = 30
    cv2.putText(blended, f"Frame: {frame_idx}  Scenario: {scenario}",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 35
    if radar_stats:
        cv2.putText(blended,
                    f"Radar: {radar_stats['num_pts']} pts, depth=[{radar_stats['depth_range'][0]:.1f}~{radar_stats['depth_range'][1]:.1f}]m",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
    cv2.putText(blended, f"Ann boxes: {box_count}",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 30
    cv2.putText(blended, f"cam_trans: [{cam_trans[0]:.1f}, {cam_trans[1]:.1f}, {cam_trans[2]:.1f}]",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── 图例 ──
    legend_y = H_img - 80
    cv2.rectangle(blended, (20, legend_y), (350, H_img - 10), (0, 0, 0), -1)
    cv2.putText(blended, "Radar: near=green  far=red", (30, legend_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(blended, "Boxes: green=in-view  orange=partial", (30, legend_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if save:
        out_path = os.path.join(OUTPUT_DIR, f"verify_{frame_idx}.jpg")
        cv2.imwrite(out_path, blended)
        print(f"  -> 已保存: {out_path}")

    return blended


def find_frames_by_scenario(infos, scenario_name):
    """按场景筛选帧"""
    result = []
    for info in infos:
        if get_scenario_name(info) == scenario_name:
            result.append(info)
    return result


def main():
    parser = argparse.ArgumentParser(description="CRN 数据预处理 — 统一验证")
    parser.add_argument('--frame', type=int, default=None,
                        help='验证指定帧号')
    parser.add_argument('--scenario', type=str, default=None,
                        choices=['frontal', 'oblique'],
                        help='只验证指定场景的帧')
    parser.add_argument('--num', type=int, default=5,
                        help='验证帧数（默认 5）')
    parser.add_argument('--no-radar', action='store_true',
                        help='不绘制雷达点云')
    parser.add_argument('--no-boxes', action='store_true',
                        help='不绘制 3D 标注框')
    parser.add_argument('--interactive', action='store_true',
                        help='逐帧交互模式（按任意键下一帧，q退出）')
    parser.add_argument('--range', type=int, nargs=2, metavar=('START', 'END'),
                        help='验证连续帧号范围，如 --range 100 110')
    args = parser.parse_args()

    print("=" * 60)
    print("CRN 数据预处理 — 统一验证")
    print("=" * 60)

    infos = load_pkl()

    # 构建帧号→info 映射
    frame_map = {get_frame_idx(info): info for info in infos}

    # 确定要验证的帧
    target_infos = []

    if args.range is not None:
        start, end = args.range
        missing = []
        for fid in range(start, end + 1):
            if fid in frame_map:
                target_infos.append(frame_map[fid])
            else:
                missing.append(fid)
        if missing:
            print(f"[WARN] 以下帧不在 PKL 中，已跳过: {missing}")
        print(f"[OK] 范围 [{start}, {end}]: 实际验证 {len(target_infos)} 帧")
    elif args.frame is not None:
        if args.frame in frame_map:
            target_infos = [frame_map[args.frame]]
        else:
            print(f"[ERR] 帧 {args.frame} 不在 PKL 中")
            print(f"  可用帧范围: {min(frame_map.keys())} ~ {max(frame_map.keys())}")
            return
    elif args.scenario is not None:
        matched = find_frames_by_scenario(infos, args.scenario)
        print(f"[OK] 场景 '{args.scenario}': {len(matched)} 帧")
        target_infos = matched[:args.num]
    else:
        # 默认：取各场景的前几帧
        for sc in ['frontal', 'oblique']:
            matched = find_frames_by_scenario(infos, sc)
            n = min(args.num // 2 + 1, len(matched))
            target_infos.extend(matched[:n])

    if not target_infos:
        print("[ERR] 没有找到匹配的帧")
        return

    print(f"[OK] 将验证 {len(target_infos)} 帧\n")

    if args.interactive:
        # 交互模式
        for info in target_infos:
            print(f"\n--- 按任意键继续，q 退出 ---")
            key = input().strip().lower()
            if key == 'q':
                break
            verify_frame(info, draw_radar=not args.no_radar,
                        draw_boxes=not args.no_boxes)
    else:
        for info in target_infos:
            verify_frame(info, draw_radar=not args.no_radar,
                        draw_boxes=not args.no_boxes)

    print(f"\n{'='*60}")
    print(f"[DONE] 验证完成！结果保存在: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
