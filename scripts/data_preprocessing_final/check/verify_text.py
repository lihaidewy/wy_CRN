#!/usr/bin/env python
"""
CRN 数据预处理 - 纯文本验证（无需 cv2）
=========================================
检查 PKL 外参、标注框投影、雷达深度分布是否合理。

用法:
    python scripts/data_preprocessing_final/check/verify_text.py
    python scripts/data_preprocessing_final/check/verify_text.py --frame 1
"""

import os
import sys
import pickle
import argparse
import numpy as np

DATA_ROOT = "./data/my_formatted_data"
PKL_PATH = os.path.join(DATA_ROOT, "nuscenes_infos_train.pkl")
DEPTH_DIR = os.path.join(DATA_ROOT, "depth_gt")


def load_pkl():
    with open(PKL_PATH, 'rb') as f:
        infos = pickle.load(f)
    print(f"[OK] PKL: {len(infos)} 帧")
    return infos


def get_frame_idx(info):
    return int(info['sample_token'].split('_')[-1])


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y],
    ])


def build_world2cam(info):
    calib = info['cam_infos']['CAM_FRONT']['calibrated_sensor']
    trans = np.array(calib['translation'])
    rot = calib['rotation']
    R = quat_to_R(rot)
    cam2world = np.eye(4)
    cam2world[:3, :3] = R
    cam2world[:3, 3] = trans
    return np.linalg.inv(cam2world)


def get_box_corners(size, trans, rot_q):
    l, w, h = size
    corners = np.array([
        [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2],
        [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2],
        [-h/2, -h/2, -h/2, -h/2,  h/2,  h/2,  h/2,  h/2],
    ])
    R = quat_to_R(rot_q)
    corners = R @ corners
    corners[0, :] += trans[0]
    corners[1, :] += trans[1]
    corners[2, :] += trans[2]
    return corners


def project_corners(corners_3d, world2cam, K, img_w, img_h):
    homo = np.vstack([corners_3d, np.ones((1, 8))])
    cam = world2cam @ homo
    z = cam[2, :]
    if np.all(z <= 0.1):
        return None, None
    proj = K @ cam[:3, :]
    uv = proj[:2, :] / proj[2, :]
    uv = uv.T
    in_view = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    return uv, in_view


def load_depth_gt(frame_idx):
    path = os.path.join(DEPTH_DIR, f"{frame_idx}.png.bin")
    if not os.path.exists(path):
        return None
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 3)
    return pts


def verify_frame(info, img_w=3840, img_h=2160):
    frame_idx = get_frame_idx(info)
    K = np.array(info['cam_infos']['CAM_FRONT']['calibrated_sensor']['camera_intrinsic'])
    world2cam = build_world2cam(info)
    cam_trans = info['cam_infos']['CAM_FRONT']['calibrated_sensor']['translation']
    cam_rot = info['cam_infos']['CAM_FRONT']['calibrated_sensor']['rotation']

    print(f"\n{'='*60}")
    print(f"Frame {frame_idx}")
    print(f"  相机外参: trans=[{cam_trans[0]:.2f}, {cam_trans[1]:.2f}, {cam_trans[2]:.2f}]")
    print(f"            rot=[{cam_rot[0]:.4f}, {cam_rot[1]:.4f}, {cam_rot[2]:.4f}, {cam_rot[3]:.4f}]")

    # --- 雷达深度 ---
    depth_pts = load_depth_gt(frame_idx)
    if depth_pts is not None and len(depth_pts) > 0:
        u, v, z = depth_pts[:, 0], depth_pts[:, 1], depth_pts[:, 2]
        in_view = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        print(f"  雷达点数: {len(depth_pts)}  |  深度范围: [{z.min():.1f}, {z.max():.1f}]m  |  平均: {z.mean():.1f}m")
        print(f"  视野内: {in_view.sum()}/{len(depth_pts)} ({in_view.sum()/len(depth_pts)*100:.1f}%)")
        # 检查左右分布
        left = (u < img_w / 2).sum()
        right = (u >= img_w / 2).sum()
        print(f"  左右分布: 左半屏={left}  右半屏={right}")
    else:
        print(f"  [WARN] 无雷达深度数据")

    # --- 3D 框投影 ---
    box_count = 0
    in_view_count = 0
    behind_count = 0
    for ann in info['ann_infos']:
        corners = get_box_corners(ann['size'], ann['translation'], ann['rotation'])
        uv, mask = project_corners(corners, world2cam, K, img_w, img_h)
        if uv is None:
            behind_count += 1
            continue
        box_count += 1
        if np.any(mask):
            in_view_count += 1

    print(f"  标注总数: {len(info['ann_infos'])}")
    print(f"  有效投影: {box_count}  (在视野内: {in_view_count}, 全在相机后: {behind_count})")

    if box_count == 0 and len(info['ann_infos']) > 0:
        print(f"  [ERR] 所有标注框都在相机后方！外参可能有误！")
    elif in_view_count == 0 and box_count > 0:
        print(f"  [WARN] 所有框都投影到视野外，检查外参或场景配置")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=None)
    parser.add_argument('--num', type=int, default=5)
    args = parser.parse_args()

    infos = load_pkl()
    frame_map = {get_frame_idx(i): i for i in infos}

    targets = []
    if args.frame is not None:
        targets = [frame_map.get(args.frame)] if args.frame in frame_map else []
    else:
        targets = list(frame_map.values())[:args.num]

    for info in targets:
        if info is not None:
            verify_frame(info)

    print(f"\n{'='*60}")
    print("[DONE] 文本验证完成。")
    print("提示: 若所有框都在相机后方，说明 cam_rot 仍是 identity [1,0,0,0]")
    print("      若左右分布严重失衡，说明 cam_x 符号可能反了")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
