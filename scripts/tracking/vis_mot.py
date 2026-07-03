"""
跟踪结果可视化
==============
把 tracks_*.csv 中的跟踪框画到原图上，不同 track_id 用不同颜色，
并在框旁边标注 ID。
"""
import os
import csv
import pickle
import numpy as np
import cv2
from collections import defaultdict
from pyquaternion import Quaternion


# ==============================================================================
# 1. 工具函数
# ==============================================================================
def get_pred_box_corners(box):
    x, y, z, w, l, h, yaw = box[:7]
    x_corners = [-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]
    y_corners = [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2]
    z_corners = [0, 0, 0, 0, h, h, h, h]
    corners = np.vstack([x_corners, y_corners, z_corners])

    yaw_calibrated = yaw - (np.pi / 2.0)
    rot_mat = np.array([
        [np.cos(yaw_calibrated), -np.sin(yaw_calibrated), 0],
        [np.sin(yaw_calibrated),  np.cos(yaw_calibrated), 0],
        [0,                      0,                      1]
    ])
    corners = np.dot(rot_mat, corners)
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    return corners


def draw_3d_box(img, corners_2d, color, thickness=2):
    pts = corners_2d[:, :2].astype(int)
    lines = [[0,1], [1,2], [2,3], [3,0],
             [4,5], [5,6], [6,7], [7,4],
             [0,4], [1,5], [2,6], [3,7]]
    for line in lines:
        pt1 = tuple(pts[line[0]])
        pt2 = tuple(pts[line[1]])
        cv2.line(img, pt1, pt2, color, thickness)


def get_color(track_id):
    """为每个 track_id 生成一个鲜艳的颜色"""
    hue = (track_id * 47) % 180  # 47 是质数，能让相邻 ID 色差大
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ==============================================================================
# 2. 主可视化流程
# ==============================================================================
def vis_tracks(tracks_csv, pkl_path, img_root, output_dir, max_frames=999999):
    print(f">>> 可视化: {tracks_csv}")
    os.makedirs(output_dir, exist_ok=True)

    # 读取跟踪结果
    with open(tracks_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    tracks_by_frame = defaultdict(list)
    for r in rows:
        fid = int(r['frame_id'])
        tracks_by_frame[fid].append(r)

    # 读取 PKL 获取相机参数
    with open(pkl_path, 'rb') as f:
        infos = pickle.load(f)

    # 建立 sample_token -> info 映射
    info_map = {}
    for info in infos:
        token = info['sample_token']
        frame_id = int(token.split('_')[-1])
        info_map[frame_id] = info

    sorted_fids = sorted(tracks_by_frame.keys())[:max_frames]
    print(f"    共 {len(sorted_fids)} 帧需要可视化")

    for fid in sorted_fids:
        info = info_map.get(fid)
        if info is None:
            continue

        cam_info = info['cam_infos']['CAM_FRONT']
        img_path = os.path.join(img_root, cam_info['filename'])
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        K = np.array(cam_info['calibrated_sensor']['camera_intrinsic'])
        quat = Quaternion(cam_info['calibrated_sensor']['rotation'])
        trans = np.array(cam_info['calibrated_sensor']['translation'])
        cam2world = np.eye(4)
        cam2world[:3, :3] = quat.rotation_matrix
        cam2world[:3, 3] = trans
        ego2cam = np.linalg.inv(cam2world)

        for r in tracks_by_frame[fid]:
            x, y, z = float(r['x']), float(r['y']), float(r['z'])
            w, l, h = float(r['w']), float(r['l']), float(r['h'])
            yaw = float(r['yaw'])
            tid = int(r['track_id'])
            score = float(r['score'])

            # 跳过低分框（可选）
            if score < 0.35:
                continue

            box = [x, y, z, w, l, h, yaw]
            corners_world = get_pred_box_corners(box)
            corners_world_homo = np.vstack([corners_world, np.ones((1, 8))])
            corners_cam = ego2cam @ corners_world_homo
            if np.any(corners_cam[2, :] > 0.1):
                corners_2d = (K @ corners_cam[:3, :] / corners_cam[2, :]).T
                color = get_color(tid)
                draw_3d_box(img, corners_2d, color, thickness=2)
                # 在框的左上角标注 ID
                pt_text = tuple(corners_2d[0][:2].astype(int))
                cv2.putText(img, f"ID{tid}", pt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_path = os.path.join(output_dir, f"track_{fid:06d}.jpg")
        cv2.imwrite(out_path, img)

    print(f"    完成，已保存到: {output_dir}")


# ==============================================================================
# 3. 入口
# ==============================================================================
if __name__ == '__main__':
    BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'work_dirs')
    DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'my_formatted_data')

    # Frontal
    vis_tracks(
        os.path.join(BASE, 'model_predict_results_frontal_40epoch', 'tracks_1.csv'),
        os.path.join(DATA, 'nuscenes_infos_frontal_test.pkl'),
        DATA,
        os.path.join(BASE, 'model_predict_results_frontal_40epoch', 'vis_tracks')
    )

    # Oblique
    vis_tracks(
        os.path.join(BASE, 'model_predict_results_oblique_240', 'tracks_2.csv'),
        os.path.join(DATA, 'nuscenes_infos_oblique_test.pkl'),
        DATA,
        os.path.join(BASE, 'model_predict_results_oblique_240', 'vis_tracks')
    )
