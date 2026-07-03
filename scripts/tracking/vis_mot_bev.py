"""
跟踪结果 BEV (鸟瞰图) 2D 可视化
=================================
只关注 x, y 平面，在 BEV 上画带 track_id 的矩形框。

坐标映射:
  x: [0, 240m]   -> 横向
  y: [-25.6, 25.6m] -> 纵向
"""
import os
import csv
import numpy as np
import cv2
from collections import defaultdict


# ==============================================================================
# 1. BEV 画布配置
# ==============================================================================
X_MIN, X_MAX = 0.0, 240.0
Y_MIN, Y_MAX = -25.6, 25.6
SCALE = 5  # pixels per meter
W = int((X_MAX - X_MIN) * SCALE)
H = int((Y_MAX - Y_MIN) * SCALE)
ORIGIN_X = 0
ORIGIN_Y = H // 2


def world_to_bev(x, y):
    """世界坐标 (x, y) -> BEV 图像坐标 (u, v)"""
    u = int((x - X_MIN) * SCALE)
    v = int(ORIGIN_Y - y * SCALE)
    return u, v


def get_color(track_id):
    hue = (track_id * 47) % 180
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ==============================================================================
# 2. 绘制工具
# ==============================================================================
def draw_bev_box(img, x, y, w, l, yaw, color, thickness=2):
    """
    在 BEV 上画一个旋转矩形。
    注意: yaw 是模型原始输出，这里使用和 3D 可视化一致的校准方式。
    """
    # 与 model_inference_vis2.py 保持一致
    yaw_calibrated = yaw - (np.pi / 2.0)

    # 本地角点 (半长 l/2, 半宽 w/2)
    dx = l / 2.0
    dy = w / 2.0
    local_corners = np.array([
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy],
        [-dx,  dy],
    ])

    # 旋转矩阵
    cos_y = np.cos(yaw_calibrated)
    sin_y = np.sin(yaw_calibrated)
    rot = np.array([[cos_y, -sin_y],
                    [sin_y,  cos_y]])

    rotated = (rot @ local_corners.T).T
    rotated[:, 0] += x
    rotated[:, 1] += y

    # 映射到像素
    pts = np.array([world_to_bev(px, py) for px, py in rotated], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    return pts[0]  # 返回第一个角点，用于标注文字


def draw_grid(img):
    """画 BEV 网格和刻度"""
    # 竖线 (x 方向，每 20m)
    for xm in range(int(X_MIN), int(X_MAX) + 1, 20):
        u, _ = world_to_bev(xm, 0)
        cv2.line(img, (u, 0), (u, H), (50, 50, 50), 1)
        cv2.putText(img, f"{xm}m", (u + 2, H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # 横线 (y 方向，每 10m)
    for ym in range(int(Y_MIN), int(Y_MAX) + 1, 10):
        _, v = world_to_bev(0, ym)
        cv2.line(img, (0, v), (W, v), (50, 50, 50), 1)
        if ym != 0:
            cv2.putText(img, f"{ym}m", (5, v - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # 中心线 y=0
    _, v0 = world_to_bev(0, 0)
    cv2.line(img, (0, v0), (W, v0), (80, 80, 80), 2)


# ==============================================================================
# 3. 主可视化流程
# ==============================================================================
def vis_bev(tracks_csv, output_dir, max_frames=999999):
    print(f">>> BEV 可视化: {tracks_csv}")
    os.makedirs(output_dir, exist_ok=True)

    with open(tracks_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    tracks_by_frame = defaultdict(list)
    for r in rows:
        fid = int(r['frame_id'])
        tracks_by_frame[fid].append(r)

    sorted_fids = sorted(tracks_by_frame.keys())[:max_frames]
    print(f"    共 {len(sorted_fids)} 帧")

    for fid in sorted_fids:
        img = np.zeros((H, W, 3), dtype=np.uint8)
        draw_grid(img)

        # 可选：画标题
        cv2.putText(img, f"Frame {fid}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        for r in tracks_by_frame[fid]:
            score = float(r['score'])
            if score < 0.35:
                continue
            x = float(r['x'])
            y = float(r['y'])
            w = float(r['w'])
            l = float(r['l'])
            yaw = float(r['yaw'])
            tid = int(r['track_id'])

            color = get_color(tid)
            pt_text = draw_bev_box(img, x, y, w, l, yaw, color, thickness=2)
            cv2.putText(img, f"ID{tid}", tuple(pt_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out_path = os.path.join(output_dir, f"bev_{fid:06d}.jpg")
        cv2.imwrite(out_path, img)

    print(f"    完成，已保存到: {output_dir}")


# ==============================================================================
# 4. 入口
# ==============================================================================
if __name__ == '__main__':
    BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'work_dirs')

    vis_bev(
        os.path.join(BASE, 'model_predict_results_frontal_40epoch', 'tracks_1.csv'),
        os.path.join(BASE, 'model_predict_results_frontal_40epoch', 'vis_bev')
    )
    vis_bev(
        os.path.join(BASE, 'model_predict_results_oblique_240', 'tracks_2.csv'),
        os.path.join(BASE, 'model_predict_results_oblique_240', 'vis_bev')
    )
