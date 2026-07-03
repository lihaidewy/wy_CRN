"""
跟踪结果 BEV 轨迹拖尾可视化 (输出 MP4 视频)
=============================================
在同一 BEV 画布上：
  - 保留每个 track_id 的历史中心点轨迹（拖尾）
  - 画当前帧的检测框
  - 输出为 MP4 视频
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
SCALE = 5
W = int((X_MAX - X_MIN) * SCALE)
H = int((Y_MAX - Y_MIN) * SCALE)
ORIGIN_X = 0
ORIGIN_Y = H // 2
MAX_TRAIL_LEN = 30  # 轨迹保留最近多少帧


def world_to_bev(x, y):
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
    yaw_calibrated = yaw - (np.pi / 2.0)
    dx = l / 2.0
    dy = w / 2.0
    local_corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    cos_y = np.cos(yaw_calibrated)
    sin_y = np.sin(yaw_calibrated)
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    rotated = (rot @ local_corners.T).T
    rotated[:, 0] += x
    rotated[:, 1] += y
    pts = np.array([world_to_bev(px, py) for px, py in rotated], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    return pts[0]


def draw_grid(img):
    for xm in range(int(X_MIN), int(X_MAX) + 1, 20):
        u, _ = world_to_bev(xm, 0)
        cv2.line(img, (u, 0), (u, H), (50, 50, 50), 1)
        cv2.putText(img, f"{xm}m", (u + 2, H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    for ym in range(int(Y_MIN), int(Y_MAX) + 1, 10):
        _, v = world_to_bev(0, ym)
        cv2.line(img, (0, v), (W, v), (50, 50, 50), 1)
        if ym != 0:
            cv2.putText(img, f"{ym}m", (5, v - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    _, v0 = world_to_bev(0, 0)
    cv2.line(img, (0, v0), (W, v0), (80, 80, 80), 2)


# ==============================================================================
# 3. 主可视化流程 (输出 MP4)
# ==============================================================================
def vis_bev_trail(tracks_csv, output_mp4, fps=10):
    print(f">>> BEV 轨迹可视化: {tracks_csv}")
    print(f"    输出视频: {output_mp4}")

    with open(tracks_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    tracks_by_frame = defaultdict(list)
    for r in rows:
        fid = int(r['frame_id'])
        tracks_by_frame[fid].append(r)

    sorted_fids = sorted(tracks_by_frame.keys())
    print(f"    共 {len(sorted_fids)} 帧")

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_mp4, fourcc, fps, (W, H))
    if not writer.isOpened():
        print(f"[!] 无法初始化 VideoWriter，请检查 OpenCV 编码器支持。")
        return

    # 维护每个 track_id 的历史轨迹
    trails = defaultdict(list)  # tid -> [(x,y), ...]

    for fid in sorted_fids:
        img = np.zeros((H, W, 3), dtype=np.uint8)
        draw_grid(img)
        cv2.putText(img, f"Frame {fid}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # 更新历史轨迹
        for r in tracks_by_frame[fid]:
            tid = int(r['track_id'])
            x = float(r['x'])
            y = float(r['y'])
            trails[tid].append((x, y))
            if len(trails[tid]) > MAX_TRAIL_LEN:
                trails[tid].pop(0)

        # 画历史轨迹线（所有已知 track）
        for tid, pts_world in trails.items():
            if len(pts_world) < 2:
                continue
            color = get_color(tid)
            pts_bev = np.array([world_to_bev(px, py) for px, py in pts_world], dtype=np.int32)
            cv2.polylines(img, [pts_bev], isClosed=False, color=color, thickness=2)
            # 轨迹起点画小圆点
            cv2.circle(img, tuple(pts_bev[0]), 3, color, -1)

        # 画当前帧检测框
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
            # 中心点画圆
            cx, cy = world_to_bev(x, y)
            cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1)

        writer.write(img)

    writer.release()
    print(f"    视频已保存: {output_mp4}")


# ==============================================================================
# 4. 入口
# ==============================================================================
if __name__ == '__main__':
    BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'work_dirs')

    vis_bev_trail(
        os.path.join(BASE, 'model_predict_results_frontal_40epoch', 'tracks_1.csv'),
        os.path.join(BASE, 'model_predict_results_frontal_40epoch', 'bev_trail_frontal.mp4'),
        fps=10
    )
    vis_bev_trail(
        os.path.join(BASE, 'model_predict_results_oblique_240', 'tracks_2.csv'),
        os.path.join(BASE, 'model_predict_results_oblique_240', 'bev_trail_oblique.mp4'),
        fps=10
    )
