"""
多目标跟踪 (MOT) — 基于 3D 检测框的航迹关联
=============================================
输入: preds_*.csv (含 x,y,z,vx,vy)
输出: tracks_*.csv (原列 + track_id)

算法:
  - 简单匀速运动模型 (vx, vy 预测)
  - 匈牙利算法 (scipy) 做帧间数据关联
  - 距离阈值门控 + 航迹生命周期管理
"""
import os
import csv
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


# ==============================================================================
# 1. 参数配置
# ==============================================================================
MAX_AGE = 3               # 航迹最多丢失多少帧后删除
MIN_HITS = 2              # 新航迹至少需要命中多少帧才确认
MATCH_DIST_THRESH = 5.0   # 匹配距离阈值 (米)
DT = 0.1                  # 帧间隔 (秒)，用于运动预测


# ==============================================================================
# 2. 航迹类
# ==============================================================================
class Track:
    _id_counter = 0

    def __init__(self, det):
        """
        det: dict，包含检测框信息
        """
        Track._id_counter += 1
        self.id = Track._id_counter
        self.x = float(det['x'])
        self.y = float(det['y'])
        self.z = float(det['z'])
        self.vx = float(det.get('vx', 0.0))
        self.vy = float(det.get('vy', 0.0))
        self.hits = 1
        self.time_since_update = 0
        self.age = 1
        self.confirmed = False
        # 保存最新检测信息，用于输出
        self.last_det = det

    def predict(self):
        """匀速预测下一帧位置"""
        self.x += self.vx * DT
        self.y += self.vy * DT
        self.age += 1
        self.time_since_update += 1

    def update(self, det):
        """用新检测更新航迹"""
        self.x = float(det['x'])
        self.y = float(det['y'])
        self.z = float(det['z'])
        self.vx = float(det.get('vx', 0.0))
        self.vy = float(det.get('vy', 0.0))
        self.hits += 1
        self.time_since_update = 0
        self.last_det = det
        if self.hits >= MIN_HITS:
            self.confirmed = True

    def get_state(self):
        return np.array([self.x, self.y, self.z])


# ==============================================================================
# 3. 跟踪器
# ==============================================================================
class Tracker:
    def __init__(self):
        self.tracks = []

    def predict(self):
        for t in self.tracks:
            t.predict()

    def update(self, dets):
        """
        dets: list[dict]，当前帧的所有检测框
        """
        if len(self.tracks) == 0 and len(dets) == 0:
            return []

        # 提取航迹预测位置
        track_states = np.array([t.get_state() for t in self.tracks]) if self.tracks else np.zeros((0, 3))
        det_states = np.array([[float(d['x']), float(d['y']), float(d['z'])] for d in dets]) if dets else np.zeros((0, 3))

        n_t = len(self.tracks)
        n_d = len(dets)

        if n_t > 0 and n_d > 0:
            # 构建代价矩阵 (欧氏距离)
            cost = np.zeros((n_t, n_d), dtype=np.float32)
            for i in range(n_t):
                for j in range(n_d):
                    cost[i, j] = np.linalg.norm(track_states[i] - det_states[j])

            row_ind, col_ind = linear_sum_assignment(cost)

            matched_track_idx = set()
            matched_det_idx = set()
            matches = []
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] <= MATCH_DIST_THRESH:
                    matches.append((i, j))
                    matched_track_idx.add(i)
                    matched_det_idx.add(j)
        else:
            matches = []
            matched_track_idx = set()
            matched_det_idx = set()

        # 更新匹配航迹
        for ti, di in matches:
            self.tracks[ti].update(dets[di])

        # 未匹配的检测 -> 初始化新航迹
        for di in range(n_d):
            if di not in matched_det_idx:
                self.tracks.append(Track(dets[di]))

        # 未匹配的航迹 -> 标记丢失
        unmatched_tracks = [i for i in range(n_t) if i not in matched_track_idx]
        # 这里不立即删除，等 predict 阶段自然增加 time_since_update

        # 过滤死亡航迹 (超过 MAX_AGE 未更新)
        self.tracks = [t for t in self.tracks if t.time_since_update <= MAX_AGE]

        # 返回当前帧所有已确认航迹的最新检测（带 track_id）
        results = []
        for t in self.tracks:
            if t.confirmed:
                out = dict(t.last_det)
                out['track_id'] = t.id
                results.append(out)
        return results


# ==============================================================================
# 4. 主流程
# ==============================================================================
def run_mot_on_csv(input_path, output_path):
    print(f">>> 读取检测框: {input_path}")
    # 读取 CSV
    with open(input_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 按 frame_id 分组
    frames = defaultdict(list)
    for r in rows:
        fid = int(r['frame_id'])
        frames[fid].append(r)

    sorted_frame_ids = sorted(frames.keys())
    print(f"    共 {len(sorted_frame_ids)} 帧, {len(rows)} 个检测框")

    tracker = Tracker()
    all_results = []

    for fid in sorted_frame_ids:
        dets = frames[fid]
        tracker.predict()
        results = tracker.update(dets)
        all_results.extend(results)
        # 可选：打印每帧跟踪数
        # print(f"  Frame {fid}: {len(dets)} dets -> {len(results)} tracks")

    print(f"    跟踪完成，共输出 {len(all_results)} 条带 ID 的航迹点")

    # 写输出 CSV
    fieldnames = list(rows[0].keys()) + ['track_id']
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"    结果已保存: {output_path}")


if __name__ == '__main__':
    BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'work_dirs')

    frontal_csv = os.path.join(BASE_DIR, 'model_predict_results_frontal_40epoch', 'preds_1.csv')
    oblique_csv = os.path.join(BASE_DIR, 'model_predict_results_oblique_240', 'preds_2.csv')

    if os.path.exists(frontal_csv):
        run_mot_on_csv(frontal_csv, os.path.join(BASE_DIR, 'model_predict_results_frontal_40epoch', 'tracks_1.csv'))
    else:
        print(f"[!] 找不到 {frontal_csv}")

    if os.path.exists(oblique_csv):
        run_mot_on_csv(oblique_csv, os.path.join(BASE_DIR, 'model_predict_results_oblique_240', 'tracks_2.csv'))
    else:
        print(f"[!] 找不到 {oblique_csv}")
