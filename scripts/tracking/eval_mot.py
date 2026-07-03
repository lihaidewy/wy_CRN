"""
多目标跟踪 (MOT) 评估脚本
==========================
输入:
  - tracks_*.csv (含 track_id 的跟踪结果)
  - nuscenes_infos_*.pkl (真值标注)

输出指标:
  - MOTA, MOTP, Precision, Recall
  - ID Switches
  - MT (Mostly Tracked), ML (Mostly Lost)
"""
import os
import csv
import pickle
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


# ==============================================================================
# 1. 配置
# ==============================================================================
MATCH_DIST_THRESH = 5.0   # 匹配距离阈值 (米)
GT_LINK_DIST = 2.0        # 跨帧 GT 关联距离 (米)


# ==============================================================================
# 2. 读取真值
# ==============================================================================
def load_gt_from_pkl(pkl_path):
    """从 nuScenes info PKL 中提取逐帧真值框"""
    with open(pkl_path, 'rb') as f:
        infos = pickle.load(f)

    gt_by_frame = defaultdict(list)
    for info in infos:
        token = info['sample_token']
        frame_id = int(token.split('_')[-1])
        for ann in info.get('ann_infos', []):
            gx, gy, gz = ann['translation']
            gt_by_frame[frame_id].append({
                'x': gx, 'y': gy, 'z': gz,
            })
    return gt_by_frame


# ==============================================================================
# 3. 读取跟踪结果
# ==============================================================================
def load_tracks_from_csv(csv_path):
    """从 tracks CSV 中提取逐帧跟踪框"""
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    tracks_by_frame = defaultdict(list)
    for r in rows:
        fid = int(r['frame_id'])
        tracks_by_frame[fid].append({
            'x': float(r['x']),
            'y': float(r['y']),
            'z': float(r['z']),
            'track_id': int(r['track_id']),
        })
    return tracks_by_frame


# ==============================================================================
# 4. 逐帧评估 (含 ID Switch)
# ==============================================================================
def evaluate_mot(tracks_by_frame, gt_by_frame):
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    total_dist = 0.0

    # 全局 GT 生命周期统计: gt_uid -> {'appear_frames': int, 'tracked_frames': int}
    gt_life = defaultdict(lambda: {'appear': 0, 'tracked': 0})

    all_frame_ids = sorted(set(list(tracks_by_frame.keys()) + list(gt_by_frame.keys())))

    # 上一帧状态
    prev_gt_uid_list = []      # list of uid
    prev_gt_pos = np.zeros((0, 3))
    prev_gt_tid = {}           # uid -> last matched track_id

    next_gt_uid = 0

    for fid in all_frame_ids:
        trks = tracks_by_frame.get(fid, [])
        gts = gt_by_frame.get(fid, [])

        n_t = len(trks)
        n_g = len(gts)
        total_gt += n_g

        # 当前帧 GT 位置
        cur_pos = np.array([[g['x'], g['y'], g['z']] for g in gts]) if n_g > 0 else np.zeros((0, 3))

        # 跨帧关联 GT：当前 GT 与上一帧 GT 按距离关联
        cur_uid_list = [None] * n_g
        if n_g > 0 and len(prev_gt_uid_list) > 0 and len(prev_gt_pos) > 0:
            cost = np.zeros((n_g, len(prev_gt_pos)), dtype=np.float32)
            for i in range(n_g):
                for j in range(len(prev_gt_pos)):
                    cost[i, j] = np.linalg.norm(cur_pos[i] - prev_gt_pos[j])
            row_ind, col_ind = linear_sum_assignment(cost)
            used_prev = set()
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] <= GT_LINK_DIST and j not in used_prev:
                    cur_uid_list[i] = prev_gt_uid_list[j]
                    used_prev.add(j)

        # 为新出现的 GT 分配 uid
        for i in range(n_g):
            if cur_uid_list[i] is None:
                cur_uid_list[i] = next_gt_uid
                next_gt_uid += 1
            gt_life[cur_uid_list[i]]['appear'] += 1

        # 匹配当前帧 GT 和 tracks
        matched_pairs = []
        matched_gt_idx = set()
        matched_trk_idx = set()

        if n_g > 0 and n_t > 0:
            cost = np.zeros((n_g, n_t), dtype=np.float32)
            for i in range(n_g):
                for j in range(n_t):
                    cost[i, j] = np.linalg.norm(cur_pos[i] - np.array([trks[j]['x'], trks[j]['y'], trks[j]['z']]))
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] <= MATCH_DIST_THRESH:
                    matched_pairs.append((i, j))
                    matched_gt_idx.add(i)
                    matched_trk_idx.add(j)

        # 统计 TP/FP/FN
        total_tp += len(matched_pairs)
        total_fp += n_t - len(matched_trk_idx)
        total_fn += n_g - len(matched_gt_idx)
        for i, j in matched_pairs:
            total_dist += np.linalg.norm(cur_pos[i] - np.array([trks[j]['x'], trks[j]['y'], trks[j]['z']]))

        # ID Switch 检查 & 生命周期统计
        cur_gt_tid = {}
        for i, j in matched_pairs:
            uid = cur_uid_list[i]
            tid = trks[j]['track_id']
            gt_life[uid]['tracked'] += 1
            cur_gt_tid[uid] = tid
            # ID Switch: 这个 GT 上一帧匹配了另一个 track_id
            if uid in prev_gt_tid and prev_gt_tid[uid] != tid:
                total_idsw += 1

        # 更新上一帧状态
        prev_gt_uid_list = cur_uid_list
        prev_gt_pos = cur_pos
        prev_gt_tid = cur_gt_tid

    # MT / ML
    mt_count = 0
    ml_count = 0
    for life in gt_life.values():
        ratio = life['tracked'] / life['appear'] if life['appear'] > 0 else 0
        if ratio >= 0.8:
            mt_count += 1
        elif ratio <= 0.2:
            ml_count += 1

    # 指标
    mota = 1.0 - (total_fn + total_fp + total_idsw) / total_gt if total_gt > 0 else 0.0
    motp = total_dist / total_tp if total_tp > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        'total_frames': len(all_frame_ids),
        'total_gt': total_gt,
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'ID_SW': total_idsw,
        'MT': mt_count,
        'ML': ml_count,
        'MOTA': mota,
        'MOTP': motp,
        'Precision': precision,
        'Recall': recall,
    }


# ==============================================================================
# 5. 主流程
# ==============================================================================
def eval_one(tracks_csv, pkl_path, label):
    print(f"\n{'='*60}")
    print(f"  评估: {label}")
    print(f"{'='*60}")
    print(f"  跟踪结果: {tracks_csv}")
    print(f"  真值 PKL:  {pkl_path}")

    if not os.path.exists(tracks_csv):
        print(f"  [!] 找不到跟踪结果 CSV")
        return
    if not os.path.exists(pkl_path):
        print(f"  [!] 找不到真值 PKL")
        return

    tracks = load_tracks_from_csv(tracks_csv)
    gt = load_gt_from_pkl(pkl_path)
    m = evaluate_mot(tracks, gt)

    print(f"\n  帧数:        {m['total_frames']}")
    print(f"  总 GT 框数:  {m['total_gt']}")
    print(f"  TP:          {m['TP']}  |  FP: {m['FP']}  |  FN: {m['FN']}")
    print(f"  ID Switches: {m['ID_SW']}")
    print(f"  MT:          {m['MT']}  |  ML: {m['ML']}")
    print(f"\n  MOTA:        {m['MOTA']*100:.2f}%")
    print(f"  MOTP:        {m['MOTP']:.3f} m")
    print(f"  Precision:   {m['Precision']*100:.2f}%")
    print(f"  Recall:      {m['Recall']*100:.2f}%")


if __name__ == '__main__':
    BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'work_dirs')
    DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'my_formatted_data')

    eval_one(
        os.path.join(BASE, 'model_predict_results_frontal_40epoch', 'tracks_1.csv'),
        os.path.join(DATA, 'nuscenes_infos_frontal_test.pkl'),
        'Frontal'
    )
    eval_one(
        os.path.join(BASE, 'model_predict_results_oblique_240', 'tracks_2.csv'),
        os.path.join(DATA, 'nuscenes_infos_oblique_test.pkl'),
        'Oblique'
    )
