"""
CRN Frontal 场景 — 通用评估脚本
==============================
通过命令行参数指定 checkpoint 和输出目录，支持评估任意模型。

用法示例:
  # 评估 Baseline
  python eval_frontal.py --ckpt outputs/det/CRN_frontal_baseline_finetune/.../epoch=7.ckpt --out_dir work_dirs/eval_baseline

  # 评估 Distance-Aware
  python eval_frontal.py --ckpt outputs/det/CRN_frontal_distweight_finetune/.../epoch=7.ckpt --out_dir work_dirs/eval_distweight

  # 评估 Ultimate（最终权重）
  python eval_frontal.py --ckpt outputs/det/CRN_frontal_ultimate_finetune/.../last.ckpt --out_dir work_dirs/eval_ultimate

  # 只跑前 100 帧快速验证
  python eval_frontal.py --ckpt ... --out_dir ... --max_frames 100
"""
import os
import sys
import csv
import argparse
import torch
import cv2
import numpy as np
import functools
from pyquaternion import Quaternion
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from plot_utils import plot_all

# ==============================================================================
# 1. 基础配置（通过命令行参数覆盖）
# ==============================================================================
SCORE_THR = 0.35
NUM_FRAMES_TO_VIS = 999999
EXPORT_TRACKING_CSV = True
EVAL_SPLIT = 'val'
CONGESTION_LEVEL = 'all'

CONGESTION_OFFSET_MAP = {
    'frontal': {
        'moderate': [200000],
        'heavy':    [300000],
    },
    'oblique': {
        'moderate': [500000],
        'heavy':    [600000],
    },
}

# ---- 模型类将在 run_pure_inference 中根据 --scene 动态导入 ----
from datasets.nusc_det_dataset import collate_fn
from mmdet3d.core import LiDARInstance3DBoxes


# ==============================================================================
# 2. 解耦模块：结果 CSV 导出器
# ==============================================================================
class TrackingCSVExporter:
    def __init__(self, output_dir, enabled=True, suffix=""):
        self.enabled = enabled
        filename = f"tracking_preds{suffix}.csv" if suffix else "tracking_preds.csv"
        self.filepath = os.path.join(output_dir, filename)
        self.file = None
        self.writer = None

        if self.enabled:
            self.file = open(self.filepath, mode='w', newline='', encoding='utf-8')
            self.writer = csv.writer(self.file)
            self.writer.writerow(['frame_id', 'scene_label', 'scene_name', 'class_name', 'score',
                                  'x', 'y', 'z', 'w', 'l', 'h', 'yaw', 'vx', 'vy', 'gt_token'])

    def write_box(self, frame_id, scene_label, scene_name, class_name, score,
                  x, y, z, w, l, h, yaw, vx=0.0, vy=0.0, gt_token=''):
        if not self.enabled:
            return
        self.writer.writerow([
            frame_id, scene_label, scene_name, class_name, f"{score:.4f}",
            f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
            f"{w:.4f}", f"{l:.4f}", f"{h:.4f}",
            f"{yaw:.4f}", f"{vx:.4f}", f"{vy:.4f}", gt_token
        ])

    def close(self):
        if self.enabled and self.file is not None:
            self.file.close()
            abs_path = os.path.abspath(self.filepath)
            print("\n" + "="*60)
            print(f"🚀 [MOT 导出成功] 跟踪所需的目标信息已解耦导出！")
            print(f"📁 存储位置: {abs_path}")
            print("="*60)


# ==============================================================================
# 3. 核心三维几何函数
# ==============================================================================
def get_gt_box_corners(size, translation, rotation_quat):
    l, w, h = size
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
    corners = np.vstack([x_corners, y_corners, z_corners])
    rot_matrix = Quaternion(rotation_quat).rotation_matrix
    corners = np.dot(rot_matrix, corners)
    corners[0, :] += translation[0]
    corners[1, :] += translation[1]
    corners[2, :] += translation[2]
    return corners


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


def draw_3d_box(img, corners_2d, color):
    pts = corners_2d[:, :2].astype(int)
    lines = [[0,1], [1,2], [2,3], [3,0],
             [4,5], [5,6], [6,7], [7,4],
             [0,4], [1,5], [2,6], [3,7]]
    for line in lines:
        pt1 = tuple(pts[line[0]])
        pt2 = tuple(pts[line[1]])
        cv2.line(img, pt1, pt2, color, 2)


# ==============================================================================
# 4. 匹配函数 (贪婪最近邻)
# ==============================================================================
def match_boxes(gt_list, pred_list, dist_thresh=4.5, debug=False):
    n_gt = len(gt_list)
    n_pred = len(pred_list)
    if n_gt == 0 or n_pred == 0:
        return [], list(range(n_gt)), list(range(n_pred))

    cost = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i, (gx, gy, gz, gyaw) in enumerate(gt_list):
        for j, (px, py, pz, pyaw, _) in enumerate(pred_list):
            cost[i, j] = np.sqrt((gx - px)**2 + (gy - py)**2)

    matched_pairs = []
    matched_gt = set()
    matched_pred = set()

    all_pairs = []
    for i in range(n_gt):
        for j in range(n_pred):
            all_pairs.append((cost[i, j], i, j))
    all_pairs.sort(key=lambda x: x[0])

    for dist, i, j in all_pairs:
        if dist > dist_thresh:
            break
        if i not in matched_gt and j not in matched_pred:
            matched_pairs.append((i, j))
            matched_gt.add(i)
            matched_pred.add(j)

    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_pred = [j for j in range(n_pred) if j not in matched_pred]

    if debug and unmatched_pred:
        for j in unmatched_pred:
            px, py, pz, pyaw, pscore = pred_list[j]
            min_dist = float('inf')
            nearest_gt_idx = -1
            for i in range(n_gt):
                gx, gy, gz, gyaw = gt_list[i]
                d = np.sqrt((gx - px)**2 + (gy - py)**2)
                if d < min_dist:
                    min_dist = d
                    nearest_gt_idx = i
            if nearest_gt_idx >= 0:
                gx, gy, gz, gyaw = gt_list[nearest_gt_idx]
                print(f"    [FP调试] Pred{j} ({px:.2f}, {py:.2f}) -> 最近GT{nearest_gt_idx} ({gx:.2f}, {gy:.2f}), 距离={min_dist:.2f}m")

    return matched_pairs, unmatched_gt, unmatched_pred


# ==============================================================================
# 5. 统计辅助函数
# ==============================================================================
DISTANCE_BINS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]


def _empty_stats():
    s = {
        'total_frames': 0,
        'total_gt': 0,
        'total_pred': 0,
        'total_tp': 0,
        'sum_ate_3d': 0.0,
        'sum_ate_2d': 0.0,
        'sum_lon_error': 0.0,
        'sum_lat_error': 0.0,
        'sum_aoe': 0.0,
        'ate_2d_list': [],
        'lon_error_list': [],
        'lat_error_list': [],
    }
    for i in range(len(DISTANCE_BINS) - 1):
        bin_name = f"{DISTANCE_BINS[i]}-{DISTANCE_BINS[i+1]}m"
        s[f'bin_{bin_name}_lon'] = []
        s[f'bin_{bin_name}_lat'] = []
        s[f'bin_{bin_name}_ate2d'] = []
    return s


def _accumulate_stats(s, gt_boxes, pred_boxes, matched_pairs):
    s['total_frames'] += 1
    s['total_gt'] += len(gt_boxes)
    s['total_pred'] += len(pred_boxes)
    s['total_tp'] += len(matched_pairs)
    for gt_idx, pred_idx in matched_pairs:
        gx, gy, gz, gyaw = gt_boxes[gt_idx]
        px, py, pz, pyaw, _ = pred_boxes[pred_idx]
        ate_2d = np.sqrt((gx-px)**2 + (gy-py)**2)
        lon_err = abs(gx - px)
        lat_err = abs(gy - py)
        s['sum_ate_3d'] += np.sqrt((gx-px)**2 + (gy-py)**2 + (gz-pz)**2)
        s['sum_ate_2d'] += ate_2d
        s['sum_lon_error'] += lon_err
        s['sum_lat_error'] += lat_err
        s['ate_2d_list'].append(ate_2d)
        s['lon_error_list'].append(lon_err)
        s['lat_error_list'].append(lat_err)
        angle_diff = abs((pyaw - gyaw + np.pi) % (2 * np.pi) - np.pi)
        s['sum_aoe'] += angle_diff

        for b in range(len(DISTANCE_BINS) - 1):
            if DISTANCE_BINS[b] <= gx < DISTANCE_BINS[b + 1]:
                bin_name = f"{DISTANCE_BINS[b]}-{DISTANCE_BINS[b+1]}m"
                s[f'bin_{bin_name}_lon'].append(lon_err)
                s[f'bin_{bin_name}_lat'].append(lat_err)
                s[f'bin_{bin_name}_ate2d'].append(ate_2d)
                break


def _print_stats(stats_dict, title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    print(f"  帧数: {stats_dict['total_frames']}")
    print(f"  真值数: {stats_dict['total_gt']} | 预测数: {stats_dict['total_pred']} | TP: {stats_dict['total_tp']}")
    if stats_dict['total_tp'] > 0:
        tp = stats_dict['total_tp']
        print(f"  2D ATE:        {stats_dict['sum_ate_2d']/tp:.3f} m")
        print(f"  纵向误差(X):   {stats_dict['sum_lon_error']/tp:.3f} m")
        print(f"  横向误差(Y):   {stats_dict['sum_lat_error']/tp:.3f} m")
        if stats_dict['ate_2d_list']:
            print(f"  2D ATE P90:    {np.percentile(stats_dict['ate_2d_list'], 90):.3f} m")
        if stats_dict['lon_error_list']:
            print(f"  纵向误差 P90:  {np.percentile(stats_dict['lon_error_list'], 90):.3f} m")
        if stats_dict['lat_error_list']:
            print(f"  横向误差 P90:  {np.percentile(stats_dict['lat_error_list'], 90):.3f} m")
        aoe_deg = (stats_dict['sum_aoe']/tp) * 180.0 / np.pi
        print(f"  偏航角误差:    {aoe_deg:.2f}°")
    if stats_dict['total_gt'] > 0:
        fn = stats_dict['total_gt'] - stats_dict['total_tp']
        fp_count = stats_dict['total_pred'] - stats_dict['total_tp']
        mota = 1.0 - (fn + fp_count) / stats_dict['total_gt']
        print(f"  MOTA:          {mota*100:.1f}%")
        mr = fn / stats_dict['total_gt']
        print(f"  漏检率:        {mr*100:.1f}%")
    if stats_dict['total_pred'] > 0:
        fp_rate = (stats_dict['total_pred'] - stats_dict['total_tp']) / stats_dict['total_pred']
        print(f"  误检率:        {fp_rate*100:.1f}%")

    print(f"\n  📏 距离分段误差统计 (GT纵向距离 x):")
    print(f"  {'距离段':<12} {'数量':<8} {'纵向误差(m)':<14} {'横向误差(m)':<14} {'2D ATE(m)':<12}")
    print(f"  {'-'*60}")
    for b in range(len(DISTANCE_BINS) - 1):
        bin_name = f"{DISTANCE_BINS[b]}-{DISTANCE_BINS[b+1]}m"
        lon_list = stats_dict.get(f'bin_{bin_name}_lon', [])
        lat_list = stats_dict.get(f'bin_{bin_name}_lat', [])
        ate_list = stats_dict.get(f'bin_{bin_name}_ate2d', [])
        if lon_list:
            print(f"  {bin_name:<12} {len(lon_list):<8} {np.mean(lon_list):<14.3f} {np.mean(lat_list):<14.3f} {np.mean(ate_list):<12.3f}")


# ==============================================================================
# 6. 主推理函数
# ==============================================================================
def run_pure_inference(args):
    # 根据场景动态导入对应的模型类
    if args.scene == 'frontal':
        from exps.det.CRN_r50_256x704_128x128_4key_frontal import CRNFrontalModel as CRNLightningModel
        print(f">>> 使用 Frontal 模型类")
    else:
        from exps.det.CRN_r50_256x704_128x128_4key_oblique import CRNObliqueModel as CRNLightningModel
        print(f">>> 使用 Oblique 模型类")

    print(f">>> 正在初始化模型并加载权重...")
    print(f"    Checkpoint: {args.ckpt}")
    print(f"    Output:     {args.out_dir}")
    model = CRNLightningModel()
    checkpoint = torch.load(args.ckpt, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda().eval()

    from datasets.nusc_det_dataset import NuscDatasetRadarDet
    if args.eval_split == 'train':
        info_path = model.train_info_paths
        print(f"📊 评估数据集: TRAIN ({info_path})")
    else:
        info_path = model.val_info_paths
        print(f"📊 评估数据集: VAL ({info_path})")

    dataset = NuscDatasetRadarDet(
        ida_aug_conf=model.ida_aug_conf,
        bda_aug_conf=model.bda_aug_conf,
        rda_aug_conf=model.rda_aug_conf if hasattr(model, 'rda_aug_conf') else {},
        img_backbone_conf=model.backbone_img_conf,
        classes=model.class_names,
        data_root=model.data_root,
        info_paths=info_path,
        is_train=False,
        img_conf=model.img_conf,
        load_interval=1,
        num_sweeps=1,
        sweep_idxes=[],
        key_idxes=[],
        return_image=True,
        return_depth=model.return_depth,
        return_radar_pv=model.return_radar_pv,
        remove_z_axis=getattr(model, 'remove_z_axis', False),
        radar_pv_path='radar_pv_filter',
    )

    custom_collate = functools.partial(
        collate_fn,
        is_return_image=True,
        is_return_depth=model.return_depth,
        is_return_radar_pv=model.return_radar_pv
    )
    seq_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    print("✅ 模型与数据加载完成，启动推理...\n")

    # ---- 拥堵级别过滤 ----
    if args.congestion != 'all':
        def _get_offset_by_frame_idx(frame_idx):
            matched_offset = -1
            for offset in [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000]:
                if offset <= frame_idx and offset > matched_offset:
                    matched_offset = offset
            return matched_offset

        target_offsets = CONGESTION_OFFSET_MAP.get(args.scene, {}).get(args.congestion, [])
        original_len = len(dataset.infos)
        filtered_infos = []
        for info in dataset.infos:
            frame_idx = int(info['sample_token'].split('_')[-1])
            offset = _get_offset_by_frame_idx(frame_idx)
            if offset in target_offsets:
                filtered_infos.append(info)
        dataset.infos = filtered_infos
        print(f"🚦 拥堵过滤 [{args.scene} / {args.congestion}]:")
        print(f"   原始帧数: {original_len} -> 过滤后: {len(dataset.infos)}")
        print(f"   保留 offsets: {target_offsets}\n")

        global OUTPUT_DIR
        suffix = f"_{args.congestion}"
        if not args.out_dir.endswith(suffix):
            args.out_dir = f"{args.out_dir}{suffix}"

    os.makedirs(args.out_dir, exist_ok=True)

    csv_exporter = TrackingCSVExporter(args.out_dir, enabled=EXPORT_TRACKING_CSV)
    csv_exporter_tp = TrackingCSVExporter(args.out_dir, enabled=EXPORT_TRACKING_CSV, suffix="_tp_only")
    class_names_map = ['car', 'truck', 'bus', 'pedestrian', 'bicycle', 'motorcycle']

    def _get_scene_label(frame_id):
        if 200000 <= frame_id < 300000:
            return 'S1M_FLAG_TEST'
        elif 300000 <= frame_id < 400000:
            return 'S1S_FLAG_TEST'
        elif 500000 <= frame_id < 600000:
            return 'S2M_FLAG_TEST'
        elif 600000 <= frame_id < 700000:
            return 'S2S_FLAG_TEST'
        else:
            return ''

    stats = _empty_stats()
    MATCH_DIST_THRESH = 2.0

    for batch_idx, batch in enumerate(seq_loader):
        if batch_idx >= args.max_frames:
            break

        (sweep_imgs, mats, _, _, _, _, _, pts_pv) = batch
        sweep_imgs_cuda = sweep_imgs.cuda().half()
        pts_pv_cuda = pts_pv.cuda().float()
        mats_cuda = {k: (v.cuda().float() if isinstance(v, torch.Tensor) else v) for k, v in mats.items()}

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model.model(sweep_imgs_cuda, mats_cuda, sweep_ptss=pts_pv_cuda, is_train=True)
                preds = outputs[0]
            img_meta = dataset.infos[batch_idx].copy()
            img_meta['box_type_3d'] = LiDARInstance3DBoxes
            bbox_results = model.model.get_bboxes(preds, [img_meta])

        pred_boxes3d = bbox_results[0][0].tensor.cpu().numpy()
        pred_scores = bbox_results[0][1].cpu().numpy()
        pred_labels = bbox_results[0][2].cpu().numpy()

        info = dataset.infos[batch_idx]
        cam_info = info['cam_infos']['CAM_FRONT']
        img = cv2.imread(os.path.join("./data/my_formatted_data", cam_info['filename']))
        K = np.array(cam_info['calibrated_sensor']['camera_intrinsic'])

        quat = Quaternion(cam_info['calibrated_sensor']['rotation'])
        trans = np.array(cam_info['calibrated_sensor']['translation'])
        cam2world = np.eye(4)
        cam2world[:3, :3] = quat.rotation_matrix
        cam2world[:3, 3] = trans
        ego2cam = np.linalg.inv(cam2world)

        sample_token = info['sample_token']
        frame_id = int(sample_token.split('_')[-1])
        scene_label = _get_scene_label(frame_id)

        print(f"\n====================== 🛰️ Frame #{batch_idx} ======================")

        # ----- 提取真值 -----
        gt_boxes = []
        gt_tokens = []
        for ann in info['ann_infos']:
            gx, gy, gz = ann['translation']
            q = Quaternion(ann['rotation'])
            v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
            gt_yaw = np.arctan2(v[1], v[0])
            gt_boxes.append((gx, gy, gz, gt_yaw))
            gt_tokens.append(ann.get('token', ''))

            corners_world = get_gt_box_corners(ann['size'], ann['translation'], ann['rotation'])
            corners_world_homo = np.vstack([corners_world, np.ones((1, 8))])
            corners_cam = ego2cam @ corners_world_homo
            if np.any(corners_cam[2, :] > 0.1):
                corners_2d = (K @ corners_cam[:3, :] / corners_cam[2, :]).T
                draw_3d_box(img, corners_2d, (0, 0, 255))

        # ----- 提取预测 -----
        pred_boxes = []
        pred_visuals = []
        pred_metadata = []
        p_idx = 0
        for i in range(len(pred_boxes3d)):
            if pred_scores[i] < args.score_thr:
                continue
            box = pred_boxes3d[i]
            px, py, pz = box[0], box[1], box[2]
            pw, pl, ph = box[3], box[4], box[5]
            pyaw = box[6]
            pvx = float(box[7]) if len(box) > 7 else 0.0
            pvy = float(box[8]) if len(box) > 8 else 0.0

            pred_boxes.append((px, py, pz, pyaw, pred_scores[i]))
            class_idx = pred_labels[i]
            class_name = class_names_map[class_idx] if class_idx < len(class_names_map) else 'car'
            pred_metadata.append({
                'class_name': class_name, 'score': pred_scores[i],
                'px': px, 'py': py, 'pz': pz, 'pw': pw, 'pl': pl, 'ph': ph,
                'pyaw': pyaw, 'pvx': pvx, 'pvy': pvy,
            })

            corners_world = get_pred_box_corners(box)
            corners_world_homo = np.vstack([corners_world, np.ones((1, 8))])
            corners_cam = ego2cam @ corners_world_homo
            if np.any(corners_cam[2, :] > 0.1):
                corners_2d = (K @ corners_cam[:3, :] / corners_cam[2, :]).T
                pred_visuals.append((corners_2d, p_idx))
            p_idx += 1

        # ----- 匹配与统计 -----
        matched_pairs, unmatched_gt, unmatched_pred = match_boxes(
            gt_boxes, pred_boxes, dist_thresh=MATCH_DIST_THRESH, debug=True
        )
        _accumulate_stats(stats, gt_boxes, pred_boxes, matched_pairs)

        # 建立预测框 -> 真实token 映射
        pred_to_gt_token = {}
        for gt_idx, pred_idx in matched_pairs:
            pred_to_gt_token[pred_idx] = gt_tokens[gt_idx]

        # ----- 写入所有预测框 (tracking_preds.csv) -----
        for p_idx, meta in enumerate(pred_metadata):
            gt_token = pred_to_gt_token.get(p_idx, '')
            csv_exporter.write_box(
                frame_id, scene_label, args.scene, meta['class_name'], meta['score'],
                meta['px'], meta['py'], meta['pz'], meta['pw'], meta['pl'], meta['ph'],
                meta['pyaw'], meta['pvx'], meta['pvy'], gt_token
            )

        # ----- TP-only CSV -----
        for gt_idx, pred_idx in matched_pairs:
            meta = pred_metadata[pred_idx]
            gt_token = gt_tokens[gt_idx]
            csv_exporter_tp.write_box(
                frame_id, scene_label, args.scene, meta['class_name'], meta['score'],
                meta['px'], meta['py'], meta['pz'], meta['pw'], meta['pl'], meta['ph'],
                meta['pyaw'], meta['pvx'], meta['pvy'], gt_token
            )

        # ----- 可视化 -----
        matched_pred_set = set(p for _, p in matched_pairs)
        for corners_2d, p_idx in pred_visuals:
            if p_idx in matched_pred_set:
                color = (0, 255, 0)
                label = f"P{p_idx}(TP)"
            else:
                color = (255, 0, 255)
                label = f"P{p_idx}(FP)"
            draw_3d_box(img, corners_2d, color)
            cv2.putText(img, label, tuple(corners_2d[0][:2].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print(f"  ➤ 匹配结果: TP={len(matched_pairs)}, FN={len(unmatched_gt)}, FP={len(unmatched_pred)}")
        save_path = os.path.join(args.out_dir, f"predict_frame_{batch_idx:03d}.jpg")
        cv2.imwrite(save_path, img)

    # ========== 统计输出 ==========
    _print_stats(stats, "📊 全局汇总")
    plot_all(stats, args.out_dir, DISTANCE_BINS, prefix=f"{args.scene}_")

    print("\n" + "="*60)
    csv_exporter.close()
    csv_exporter_tp.close()

    import pickle
    stats_file = os.path.join(args.out_dir, 'eval_stats.pkl')
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)
    print(f"  📦 统计数据已保存: {stats_file}")


# ==============================================================================
# 7. 命令行入口
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRN Frontal 通用评估脚本')
    parser.add_argument('--ckpt', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--out_dir', type=str, required=True, help='评估结果输出目录')
    parser.add_argument('--scene', type=str, default='frontal', choices=['frontal', 'oblique'])
    parser.add_argument('--score_thr', type=float, default=0.35, help='置信度阈值')
    parser.add_argument('--max_frames', type=int, default=999999, help='最大评估帧数')
    parser.add_argument('--eval_split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--congestion', type=str, default='all', choices=['all', 'moderate', 'heavy'])
    args = parser.parse_args()
    run_pure_inference(args)
