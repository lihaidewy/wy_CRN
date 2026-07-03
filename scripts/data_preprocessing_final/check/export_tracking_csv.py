"""
CRN 跟踪 CSV 独立导出脚本
============================
仅导出跟踪用 CSV，不做可视化/统计/PKL。
自动按拥堵级别拆分 moderate/heavy，分别输出到不同 CSV。
按 eval_split 自动分目录：测试集→泛化，训练集→非泛化。

用法示例:
  # Frontal 480 格，测试集（泛化）
  PYTHONPATH=. python export_tracking_csv.py \
      --ckpt outputs/det/.../last.ckpt \
      --out_dir work_dirs/export_frontal_480 \
      --scene frontal --resolution 480 --eval_split val

  # Oblique 960 格，训练集（非泛化）
  PYTHONPATH=. python export_tracking_csv.py \
      --ckpt outputs/det/.../last.ckpt \
      --out_dir work_dirs/export_oblique_960 \
      --scene oblique --resolution 960 --eval_split train
"""
import os
import sys
import csv
import argparse
import torch
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
import functools

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from datasets.nusc_det_dataset import collate_fn
from mmdet3d.core import LiDARInstance3DBoxes

# ==============================================================================
# 1. 配置常量
# ==============================================================================
# 测试集拥堵级别 offset 映射
CONGESTION_OFFSET_MAP_VAL = {
    'frontal': {
        'moderate': [200000],
        'heavy':    [300000],
    },
    'oblique': {
        'moderate': [500000],
        'heavy':    [600000],
    },
}

# 训练集拥堵级别 offset 映射
CONGESTION_OFFSET_MAP_TRAIN = {
    'frontal': {
        'moderate': [0, 800000],
        'heavy':    [100000],
    },
    'oblique': {
        'moderate': [400000, 900000],
        'heavy':    [700000],
    },
}

# 全局 offset 映射（用于查找帧号所属批次）
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
    900000:  'oblique',
}


def _get_scenario_by_frame_idx(frame_idx):
    """根据帧号查找对应场景和 offset"""
    matched_offset = -1
    matched_scenario = None
    for offset, scenario_name in OFFSET_SCENARIO_MAP.items():
        if offset <= frame_idx and offset > matched_offset:
            matched_offset = offset
            matched_scenario = scenario_name
    return matched_scenario, matched_offset


SCENE_PREFIX_MAP = {
    'frontal': 'S1',
    'oblique': 'S2',
}

CONGESTION_SUFFIX_MAP = {
    'moderate': 'm',
    'heavy':    's',
}


def get_congestion_level(frame_idx, scene, split='val'):
    """根据帧号和 split 判断拥堵级别（moderate / heavy）"""
    offset_map = CONGESTION_OFFSET_MAP_VAL if split == 'val' else CONGESTION_OFFSET_MAP_TRAIN
    offsets = offset_map.get(scene, {})
    for level, level_offsets in offsets.items():
        for offset in level_offsets:
            if split == 'val':
                # test 集按固定范围判断
                if scene == 'frontal':
                    if level == 'moderate' and 200000 <= frame_idx < 300000:
                        return level
                    if level == 'heavy' and 300000 <= frame_idx < 400000:
                        return level
                elif scene == 'oblique':
                    if level == 'moderate' and 500000 <= frame_idx < 600000:
                        return level
                    if level == 'heavy' and 600000 <= frame_idx < 700000:
                        return level
            else:
                # train 集：根据 offset 本身判断
                _, matched_offset = _get_scenario_by_frame_idx(frame_idx)
                if matched_offset == offset:
                    return level
    return None


def match_boxes(gt_xy_list, pred_xy_list, dist_thresh=2.0):
    """
    匈牙利匹配。
    Args:
        gt_xy_list:  [(gx, gy), ...]
        pred_xy_list: [(px, py, pz, pyaw, score), ...]
    Returns:
        matched_pairs: [(gt_idx, pred_idx), ...]
        unmatched_gt: [gt_idx, ...]
        unmatched_pred: [pred_idx, ...]
    """
    n_gt = len(gt_xy_list)
    n_pred = len(pred_xy_list)
    if n_gt == 0 or n_pred == 0:
        return [], list(range(n_gt)), list(range(n_pred))

    cost = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i, (gx, gy) in enumerate(gt_xy_list):
        for j, (px, py, _, _, _) in enumerate(pred_xy_list):
            cost[i, j] = np.sqrt((gx - px) ** 2 + (gy - py) ** 2)

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < dist_thresh:
            matched_pairs.append((r, c))

    matched_gt = set(r for r, _ in matched_pairs)
    matched_pred = set(c for _, c in matched_pairs)
    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_pred = [j for j in range(n_pred) if j not in matched_pred]
    return matched_pairs, unmatched_gt, unmatched_pred


# ==============================================================================
# 2. 主逻辑
# ==============================================================================
def run_export(args):
    # ---- 动态导入模型类 ----
    if args.scene == 'frontal':
        if args.resolution == '480':
            from exps.det.CRN_r50_256x704_128x128_4key_frontal_480 import CRNFrontalModel_480 as CRNLightningModel
        else:
            from exps.det.CRN_r50_256x704_128x128_4key_frontal import CRNFrontalModel as CRNLightningModel
    else:
        if args.resolution == '480':
            from exps.det.CRN_r50_256x704_128x128_4key_oblique_480 import CRNObliqueModel_480 as CRNLightningModel
        else:
            from exps.det.CRN_r50_256x704_128x128_4key_oblique import CRNObliqueModel as CRNLightningModel

    print(f">>> 场景: {args.scene}, 分辨率: {args.resolution}, split: {args.eval_split}")
    print(f">>> Checkpoint: {args.ckpt}")

    model = CRNLightningModel()
    checkpoint = torch.load(args.ckpt, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda().eval()

    # ---- 数据集 ----
    from datasets.nusc_det_dataset import NuscDatasetRadarDet
    if args.eval_split == 'train':
        info_path = model.train_info_paths
        sub_dir = '非泛化'
    else:
        info_path = model.val_info_paths
        sub_dir = '泛化'
    print(f">>> 数据集: {info_path}")
    print(f">>> 输出子目录: {sub_dir}")

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

    # ---- 创建输出目录 ----
    out_dir = os.path.join(args.out_dir, sub_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 初始化 CSV 文件 ----
    prefix = SCENE_PREFIX_MAP[args.scene]
    csv_files = {}
    csv_writers = {}

    # 全部预测框 CSV（无 gt 列）
    for level in ['moderate', 'heavy']:
        key = f'all_{level}'
        filename = f"{prefix}{CONGESTION_SUFFIX_MAP[level]}.csv"
        f = open(os.path.join(out_dir, filename), 'w', newline='', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(['frame', 'x', 'y', 'z', 'w', 'l', 'h', 'yaw', 'vx', 'vy'])
        csv_files[key] = f
        csv_writers[key] = writer

    # 仅 TP CSV（有 gt 列）
    for level in ['moderate', 'heavy']:
        key = f'tp_{level}'
        filename = f"{prefix}{CONGESTION_SUFFIX_MAP[level]}_tp.csv"
        f = open(os.path.join(out_dir, filename), 'w', newline='', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(['frame', 'gt_id', 'gt_x', 'gt_y', 'x', 'y', 'z', 'w', 'l', 'h', 'yaw', 'vx', 'vy'])
        csv_files[key] = f
        csv_writers[key] = writer

    # ---- 推理遍历 ----
    frame_count = 0
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
        frame_id = int(info['sample_token'].split('_')[-1])

        # 判断拥堵级别，不属于 moderate/heavy 则跳过
        congestion_level = get_congestion_level(frame_id, args.scene, args.eval_split)
        if congestion_level is None:
            continue

        # ---- 提取真值 ----
        gt_boxes = []
        gt_tokens = []
        for ann in info['ann_infos']:
            gx, gy, gz = ann['translation']
            gt_boxes.append((gx, gy, gz))
            gt_tokens.append(ann.get('token', ''))

        # ---- 提取预测 ----
        pred_boxes = []
        pred_metadata = []
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
            pred_metadata.append({
                'px': px, 'py': py, 'pz': pz,
                'pw': pw, 'pl': pl, 'ph': ph,
                'pyaw': pyaw, 'pvx': pvx, 'pvy': pvy,
            })

        # ---- 匹配 ----
        gt_xy = [(gx, gy) for gx, gy, _ in gt_boxes]
        pred_xy = [(px, py, pz, pyaw, s) for px, py, pz, pyaw, s in pred_boxes]
        matched_pairs, unmatched_gt, unmatched_pred = match_boxes(gt_xy, pred_xy, dist_thresh=args.match_dist)

        # ---- 写入全部预测框（无 gt 列）----
        writer_all = csv_writers[f'all_{congestion_level}']
        for meta in pred_metadata:
            writer_all.writerow([
                frame_id,
                f"{meta['px']:.4f}", f"{meta['py']:.4f}", f"{meta['pz']:.4f}",
                f"{meta['pw']:.4f}", f"{meta['pl']:.4f}", f"{meta['ph']:.4f}",
                f"{meta['pyaw']:.4f}", f"{meta['pvx']:.4f}", f"{meta['pvy']:.4f}",
            ])

        # ---- 写入 TP（有 gt 列）----
        writer_tp = csv_writers[f'tp_{congestion_level}']
        for gt_idx, pred_idx in matched_pairs:
            meta = pred_metadata[pred_idx]
            gx, gy, gz = gt_boxes[gt_idx]
            gt_token = gt_tokens[gt_idx]
            writer_tp.writerow([
                frame_id, gt_token, f"{gx:.4f}", f"{gy:.4f}",
                f"{meta['px']:.4f}", f"{meta['py']:.4f}", f"{meta['pz']:.4f}",
                f"{meta['pw']:.4f}", f"{meta['pl']:.4f}", f"{meta['ph']:.4f}",
                f"{meta['pyaw']:.4f}", f"{meta['pvx']:.4f}", f"{meta['pvy']:.4f}",
            ])

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  已处理 {frame_count} 帧（当前 frame_id={frame_id}）...")

    # ---- 关闭文件 ----
    for key, f in csv_files.items():
        f.close()

    print(f"\n[[OK]] CSV 导出完成！输出目录: {out_dir}")
    for level in ['moderate', 'heavy']:
        suffix = CONGESTION_SUFFIX_MAP[level]
        all_path = os.path.join(out_dir, f"{prefix}{suffix}.csv")
        tp_path = os.path.join(out_dir, f"{prefix}{suffix}_tp.csv")
        print(f"  {prefix}{suffix}.csv     (全部预测框)  -> {all_path}")
        print(f"  {prefix}{suffix}_tp.csv  (仅 TP)      -> {tp_path}")


# ==============================================================================
# 3. CLI 入口
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRN 跟踪 CSV 独立导出脚本')
    parser.add_argument('--ckpt', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--out_dir', type=str, required=True, help='评估结果输出根目录')
    parser.add_argument('--scene', type=str, default='frontal', choices=['frontal', 'oblique'])
    parser.add_argument('--resolution', type=str, default='960', choices=['480', '960'],
                        help='480 或 960 格 BEV 分辨率')
    parser.add_argument('--eval_split', type=str, default='val', choices=['train', 'val'],
                        help='train=非泛化, val=泛化')
    parser.add_argument('--score_thr', type=float, default=0.35, help='置信度阈值')
    parser.add_argument('--match_dist', type=float, default=2.0, help='匹配距离阈值(m)')
    parser.add_argument('--max_frames', type=int, default=999999, help='最大评估帧数')
    args = parser.parse_args()
    run_export(args)
