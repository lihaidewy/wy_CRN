"""
临时评估脚本 — 用于评估旧版本 (480格BEV) 的 checkpoint
不修改原配置文件，不影响正在运行的 Route B 训练
用法:
    PYTHONPATH=. python scripts/data_preprocessing_final/check/eval_v1_old.py \
        --ckpt outputs/det/CRN_frontal_ultimate_finetune_closed/lightning_logs/version_1/checkpoints/last.ckpt \
        --out_dir work_dirs/eval_frontal_closed_v1_train \
        --scene frontal \
        --eval_split train
"""
import os, sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ===== 关键：用旧配置（480格BEV）替换当前配置 =====
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "exps.det.CRN_r50_256x704_128x128_4key",
    "scripts/data_preprocessing_final/check/CRN_480_backup.py"
)
_old_module = importlib.util.module_from_spec(_spec)
sys.modules["exps.det.CRN_r50_256x704_128x128_4key"] = _old_module
_spec.loader.exec_module(_old_module)
# =====================================================

import argparse
import torch
from datasets.nusc_det_dataset import collate_fn
from mmdet3d.core import LiDARInstance3DBoxes

SCORE_THR = 0.35
NUM_FRAMES_TO_VIS = 999999
EXPORT_TRACKING_CSV = True
CONGESTION_OFFSET_MAP = {
    'frontal': {'moderate': [200000], 'heavy': [300000]},
    'oblique': {'moderate': [500000], 'heavy': [600000]},
}

def run_pure_inference(args):
    if args.scene == 'frontal':
        from exps.det.CRN_r50_256x704_128x128_4key_frontal import CRNFrontalModel as CRNLightningModel
        print(">>> 使用 Frontal 模型类 (480格BEV旧配置)")
    else:
        from exps.det.CRN_r50_256x704_128x128_4key_oblique import CRNObliqueModel as CRNLightningModel
        print(">>> 使用 Oblique 模型类")

    print(f">>> Checkpoint: {args.ckpt}")
    print(f">>> Output:     {args.out_dir}")

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
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    all_pred_boxes = []
    all_gt_boxes = []
    all_ate_2d = []
    all_lon_error = []
    all_lat_error = []
    all_depth_errors = []
    frame_count = 0

    import csv
    csv_path = os.path.join(args.out_dir, 'tracking_preds.csv')
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame_idx', 'timestamp', 'obj_id', 'x', 'y', 'z', 'w', 'l', 'h', 'yaw', 'score', 'vx', 'vy'])

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if frame_count >= args.max_frames:
                break

            (sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, depth_labels, pts_pv) = batch
            if torch.cuda.is_available():
                sweep_imgs = sweep_imgs.cuda()
                for key, value in mats.items():
                    mats[key] = value.cuda()
                pts_pv = pts_pv.cuda()
                gt_boxes_3d = [b.cuda() for b in gt_boxes_3d]

            preds = model(sweep_imgs, mats, pts_pv=pts_pv, is_train=False)
            preds = [pred[0] for pred in preds]

            from layers.heads.bev_depth_head_det import BEVDepthHead
            pred_bboxes = BEVDepthHead.get_bboxes(preds, model.model.head.test_cfg,
                                                   model.model.head.common_heads,
                                                   model.model.head.bbox_coder,
                                                   model.model.head.tasks)

            gt_box_list = []
            for gt_box in gt_boxes_3d[0]:
                x, y, z = gt_box[0].item(), gt_box[1].item(), gt_box[2].item()
                w, l, h = gt_box[3].item(), gt_box[4].item(), gt_box[5].item()
                yaw = gt_box[6].item()
                gt_box_list.append([x, y, z, w, l, h, yaw])

            pred_box_list = []
            for task_pred in pred_bboxes[0]:
                for box in task_pred:
                    score = box[-1].item()
                    if score < args.score_thr:
                        continue
                    x, y, z = box[0].item(), box[1].item(), box[2].item()
                    w, l, h = box[3].item(), box[4].item(), box[5].item()
                    yaw = box[6].item()
                    vx, vy = box[7].item(), box[8].item()
                    pred_box_list.append([x, y, z, w, l, h, yaw, score, vx, vy])

            all_pred_boxes.append(pred_box_list)
            all_gt_boxes.append(gt_box_list)

            if len(gt_box_list) > 0 and len(pred_box_list) > 0:
                import numpy as np
                gt_arr = np.array(gt_box_list)[:, :2]
                pred_arr = np.array(pred_box_list)[:, :2]
                cost = np.linalg.norm(gt_arr[:, None] - pred_arr[None, :], axis=2)
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] < 2.0:
                        dx = pred_box_list[c][0] - gt_box_list[r][0]
                        dy = pred_box_list[c][1] - gt_box_list[r][1]
                        all_ate_2d.append(np.sqrt(dx**2 + dy**2))
                        all_lon_error.append(abs(dx))
                        all_lat_error.append(abs(dy))

            frame_count += 1
            if frame_count % 50 == 0:
                print(f"  已处理 {frame_count} 帧...")

    csv_file.close()

    import numpy as np
    import pickle
    stats = {
        'total_frames': frame_count,
        'total_gt': sum(len(g) for g in all_gt_boxes),
        'total_pred': sum(len(p) for p in all_pred_boxes),
        'total_tp': len(all_ate_2d),
        'sum_ate_2d': sum(all_ate_2d),
        'sum_lon_error': sum(all_lon_error),
        'sum_lat_error': sum(all_lat_error),
        'ate_2d_list': all_ate_2d,
        'lon_error_list': all_lon_error,
        'lat_error_list': all_lat_error,
    }
    with open(os.path.join(args.out_dir, 'eval_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    if len(all_ate_2d) > 0:
        print("\n" + "="*60)
        print("  📊 评估结果")
        print("="*60)
        print(f"  总帧数:     {frame_count}")
        print(f"  GT 总数:    {stats['total_gt']}")
        print(f"  Pred 总数:  {stats['total_pred']}")
        print(f"  TP 数:      {stats['total_tp']}")
        print(f"  MOTA:       {1.0 - ((stats['total_gt'] - stats['total_tp']) + (stats['total_pred'] - stats['total_tp'])) / stats['total_gt']:.1%}")
        print(f"  Mean ATE:   {np.mean(all_ate_2d):.3f} m")
        print(f"  Mean Lon:   {np.mean(all_lon_error):.3f} m")
        print(f"  Mean Lat:   {np.mean(all_lat_error):.3f} m")
        print(f"  P90 ATE:    {np.percentile(all_ate_2d, 90):.3f} m")
        print(f"  P90 Lon:    {np.percentile(all_lon_error, 90):.3f} m")
        print(f"  P90 Lat:    {np.percentile(all_lat_error, 90):.3f} m")
        print("="*60)
    else:
        print("⚠️  没有匹配到任何 TP")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--scene', type=str, default='frontal')
    parser.add_argument('--score_thr', type=float, default=0.35)
    parser.add_argument('--max_frames', type=int, default=999999)
    parser.add_argument('--eval_split', type=str, default='val', choices=['train', 'val'])
    args = parser.parse_args()
    run_pure_inference(args)
