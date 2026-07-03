import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from mmdet3d.core import LiDARInstance3DBoxes
from exps.det.CRN_r50_256x704_128x128_4key import CRNLightningModel

# ==========================================
# 1. 路径配置 (请确保与你的本地路径一致)
# ==========================================
CKPT_PATH = "./outputs/det/CRN_r50_256x704_128x128_4key/lightning_logs/version_68/checkpoints/epoch=23-step=4320.ckpt"

def run_probe():
    print("\n" + "="*60)
    print("🚀 启动底层探针：停止猜测，开始观测！")
    print("="*60 + "\n")

    # 1. 加载模型与数据集
    model = CRNLightningModel()
    checkpoint = torch.load(CKPT_PATH, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda().eval()  

    original_loader = model.train_dataloader()
    dataset = original_loader.dataset
    
    # 强制关闭所有流水线增强
    if hasattr(dataset, 'ida_aug_conf'): 
        dataset.ida_aug_conf['rand_flip'] = False
    if hasattr(dataset, 'bda_aug_conf'): 
        dataset.bda_aug_conf['rot_lim'] = (0.0, 0.0)
        dataset.bda_aug_conf['scale_lim'] = (1.0, 1.0)
        dataset.bda_aug_conf['flip_dx_ratio'] = 0.0
        dataset.bda_aug_conf['flip_dy_ratio'] = 0.0

    seq_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=original_loader.collate_fn)

    # 取出第一批数据 (第 0 帧)
    batch = next(iter(seq_loader))
    (sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, depth_labels, pts_pv) = batch
    
    info = dataset.infos[0]
    
    # ---------------------------------------------------------
    # 🔍 观测 1：真值 (Ground Truth) 坐标长什么样？
    # ---------------------------------------------------------
    print("🔍 [1. Ground Truth] 看一眼真值框的坐标 (只看前2个)：")
    # 🚨 修复了 AttributeError，直接使用 .cpu().numpy()
    gt_boxes = gt_boxes_3d[0].cpu().numpy() 
    if len(gt_boxes) > 0:
        print(" -> GT [x, y, z, l, w, h, yaw]:")
        for i in range(min(2, len(gt_boxes))):
            print(f"    {np.round(gt_boxes[i], 2)}")
    else:
        print(" -> ⚠️ 这一帧没有真值框！")

    # ---------------------------------------------------------
    # 🔍 观测 2：挖出隐藏的 LiDAR 外参矩阵！
    # ---------------------------------------------------------
    print("\n🔍 [2. LiDAR Infos] 探测雷达外参的真实存储路径：")
    if 'lidar_infos' in info:
        lidar_keys = list(info['lidar_infos'].keys())
        print(f" -> lidar_infos 包含的传感器: {lidar_keys}")
        
        if len(lidar_keys) > 0:
            lidar_key = lidar_keys[0]  # 通常是 'LIDAR_TOP' 或类似名字
            print(f" -> '{lidar_key}' 包含的字段: {list(info['lidar_infos'][lidar_key].keys())}")
            
            if 'calibrated_sensor' in info['lidar_infos'][lidar_key]:
                calib_keys = list(info['lidar_infos'][lidar_key]['calibrated_sensor'].keys())
                print(f" -> 标定参数包含: {calib_keys}")
                # 顺便把平移量打印出来看看长啥样
                if 'translation' in info['lidar_infos'][lidar_key]['calibrated_sensor']:
                    print(f" -> 🎯 找到了雷达平移量 (Translation): {info['lidar_infos'][lidar_key]['calibrated_sensor']['translation']}")
    else:
        print(" -> ❌ 找不到 lidar_infos 字段！")

    # ---------------------------------------------------------
    # 🔍 观测 3：模型预测的坐标长什么样？
    # ---------------------------------------------------------
    sweep_imgs_cuda = sweep_imgs.cuda()
    pts_pv_cuda = pts_pv.cuda()
    mats_cuda = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in mats.items()}
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model.model(sweep_imgs_cuda, mats_cuda, sweep_ptss=pts_pv_cuda, is_train=True)
            preds = outputs[0]
            
        img_meta = info.copy()
        img_meta['box_type_3d'] = LiDARInstance3DBoxes
        img_metas = [img_meta]
        bbox_results = model.model.get_bboxes(preds, img_metas)
        
    pred_boxes3d = bbox_results[0][0].tensor.cpu().numpy()
    
    print("\n🔍 [3. Model Prediction] 模型预测的原始坐标 (只看前2个)：")
    if len(pred_boxes3d) > 0:
        print(" -> Pred [x, y, z, l, w, h, yaw]:")
        for i in range(min(2, len(pred_boxes3d))):
            print(f"    {np.round(pred_boxes3d[i], 2)}")
    else:
        print(" -> ⚠️ 模型在这一帧没有预测出任何框，可能该帧真的没车，建议换一帧测。")
            
    print("\n" + "="*60)
    print("✅ 探针执行完毕！请把上面全部的打印结果发给我！")

if __name__ == "__main__":
    run_probe()