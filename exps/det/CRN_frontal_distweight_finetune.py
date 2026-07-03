"""
CRN Frontal 场景 — Distance-Aware 快速微调验证
===========================================
基于已有权重，冻结图像/雷达 backbone，
只微调 Head + Fuser，验证距离感知加权效果。
"""
import sys
import os
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from exps.det.CRN_r50_256x704_128x128_4key import CRNLightningModel
from exps.base_cli import run_cli


class CRNFrontalDistWeightFinetune(CRNLightningModel):
    """Frontal 场景 — Distance-Aware Loss Weighting 微调"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 使用 frontal 全数据
        self.train_info_paths = [
            'data/my_formatted_data/nuscenes_infos_frontal_train.pkl',
            'data/my_formatted_data/nuscenes_infos_frontal_test.pkl',
        ]
        self.val_info_paths = 'data/my_formatted_data/nuscenes_infos_frontal_test.pkl'
        self.predict_info_paths = 'data/my_formatted_data/nuscenes_infos_frontal_test.pkl'

        # 设置独立输出目录，避免覆盖历史实验
        self.default_root_dir = './outputs/distweight_frontal_finetune'

        # ========== 手动加载预训练权重（不恢复训练状态）==========
        pretrained_ckpt = (
            'outputs/det/CRN_r50_256x704_128x128_4key_frontal/'
            'lightning_logs/version_11/checkpoints/'
            'epoch=95-step=63360.ckpt'
        )
        if os.path.exists(pretrained_ckpt):
            print(f"[DistWeight Finetune] 加载预训练权重: {pretrained_ckpt}")
            checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'], strict=True)
            print("[DistWeight Finetune] 权重加载成功，optimizer 状态从 0 开始")
        else:
            print(f"[DistWeight Finetune] 警告: 找不到权重 {pretrained_ckpt}")
        # ==========================================================

        # ========== 冻结 Backbone，只训练 Head + Fuser ==========
        print("[DistWeight Finetune] 冻结图像 backbone 和雷达 backbone...")
        for param in self.model.backbone_img.parameters():
            param.requires_grad = False
        for param in self.model.backbone_pts.parameters():
            param.requires_grad = False

        # 打印可训练参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[DistWeight Finetune] 总参数量: {total_params/1e6:.2f}M, "
              f"可训练参数量: {trainable_params/1e6:.2f}M "
              f"({trainable_params/total_params*100:.1f}%)")
        # ========================================================


if __name__ == '__main__':
    run_cli(CRNFrontalDistWeightFinetune,
            'det/CRN_frontal_distweight_finetune')
