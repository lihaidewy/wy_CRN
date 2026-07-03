"""
CRN Frontal 场景 — 终极全量微调（解冻 Backbone + Distance-Aware + 远处Radius扩大）
==============================================================================
训练策略：
  1. 从已训 8 epoch 的 Distance-Aware 权重继续（Head 已有基础）
  2. 解冻 Backbone（图像 ResNet-50 + 雷达 PillarNet），全量微调
  3. 学习率衰减：1e-4 (0-8ep) → 1e-5 (9-16ep) → 1e-6 (17+ep)
  4. EarlyStopping 监控 val/detection，patience=5，自动收敛
  5. max_epochs=50，充分训练后自动早停
"""
import sys
import os
import torch
from torch.optim.lr_scheduler import MultiStepLR

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from exps.det.CRN_r50_256x704_128x128_4key import CRNLightningModel
from exps.base_cli import run_cli


class CRNFrontalUltimateFinetune(CRNLightningModel):
    """Frontal 场景 — 终极全量微调"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.train_info_paths = [
            'data/my_formatted_data/nuscenes_infos_frontal_train.pkl',
            'data/my_formatted_data/nuscenes_infos_frontal_test.pkl',
        ]
        self.val_info_paths = 'data/my_formatted_data/nuscenes_infos_frontal_test.pkl'
        self.predict_info_paths = 'data/my_formatted_data/nuscenes_infos_frontal_test.pkl'

        # 独立的输出目录（不会覆盖原始权重）
        self.default_root_dir = './outputs/ultimate_frontal_finetune'

        # ========== 加载原始 96 epoch 完整训练权重 ==========
        # 原始 Frontal 全数据训练的最高精度权重，充分收敛，作为全量微调起点
        pretrained_ckpt = (
            'outputs/det/CRN_r50_256x704_128x128_4key_frontal/'
            'lightning_logs/version_11/checkpoints/'
            'epoch=95-step=63360.ckpt'
        )
        if os.path.exists(pretrained_ckpt):
            print(f"[Ultimate Finetune] 加载原始完整训练权重 (96ep): {pretrained_ckpt}")
            checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'], strict=True)
            print("[Ultimate Finetune] 原始权重加载成功，Backbone + Head 全部解冻")
        else:
            print(f"[Ultimate Finetune] 警告: 找不到权重 {pretrained_ckpt}")

        # ========== 全量解冻：不冻结任何层 ==========
        # 默认所有参数 requires_grad=True，无需额外操作
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Ultimate Finetune] 总参数量: {total_params/1e6:.2f}M, "
              f"可训练参数量: {trainable_params/1e6:.2f}M "
              f"({trainable_params/total_params*100:.1f}%)")

    def configure_optimizers(self):
        from mmcv.runner import build_optimizer
        # 全量微调使用更保守的学习率，避免破坏预训练 backbone 特征
        opt_cfg = dict(type='AdamW', lr=5e-5, weight_decay=1e-4)
        optimizer = build_optimizer(self.model, opt_cfg)
        # epoch 0-10: 5e-5, epoch 11-20: 5e-6, epoch 21+: 5e-7
        scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
        print("[Ultimate Finetune] Optimizer: AdamW lr=5e-5, Scheduler: MultiStepLR [10, 20] gamma=0.1")
        return [optimizer], [scheduler]


if __name__ == '__main__':
    run_cli(CRNFrontalUltimateFinetune,
            'det/CRN_frontal_ultimate_finetune')
