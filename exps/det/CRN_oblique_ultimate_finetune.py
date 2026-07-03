"""
CRN Oblique 场景 — 终极全量微调（解冻 Backbone + Distance-Aware + 远处Radius扩大）
==============================================================================
与 Frontal Ultimate 完全相同的训练策略，仅数据和预训练权重不同。
"""
import sys
import os
import torch
from torch.optim.lr_scheduler import MultiStepLR

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from exps.det.CRN_r50_256x704_128x128_4key_oblique import CRNObliqueModel
from exps.base_cli import run_cli


class CRNObliqueUltimateFinetune(CRNObliqueModel):
    """Oblique 场景 — 终极全量微调"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 独立的输出目录（不会覆盖原始权重）
        self.default_root_dir = './outputs/ultimate_oblique_finetune'

        # ========== 加载原始 96 epoch 完整训练权重 ==========
        pretrained_ckpt = (
            'outputs/det/CRN_r50_256x704_128x128_4key_oblique/'
            'lightning_logs/version_5/checkpoints/'
            'epoch=95-step=108864.ckpt'
        )
        if os.path.exists(pretrained_ckpt):
            print(f"[Ultimate Finetune Oblique] 加载原始完整训练权重 (96ep): {pretrained_ckpt}")
            checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'], strict=True)
            print("[Ultimate Finetune Oblique] 原始权重加载成功，Backbone + Head 全部解冻")
        else:
            print(f"[Ultimate Finetune Oblique] 警告: 找不到权重 {pretrained_ckpt}")

        # 全量解冻统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Ultimate Finetune Oblique] 总参数量: {total_params/1e6:.2f}M, "
              f"可训练参数量: {trainable_params/1e6:.2f}M "
              f"({trainable_params/total_params*100:.1f}%)")

    def configure_optimizers(self):
        from mmcv.runner import build_optimizer
        opt_cfg = dict(type='AdamW', lr=5e-5, weight_decay=1e-4)
        optimizer = build_optimizer(self.model, opt_cfg)
        scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
        print("[Ultimate Finetune Oblique] Optimizer: AdamW lr=5e-5, Scheduler: MultiStepLR [10, 20] gamma=0.1")
        return [optimizer], [scheduler]


if __name__ == '__main__':
    run_cli(CRNObliqueUltimateFinetune,
            'det/CRN_oblique_ultimate_finetune')
