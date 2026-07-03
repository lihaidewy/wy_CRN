"""
CRN Oblique 场景 — 闭卷终极微调（消除数据泄露）
================================================
与 Ultimate 全量微调完全相同，唯一区别：
  训练数据只用 oblique_train.pkl（350帧），不再混入 test.pkl
  评估数据仍为 oblique_test.pkl（784帧），模型训练时未见过
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


class CRNObliqueUltimateClosed(CRNObliqueModel):
    """Oblique 场景 — 闭卷全量微调（训练集不含测试数据）"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # ========== 闭卷：训练集只用 train，不用 test ==========
        self.train_info_paths = [
            'data/my_formatted_data/nuscenes_infos_oblique_train.pkl',
        ]
        self.val_info_paths = 'data/my_formatted_data/nuscenes_infos_oblique_test.pkl'
        self.predict_info_paths = 'data/my_formatted_data/nuscenes_infos_oblique_test.pkl'

        # 独立的输出目录
        self.default_root_dir = './outputs/ultimate_oblique_finetune_closed'

        # ========== 加载原始完整训练权重（作为初始化）==========
        pretrained_ckpt = (
            'outputs/det/CRN_r50_256x704_128x128_4key_oblique/'
            'lightning_logs/version_5/checkpoints/'
            'epoch=95-step=108864.ckpt'
        )
        if os.path.exists(pretrained_ckpt):
            print(f"[Closed Finetune Oblique] 加载原始权重作为初始化: {pretrained_ckpt}")
            checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'], strict=True)
            print("[Closed Finetune Oblique] 权重加载成功，Backbone + Head 全部解冻")
            print("[Closed Finetune Oblique] ⚠️  训练集仅使用 oblique_train.pkl（350帧），test.pkl 不参与训练")
        else:
            print(f"[Closed Finetune Oblique] 警告: 找不到权重 {pretrained_ckpt}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Closed Finetune Oblique] 总参数量: {total_params/1e6:.2f}M, "
              f"可训练参数量: {trainable_params/1e6:.2f}M "
              f"({trainable_params/total_params*100:.1f}%)")

    def configure_optimizers(self):
        from mmcv.runner import build_optimizer
        opt_cfg = dict(type='AdamW', lr=5e-5, weight_decay=1e-4)
        optimizer = build_optimizer(self.model, opt_cfg)
        scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
        print("[Closed Finetune Oblique] Optimizer: AdamW lr=5e-5, Scheduler: MultiStepLR [10, 20] gamma=0.1")
        return [optimizer], [scheduler]


if __name__ == '__main__':
    run_cli(CRNObliqueUltimateClosed,
            'det/CRN_oblique_ultimate_finetune_closed')
