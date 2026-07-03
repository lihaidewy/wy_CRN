"""
CRN Frontal 场景 — 闭卷终极微调（消除数据泄露）
=================================================
与 Ultimate 全量微调完全相同，唯一区别：
  训练数据只用 frontal_train.pkl（660帧），不再混入 test.pkl
  评估数据仍为 frontal_test.pkl（800帧），模型训练时未见过
"""
import sys
import os
import torch
from torch.optim.lr_scheduler import MultiStepLR

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from exps.det.CRN_r50_256x704_128x128_4key_frontal import CRNFrontalModel
from exps.base_cli import run_cli


class CRNFrontalUltimateClosed(CRNFrontalModel):
    """Frontal 场景 — 闭卷全量微调（训练集不含测试数据）"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # ========== 闭卷：训练集只用 train，不用 test ==========
        self.train_info_paths = [
            'data/my_formatted_data/nuscenes_infos_frontal_train.pkl',
        ]
        self.val_info_paths = 'data/my_formatted_data/nuscenes_infos_frontal_test.pkl'
        self.predict_info_paths = 'data/my_formatted_data/nuscenes_infos_frontal_test.pkl'

        # 独立的输出目录
        self.default_root_dir = './outputs/ultimate_frontal_finetune_closed'

        # ========== Frontal 专用策略标记 ==========
        if hasattr(self, 'model') and hasattr(self.model, 'pts_bbox_head'):
            self.model.pts_bbox_head.train_cfg['scenario'] = 'frontal'
            print("[Closed Finetune] Frontal 场景策略已启用：中间段保护 + 无 heatmap 加权")

        # ========== 加载原始完整训练权重（作为初始化）==========
        # 注意：原始权重是用 train+test 泄露训练的，但微调阶段不再给 test
        pretrained_ckpt = (
            'outputs/det/CRN_r50_256x704_128x128_4key_frontal/'
            'lightning_logs/version_11/checkpoints/'
            'epoch=95-step=63360.ckpt'
        )
        if os.path.exists(pretrained_ckpt):
            print(f"[Closed Finetune] 加载原始权重作为初始化: {pretrained_ckpt}")
            checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
            state_dict = checkpoint['state_dict']
            model_state = self.state_dict()
            # 过滤掉 shape 不匹配的层，以及 BEV 几何 buffer（需保持新配置的值）
            skip_patterns = ['model.fuser.positional_encoding', 'model.fuser.ref_2d',
                             'model.backbone_img.voxel_num', 'model.backbone_img.voxel_size',
                             'model.backbone_img.voxel_coord', 'model.backbone_img.frustum']
            filtered_state_dict = {}
            skipped_keys = []
            for k, v in state_dict.items():
                if any(p in k for p in skip_patterns):
                    skipped_keys.append(f"{k}: 跳过（保持新配置）")
                    continue
                if k in model_state:
                    if v.shape == model_state[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        skipped_keys.append(f"{k}: ckpt{v.shape} vs model{model_state[k].shape}")
                else:
                    skipped_keys.append(f"{k}: key not in model")
            self.load_state_dict(filtered_state_dict, strict=False)
            if skipped_keys:
                print(f"[Closed Finetune] 跳过的层 ({len(skipped_keys)} 个):")
                for sk in skipped_keys:
                    print(f"    - {sk}")
            print("[Closed Finetune] 权重加载完成，Backbone + Head 全部解冻")
            print("[Closed Finetune] ⚠️  训练集仅使用 frontal_train.pkl（660帧），test.pkl 不参与训练")
        else:
            print(f"[Closed Finetune] 警告: 找不到权重 {pretrained_ckpt}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Closed Finetune] 总参数量: {total_params/1e6:.2f}M, "
              f"可训练参数量: {trainable_params/1e6:.2f}M "
              f"({trainable_params/total_params*100:.1f}%)")

    def configure_optimizers(self):
        from mmcv.runner import build_optimizer
        opt_cfg = dict(type='AdamW', lr=5e-5, weight_decay=1e-4)
        optimizer = build_optimizer(self.model, opt_cfg)
        scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
        print("[Closed Finetune] Optimizer: AdamW lr=5e-5, Scheduler: MultiStepLR [10, 20] gamma=0.1")
        return [optimizer], [scheduler]


if __name__ == '__main__':
    run_cli(CRNFrontalUltimateClosed,
            'det/CRN_frontal_ultimate_finetune_closed')
