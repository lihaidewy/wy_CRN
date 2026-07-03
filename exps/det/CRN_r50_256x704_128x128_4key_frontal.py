"""
CRN 分场景独立训练 — Frontal 场景配置
=========================================
继承基类 CRN_r50_256x704_128x128_4key 的全部参数，
仅覆盖数据 PKL 路径，加载 frontal 场景数据。
"""
import sys
import os

# 确保项目根目录在 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from exps.det.CRN_r50_256x704_128x128_4key import CRNLightningModel
from exps.base_cli import run_cli


class CRNFrontalModel(CRNLightningModel):
    """Frontal 场景专用模型 — 全数据训练（train + test 合并）"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 只用 frontal_train 训练，test 不混入
        self.train_info_paths = [
            'data/my_formatted_data/nuscenes_infos_frontal_train.pkl',
        ]
        # val 用于评估（注意：test 参与了训练，评估结果有偏）
        self.val_info_paths = 'data/my_formatted_data/nuscenes_infos_frontal_test.pkl'
        self.predict_info_paths = 'data/my_formatted_data/nuscenes_infos_frontal_test.pkl'


if __name__ == '__main__':
    run_cli(CRNFrontalModel,
            'det/CRN_r50_256x704_128x128_4key_frontal')
