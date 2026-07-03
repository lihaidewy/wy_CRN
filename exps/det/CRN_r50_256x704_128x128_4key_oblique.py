"""
CRN 分场景独立训练 — Oblique 场景配置
=========================================
继承基类 CRN_r50_256x704_128x128_4key 的全部参数，
仅覆盖数据 PKL 路径，加载 oblique 场景数据。
"""
import sys
import os

# 确保项目根目录在 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from exps.det.CRN_r50_256x704_128x128_4key import CRNLightningModel
from exps.base_cli import run_cli


class CRNObliqueModel(CRNLightningModel):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 只用 oblique_train 训练，test 不混入
        self.train_info_paths = [
            'data/my_formatted_data/nuscenes_infos_oblique_train.pkl',
        ]
        
        self.val_info_paths = 'data/my_formatted_data/nuscenes_infos_oblique_test.pkl'
        self.predict_info_paths = 'data/my_formatted_data/nuscenes_infos_oblique_test.pkl'


if __name__ == '__main__':
    run_cli(CRNObliqueModel,
            'det/CRN_r50_256x704_128x128_4key_oblique')
