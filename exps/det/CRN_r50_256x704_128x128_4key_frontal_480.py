"""
  CRN 分场景独立训练 — Frontal 场景 480格BEV 配置
  ===============================================
  继承 480 格基类，仅覆盖数据 PKL 路径。
  用于评估所有 Frontal 旧权重（version_1 及更早）。
  """
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
      sys.path.insert(0, _project_root)

from exps.det.CRN_r50_256x704_128x128_4key_480 import CRNLightningModel_480
from exps.base_cli import run_cli


class CRNFrontalModel_480(CRNLightningModel_480):
      """Frontal 场景专用模型 — 480格BEV 版本"""
      def __init__(self, *args, **kwargs) -> None:
          super().__init__(*args, **kwargs)
          self.train_info_paths = [
              'data/my_formatted_data/nuscenes_infos_frontal_train.pkl',
          ]
          self.val_info_paths = 'data/my_formatted_data/nuscenes_infos_frontal_test.pkl'
          self.predict_info_paths = 'data/my_formatted_data/nuscenes_infos_frontal_test.pkl'

          # 设置检测头使用 Frontal 策略（BBox加权 + Radius 扩大）
          if hasattr(self, 'model') and hasattr(self.model, 'pts_bbox_head'):
              self.model.pts_bbox_head.train_cfg['scenario'] = 'frontal'
              print("[480 Frontal] 检测头策略已切换为: frontal")


if __name__ == '__main__':
      run_cli(CRNFrontalModel_480,
              'det/CRN_r50_256x704_128x128_4key_frontal_480')