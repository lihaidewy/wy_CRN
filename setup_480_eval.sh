#!/bin/bash
# 一键创建 480 格永久评估脚本
cd /home/wy666/CRN

# 1. Frontal 480 评估脚本
cp scripts/data_preprocessing_final/check/eval_frontal.py \
   scripts/data_preprocessing_final/check/eval_frontal_480.py
sed -i 's|from exps.det.CRN_r50_256x704_128x128_4key_frontal import CRNFrontalModel|from exps.det.CRN_r50_256x704_128x128_4key_frontal_480 import CRNFrontalModel_480 as CRNFrontalModel|' \
    scripts/data_preprocessing_final/check/eval_frontal_480.py
echo "✅ eval_frontal_480.py"

# 2. Oblique 480 配置
cat > exps/det/CRN_r50_256x704_128x128_4key_oblique_480.py << 'PYEOF'
"""
CRN Oblique 场景 — 480格BEV 配置
"""
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from exps.det.CRN_r50_256x704_128x128_4key_480 import CRNLightningModel_480
from exps.base_cli import run_cli


class CRNObliqueModel_480(CRNLightningModel_480):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.train_info_paths = [
            'data/my_formatted_data/nuscenes_infos_oblique_train.pkl',
            'data/my_formatted_data/nuscenes_infos_oblique_test.pkl',
        ]
        self.val_info_paths = 'data/my_formatted_data/nuscenes_infos_oblique_test.pkl'
        self.predict_info_paths = 'data/my_formatted_data/nuscenes_infos_oblique_test.pkl'


if __name__ == '__main__':
    run_cli(CRNObliqueModel_480,
            'det/CRN_r50_256x704_128x128_4key_oblique_480')
PYEOF
echo "✅ CRN_r50_256x704_128x128_4key_oblique_480.py"

# 3. Oblique 480 评估脚本
cp scripts/data_preprocessing_final/check/eval_frontal.py \
   scripts/data_preprocessing_final/check/eval_oblique_480.py
sed -i 's|from exps.det.CRN_r50_256x704_128x128_4key_frontal import CRNFrontalModel|from exps.det.CRN_r50_256x704_128x128_4key_oblique_480 import CRNObliqueModel_480 as CRNFrontalModel|' \
    scripts/data_preprocessing_final/check/eval_oblique_480.py
echo "✅ eval_oblique_480.py"

echo ""
echo "全部创建完成！"
