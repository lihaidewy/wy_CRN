#!/usr/bin/env python
"""
批量拥堵级别评估脚本
====================
自动运行 8 组评估：
  2 场景 (frontal/oblique) × 2 模型 (gen/full) × 2 拥堵级别 (moderate/heavy)

用法:
    cd ~/CRN
    PYTHONPATH=. python scripts/data_preprocessing_final/check/batch_eval_congestion.py

说明:
    - 本脚本读取 model_inference_vis2.py 作为模板，对每组配置生成临时脚本并执行
    - 不会修改原始 model_inference_vis2.py
    - 每组评估完成后自动清理临时文件
"""

import os
import sys
import tempfile
import subprocess

# ==============================================================================
# 项目根目录
# ==============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

# ==============================================================================
# 8 组评估配置
# 请根据你的实际权重路径修改下方 CKPT 路径
# ==============================================================================
EVAL_CONFIGS = [
    # ---- Frontal 泛化性能 (v11) ----
    {
        'name': 'Frontal-泛化-中度',
        'scene_name': 'frontal',
        'congestion_level': 'moderate',
        'ckpt_path': 'outputs/det/CRN_r50_256x704_128x128_4key_frontal/lightning_logs/version_11/checkpoints/epoch=95-step=63360.ckpt',
        'output_dir': 'work_dirs/congestion_eval/model_predict_results_frontal_v11_epoch95',
        'model_import': 'from exps.det.CRN_r50_256x704_128x128_4key_frontal import CRNFrontalModel as CRNLightningModel',
    },
    {
        'name': 'Frontal-泛化-重度',
        'scene_name': 'frontal',
        'congestion_level': 'heavy',
        'ckpt_path': 'outputs/det/CRN_r50_256x704_128x128_4key_frontal/lightning_logs/version_11/checkpoints/epoch=95-step=63360.ckpt',
        'output_dir': 'work_dirs/congestion_eval/model_predict_results_frontal_v11_epoch95',
        'model_import': 'from exps.det.CRN_r50_256x704_128x128_4key_frontal import CRNFrontalModel as CRNLightningModel',
    },
    # ---- Frontal 全数据 (v12) ----
    {
        'name': 'Frontal-全数据-中度',
        'scene_name': 'frontal',
        'congestion_level': 'moderate',
        'ckpt_path': 'outputs/det/CRN_r50_256x704_128x128_4key_frontal/lightning_logs/version_12/checkpoints/epoch=95-step=140160.ckpt',
        'output_dir': 'work_dirs/congestion_eval/model_predict_results_frontal_v12_epoch95_fulltrain',
        'model_import': 'from exps.det.CRN_r50_256x704_128x128_4key_frontal import CRNFrontalModel as CRNLightningModel',
    },
    {
        'name': 'Frontal-全数据-重度',
        'scene_name': 'frontal',
        'congestion_level': 'heavy',
        'ckpt_path': 'outputs/det/CRN_r50_256x704_128x128_4key_frontal/lightning_logs/version_12/checkpoints/epoch=95-step=140160.ckpt',
        'output_dir': 'work_dirs/congestion_eval/model_predict_results_frontal_v12_epoch95_fulltrain',
        'model_import': 'from exps.det.CRN_r50_256x704_128x128_4key_frontal import CRNFrontalModel as CRNLightningModel',
    },
    # ---- Oblique 泛化性能 (v4) ----
    {
        'name': 'Oblique-泛化-中度',
        'scene_name': 'oblique',
        'congestion_level': 'moderate',
        'ckpt_path': 'outputs/det/CRN_r50_256x704_128x128_4key_oblique/lightning_logs/version_4/checkpoints/epoch=95-step=33600.ckpt',
        'output_dir': 'work_dirs/congestion_eval/model_predict_results_oblique_v4_epoch95',
        'model_import': 'from exps.det.CRN_r50_256x704_128x128_4key_oblique import CRNObliqueModel as CRNLightningModel',
    },
    {
        'name': 'Oblique-泛化-重度',
        'scene_name': 'oblique',
        'congestion_level': 'heavy',
        'ckpt_path': 'outputs/det/CRN_r50_256x704_128x128_4key_oblique/lightning_logs/version_4/checkpoints/epoch=95-step=33600.ckpt',
        'output_dir': 'work_dirs/congestion_eval/model_predict_results_oblique_v4_epoch95',
        'model_import': 'from exps.det.CRN_r50_256x704_128x128_4key_oblique import CRNObliqueModel as CRNLightningModel',
    },
    # ---- Oblique 全数据 (v5) ----
    {
        'name': 'Oblique-全数据-中度',
        'scene_name': 'oblique',
        'congestion_level': 'moderate',
        'ckpt_path': 'outputs/det/CRN_r50_256x704_128x128_4key_oblique/lightning_logs/version_5/checkpoints/epoch=95-step=108864.ckpt',
        'output_dir': 'work_dirs/congestion_eval/model_predict_results_oblique_v5_epoch95_fulltrain',
        'model_import': 'from exps.det.CRN_r50_256x704_128x128_4key_oblique import CRNObliqueModel as CRNLightningModel',
    },
    {
        'name': 'Oblique-全数据-重度',
        'scene_name': 'oblique',
        'congestion_level': 'heavy',
        'ckpt_path': 'outputs/det/CRN_r50_256x704_128x128_4key_oblique/lightning_logs/version_5/checkpoints/epoch=95-step=108864.ckpt',
        'output_dir': 'work_dirs/congestion_eval/model_predict_results_oblique_v5_epoch95_fulltrain',
        'model_import': 'from exps.det.CRN_r50_256x704_128x128_4key_oblique import CRNObliqueModel as CRNLightningModel',
    },
]


def _read_template():
    """读取 model_inference_vis2.py 作为模板"""
    template_path = os.path.join(_SCRIPT_DIR, 'model_inference_vis2.py')
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def _generate_temp_script(template, cfg):
    """根据配置替换模板中的关键变量，生成临时脚本"""
    content = template

    # 替换 model import（注释掉旧的，插入新的）
    # 先找到所有模型导入行
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if 'from exps.det.CRN_r50' in line and 'import' in line:
            new_lines.append(f"# {line}")
        else:
            new_lines.append(line)
    content = '\n'.join(new_lines)

    # 插入新的模型导入（放到文件最前面，确保在 sys.path 之后）
    content = content.replace(
        'import os',
        f"import os\n{cfg['model_import']}"
    )

    # 替换 SCENE_NAME：逐行处理，确保只保留目标场景的一行
    lines = content.split('\n')
    new_lines = []
    scene_set = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("SCENE_NAME = ") or stripped.startswith("# SCENE_NAME = "):
            if not scene_set:
                new_lines.append(f"SCENE_NAME = '{cfg['scene_name']}'")
                scene_set = True
            else:
                # 已经设置了目标场景，其他 SCENE_NAME 行都注释掉
                if not stripped.startswith('#'):
                    new_lines.append(f"# {stripped}")
                else:
                    new_lines.append(line)
        else:
            new_lines.append(line)
    content = '\n'.join(new_lines)

    # CKPT_PATH
    # 找到当前模板中的 CKPT_PATH 行并替换
    lines = content.split('\n')
    new_lines = []
    ckpt_replaced = False
    for line in lines:
        if line.strip().startswith('CKPT_PATH = ') and not ckpt_replaced:
            new_lines.append(f"CKPT_PATH = \"{cfg['ckpt_path']}\"")
            ckpt_replaced = True
        else:
            new_lines.append(line)
    content = '\n'.join(new_lines)

    # OUTPUT_DIR
    lines = content.split('\n')
    new_lines = []
    output_replaced = False
    for line in lines:
        if line.strip().startswith('OUTPUT_DIR = ') and not output_replaced:
            new_lines.append(f"OUTPUT_DIR = \"{cfg['output_dir']}\"")
            output_replaced = True
        else:
            new_lines.append(line)
    content = '\n'.join(new_lines)

    # CONGESTION_LEVEL
    lines = content.split('\n')
    new_lines = []
    cong_replaced = False
    for line in lines:
        if line.strip().startswith("CONGESTION_LEVEL = ") and not cong_replaced:
            new_lines.append(f"CONGESTION_LEVEL = '{cfg['congestion_level']}'")
            cong_replaced = True
        else:
            new_lines.append(line)
    content = '\n'.join(new_lines)

    return content


def run_one_eval(cfg):
    """运行单次评估，返回关键统计指标"""
    template = _read_template()
    script_content = _generate_temp_script(template, cfg)

    # 写入临时文件
    fd, temp_path = tempfile.mkstemp(suffix='_eval.py', prefix='congestion_', dir=_SCRIPT_DIR)
    result_dict = None
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(script_content)

        print(f"\n{'='*70}")
        print(f"🚀 [{cfg['name']}] 开始评估")
        print(f"   场景: {cfg['scene_name']}")
        print(f"   拥堵: {cfg['congestion_level']}")
        print(f"   权重: {cfg['ckpt_path']}")
        print(f"   输出: {cfg['output_dir']}_{cfg['congestion_level']}")
        print(f"{'='*70}")

        env = os.environ.copy()
        env['PYTHONPATH'] = _PROJECT_ROOT

        result = subprocess.run(
            [sys.executable, temp_path],
            cwd=_PROJECT_ROOT,
            env=env,
        )

        if result.returncode == 0:
            print(f"✅ [{cfg['name']}] 评估完成")
            # 读取统计结果
            import pickle
            import numpy as np
            stats_path = os.path.join(_PROJECT_ROOT, f"{cfg['output_dir']}_{cfg['congestion_level']}", 'eval_stats.pkl')
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    stats = pickle.load(f)
                tp = stats.get('total_tp', 0)
                if tp > 0:
                    result_dict = {
                        'name': cfg['name'],
                        'scene': cfg['scene_name'],
                        'congestion': cfg['congestion_level'],
                        'frames': stats.get('total_frames', 0),
                        'gt': stats.get('total_gt', 0),
                        'pred': stats.get('total_pred', 0),
                        'tp': tp,
                        'mean_ate': stats.get('sum_ate_2d', 0.0) / tp,
                        'mean_lon': stats.get('sum_lon_error', 0.0) / tp,
                        'mean_lat': stats.get('sum_lat_error', 0.0) / tp,
                        'p90_ate': float(np.percentile(stats.get('ate_2d_list', [0]), 90)),
                        'p90_lon': float(np.percentile(stats.get('lon_error_list', [0]), 90)),
                        'p90_lat': float(np.percentile(stats.get('lat_error_list', [0]), 90)),
                        'mota': 1.0 - ((stats.get('total_gt', 0) - tp) + (stats.get('total_pred', 0) - tp)) / stats.get('total_gt', 1) if stats.get('total_gt', 0) > 0 else 0.0,
                    }
                else:
                    result_dict = {'name': cfg['name'], 'error': 'TP=0'}
            else:
                result_dict = {'name': cfg['name'], 'error': 'stats文件未找到'}
            return result_dict
        else:
            print(f"❌ [{cfg['name']}] 评估失败 (returncode={result.returncode})")
            return {'name': cfg['name'], 'error': f'returncode={result.returncode}'}
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    print("="*70)
    print("  CRN 拥堵级别批量评估")
    print("  共 8 组: 2 场景 × 2 模型 × 2 拥堵级别")
    print("="*70)

    results = []
    for i, cfg in enumerate(EVAL_CONFIGS, 1):
        print(f"\n📋 进度: {i}/{len(EVAL_CONFIGS)}")
        res = run_one_eval(cfg)
        results.append(res)

    success_count = sum(1 for r in results if 'error' not in r)
    print(f"\n{'='*70}")
    print(f"  批量评估结束: {success_count}/{len(EVAL_CONFIGS)} 组成功")
    print(f"{'='*70}")

    # 打印结果汇总
    print("\n📊 评估输出目录汇总:")
    for cfg in EVAL_CONFIGS:
        out = f"{cfg['output_dir']}_{cfg['congestion_level']}"
        print(f"   {cfg['name']:<20} -> {out}")

    # ========== 汇总表格 ==========
    print(f"\n{'='*100}")
    print("  📈 关键指标汇总")
    print(f"{'='*100}")
    print(f"  {'评估任务':<22} {'帧数':>6} {'GT':>5} {'Pred':>5} {'TP':>5} {'MOTA':>6} {'ATE':>7} {'Lon':>7} {'Lat':>7} {'P90ATE':>7} {'P90Lon':>7} {'P90Lat':>7}")
    print(f"  {'-'*100}")
    for r in results:
        if 'error' in r:
            print(f"  {r['name']:<22} {'失败':>72}")
        else:
            print(f"  {r['name']:<22} {r['frames']:>6} {r['gt']:>5} {r['pred']:>5} {r['tp']:>5} {r['mota']*100:>5.1f}% {r['mean_ate']:>7.3f} {r['mean_lon']:>7.3f} {r['mean_lat']:>7.3f} {r['p90_ate']:>7.3f} {r['p90_lon']:>7.3f} {r['p90_lat']:>7.3f}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
