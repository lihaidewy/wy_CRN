"""
独立绘图脚本 —— 从已保存的 eval_stats.pkl 加载数据
====================================================
用法：
    cd ~/CRN && PYTHONPATH=. python scripts/data_preprocessing_final/check/plot_from_saved.py

读取 OUTPUT_DIR 下的 eval_stats.pkl，秒出所有分析图。
可通过修改 STATS_FILE 变量切换不同的评估结果。
"""
import os
import sys
import pickle

# 确保同目录下的 plot_utils 可导入
sys.path.insert(0, os.path.dirname(__file__))

from plot_utils import plot_all

# ==============================================================================
# 配置：修改统计文件路径即可重新画图
# ==============================================================================
STATS_FILE = os.path.join(os.path.dirname(__file__),
                          '../../../work_dirs/model_predict_results_oblique_v4_epoch95',
                          'eval_stats.pkl')

# 自动推断输出目录和场景名
OUTPUT_DIR = os.path.dirname(STATS_FILE)
SCENE_NAME = os.path.basename(OUTPUT_DIR).replace('model_predict_results_', '').replace('_', '')

# 距离分段配置（与评估脚本保持一致）
DISTANCE_BINS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]


def main():
    if not os.path.exists(STATS_FILE):
        print(f"❌ 统计文件不存在: {STATS_FILE}")
        print(f"   请先运行评估脚本生成 eval_stats.pkl")
        sys.exit(1)

    print(f"[INFO] 加载统计数据: {STATS_FILE}")
    with open(STATS_FILE, 'rb') as f:
        stats = pickle.load(f)

    print(f"[INFO] 统计已加载: {stats['total_frames']} frames, {stats['total_gt']} GT, {stats['total_tp']} TP")
    print(f"[INFO] 开始绘制图表...\n")

    plot_all(stats, OUTPUT_DIR, DISTANCE_BINS,
             prefix=os.path.basename(os.path.dirname(STATS_FILE)).split('_')[-1] + '_')

    print("\n[INFO] 所有图表已生成完成！")


if __name__ == "__main__":
    main()
