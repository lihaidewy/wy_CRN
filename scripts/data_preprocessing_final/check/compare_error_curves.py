"""
CRN 改进前后误差曲线对比工具
==============================
读取两个 eval_stats.pkl，绘制叠加对比图。

用法:
  python scripts/data_preprocessing_final/check/compare_error_curves.py  --before work_dirs/model_predict_results_oblique_v4_epoch95/eval_stats.pkl  --after work_dirs/CRN_oblique_ultimate_finetune_closed_v0/eval_stats.pkl  --out_dir work_dirs/comparison_plots  --label_before "原始"  --label_after "优化"

"""
import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm

# ==============================================================================
# 字体配置（与 plot_utils.py 保持一致）
# ==============================================================================
def _find_chinese_font():
    candidates = [
        '/mnt/c/Windows/Fonts/simhei.ttf',
        '/mnt/c/Windows/Fonts/msyh.ttc',
        '/mnt/c/Windows/Fonts/simsun.ttc',
        '/mnt/c/Windows/Fonts/SIMHEI.TTF',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

SIMHEI_PATH = _find_chinese_font()
if SIMHEI_PATH:
    fm.fontManager.addfont(SIMHEI_PATH)
    _fp = fm.FontProperties(fname=SIMHEI_PATH)
    _font_name = _fp.get_name()
    plt.rcParams['font.family'] = [_font_name, 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    FP_SIMHEI = FontProperties(fname=SIMHEI_PATH, size=11)
    FP_SIMHEI_TITLE = FontProperties(fname=SIMHEI_PATH, size=13, weight='bold')
    FP_SIMHEI_LABEL = FontProperties(fname=SIMHEI_PATH, size=12)
    FP_SIMHEI_LEGEND = FontProperties(fname=SIMHEI_PATH, size=9)
else:
    FP_SIMHEI = FP_SIMHEI_TITLE = FP_SIMHEI_LABEL = FP_SIMHEI_LEGEND = None

# ==============================================================================
# 论文风格
# ==============================================================================
def setup_paper_style():
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['lines.linewidth'] = 1.8
    plt.rcParams['lines.markersize'] = 5
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['xtick.minor.size'] = 2.5
    plt.rcParams['ytick.minor.size'] = 2.5
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['figure.dpi'] = 150

# ==============================================================================
# 颜色配置
# ==============================================================================
COLORS = {
    'lon': '#E74C3C',   # 红
    'lat': '#3498DB',   # 蓝
    'ate': '#2ECC71',   # 绿
}

# ==============================================================================
# 读取 stats
# ==============================================================================
def load_stats(pkl_path):
    with open(pkl_path, 'rb') as f:
        stats = pickle.load(f)
    return stats

def extract_bin_means(stats, distance_bins):
    """提取各距离段的均值，返回 (x_centers, lon_means, lat_means, ate_means, counts)"""
    x_centers, lon_means, lat_means, ate_means, counts = [], [], [], [], []
    for b in range(len(distance_bins) - 1):
        bin_name = f"{distance_bins[b]}-{distance_bins[b+1]}m"
        lon_list = stats.get(f'bin_{bin_name}_lon', [])
        if not lon_list:
            continue
        lat_list = stats.get(f'bin_{bin_name}_lat', [])
        ate_list = stats.get(f'bin_{bin_name}_ate2d', [])
        x_centers.append((distance_bins[b] + distance_bins[b+1]) / 2)
        lon_means.append(np.mean(lon_list))
        lat_means.append(np.mean(lat_list))
        ate_means.append(np.mean(ate_list))
        counts.append(len(lon_list))
    return x_centers, lon_means, lat_means, ate_means, counts


def print_comparison_table(before_stats, after_stats, label_before, label_after, title='全局'):
    """打印改进前后指标对比表格"""
    def _extract(stats):
        tp = stats.get('total_tp', 0)
        gt = stats.get('total_gt', 0)
        pred = stats.get('total_pred', 0)
        frames = stats.get('total_frames', 0)
        if tp > 0:
            mean_ate = stats.get('sum_ate_2d', 0.0) / tp
            mean_lon = stats.get('sum_lon_error', 0.0) / tp
            mean_lat = stats.get('sum_lat_error', 0.0) / tp
            aoe_deg = (stats.get('sum_aoe', 0.0) / tp) * 180.0 / np.pi
            p90_ate = float(np.percentile(stats.get('ate_2d_list', [0]), 90))
            p90_lon = float(np.percentile(stats.get('lon_error_list', [0]), 90))
            p90_lat = float(np.percentile(stats.get('lat_error_list', [0]), 90))
        else:
            mean_ate = mean_lon = mean_lat = aoe_deg = p90_ate = p90_lon = p90_lat = 0.0
        if gt > 0:
            fn = gt - tp
            fp = pred - tp
            mota = 1.0 - (fn + fp) / gt
            miss_rate = fn / gt
        else:
            mota = miss_rate = 0.0
        if pred > 0:
            fp_rate = (pred - tp) / pred
        else:
            fp_rate = 0.0
        return {
            'frames': frames, 'gt': gt, 'pred': pred, 'tp': tp,
            'mean_ate': mean_ate, 'mean_lon': mean_lon, 'mean_lat': mean_lat,
            'p90_ate': p90_ate, 'p90_lon': p90_lon, 'p90_lat': p90_lat,
            'aoe': aoe_deg, 'mota': mota, 'miss': miss_rate, 'fp': fp_rate
        }

    b = _extract(before_stats)
    a = _extract(after_stats)

    print(f"\n{'='*80}")
    print(f"  📊 {title} — 改进前后指标对比")
    print(f"{'='*80}")
    print(f"  {'指标':<18} {label_before:<14} {label_after:<14} {'变化':<14}")
    print(f"  {'-'*60}")
    def _row(name, k, fmt='.3f', is_pct=False):
        vb, va = b[k], a[k]
        if is_pct:
            sb, sa = f"{vb*100:.1f}%", f"{va*100:.1f}%"
            delta = f"{((va-vb)*100):+.1f}pp"
        else:
            sb, sa = f"{vb:{fmt}}", f"{va:{fmt}}"
            if vb > 0 and k not in ['miss', 'fp']:
                delta = f"{((va-vb)/vb)*100:+.1f}%"
            elif k in ['miss', 'fp'] and vb > 0:
                delta = f"{((va-vb)/vb)*100:+.1f}%"
            else:
                delta = "-"
        print(f"  {name:<18} {sb:<14} {sa:<14} {delta:<14}")

    _row('帧数', 'frames', '.0f')
    _row('真值数', 'gt', '.0f')
    _row('预测数', 'pred', '.0f')
    _row('TP', 'tp', '.0f')
    _row('2D ATE (m)', 'mean_ate')
    _row('纵向误差 (m)', 'mean_lon')
    _row('横向误差 (m)', 'mean_lat')
    _row('2D ATE P90 (m)', 'p90_ate')
    _row('纵向 P90 (m)', 'p90_lon')
    _row('横向 P90 (m)', 'p90_lat')
    _row('偏航角误差 (°)', 'aoe')
    _row('MOTA', 'mota', is_pct=True)
    _row('漏检率', 'miss', is_pct=True)
    _row('误检率', 'fp', is_pct=True)
    print(f"{'='*80}")

# ==============================================================================
# 图1：误差-距离曲线（改进前后叠加）
# ==============================================================================
def plot_error_vs_distance_comparison(before_stats, after_stats, distance_bins,
                                      label_before, label_after, save_path):
    setup_paper_style()

    x_b, lon_b, lat_b, ate_b, cnt_b = extract_bin_means(before_stats, distance_bins)
    x_a, lon_a, lat_a, ate_a, cnt_a = extract_bin_means(after_stats, distance_bins)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # --- 纵向误差 ---
    ax.plot(x_b, lon_b, 'o--', color=COLORS['lon'], linewidth=1.8, markersize=6,
            alpha=0.5, label=f'{label_before} 纵向误差')
    ax.plot(x_a, lon_a, 'o-',  color=COLORS['lon'], linewidth=2.5, markersize=7,
            label=f'{label_after} 纵向误差')

    # --- 横向误差 ---
    ax.plot(x_b, lat_b, 's--', color=COLORS['lat'], linewidth=1.8, markersize=6,
            alpha=0.5, label=f'{label_before} 横向误差')
    ax.plot(x_a, lat_a, 's-',  color=COLORS['lat'], linewidth=2.5, markersize=7,
            label=f'{label_after} 横向误差')

    # --- 2D ATE ---
    ax.plot(x_b, ate_b, '^--', color=COLORS['ate'], linewidth=1.8, markersize=6,
            alpha=0.5, label=f'{label_before} 2D ATE')
    ax.plot(x_a, ate_a, '^-',  color=COLORS['ate'], linewidth=2.5, markersize=7,
            label=f'{label_after} 2D ATE')

    # 标注改善幅度（纵向）
    for x, v_b, v_a in zip(x_a, lon_b, lon_a):
        if v_b > 0:
            improvement = (v_b - v_a) / v_b * 100
            ax.annotate(f'↓{improvement:.0f}%', xy=(x, v_a),
                        textcoords="offset points", xytext=(0, 12),
                        ha='center', fontsize=7, color='#333333')

    ax.set_xlabel('GT 纵向距离 (m)', fontproperties=FP_SIMHEI_LABEL)
    ax.set_ylabel('定位误差 (m)', fontproperties=FP_SIMHEI_LABEL)
    ax.set_title('定位误差随距离变化（改进前后对比）', fontproperties=FP_SIMHEI_TITLE)
    ax.legend(loc='upper left', frameon=True, fancybox=False,
              edgecolor='gray', facecolor='white', bbox_to_anchor=(0, 1.02),
              prop=FP_SIMHEI_LEGEND, ncol=2)
    ax.set_xlim(0, max(x_a) + 20)
    ax.set_ylim(0, max(max(lon_b + lon_a), max(ate_b + ate_a)) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f'  [已保存] 误差-距离对比图: {save_path}')
    plt.close()

# ==============================================================================
# 图2：2D ATE CDF 对比
# ==============================================================================
def plot_cdf_comparison(before_stats, after_stats, label_before, label_after, save_path):
    setup_paper_style()

    ate_b = before_stats.get('ate_2d_list', [])
    ate_a = after_stats.get('ate_2d_list', [])

    fig, ax = plt.subplots(figsize=(7.5, 5))

    for data, color, label, ls in [
        (ate_b, '#7F8C8D', label_before, '--'),
        (ate_a, '#2C3E50', label_after,  '-'),
    ]:
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, ls, color=color, linewidth=2.5, label=label)

    # 标注 P90
    p90_b = np.percentile(ate_b, 90)
    p90_a = np.percentile(ate_a, 90)
    ax.axvline(p90_b, color='#7F8C8D', linestyle='--', alpha=0.5, linewidth=1.2)
    ax.axvline(p90_a, color='#2C3E50', linestyle='-',  alpha=0.7, linewidth=1.2)
    ax.annotate(f'P90 = {p90_b:.3f}m', xy=(p90_b, 0.9), fontsize=9, color='#7F8C8D')
    ax.annotate(f'P90 = {p90_a:.3f}m', xy=(p90_a, 0.85), fontsize=9, color='#2C3E50')

    ax.set_xlabel('2D ATE (m)', fontproperties=FP_SIMHEI_LABEL)
    ax.set_ylabel('累积概率', fontproperties=FP_SIMHEI_LABEL)
    ax.set_title('二维定位误差 CDF（改进前后对比）', fontproperties=FP_SIMHEI_TITLE)
    ax.legend(loc='lower right', frameon=True, fancybox=False,
              edgecolor='gray', facecolor='white', prop=FP_SIMHEI_LEGEND)
    ax.set_xlim(0, max(p90_b, p90_a) * 1.5)
    ax.set_ylim(0, 1.02)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f'  [已保存] CDF 对比图: {save_path}')
    plt.close()

# ==============================================================================
# 主入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='CRN 改进前后误差曲线对比')
    parser.add_argument('--before', type=str, required=True, help='改进前 eval_stats.pkl 路径（全局）')
    parser.add_argument('--after',  type=str, required=True, help='改进后 eval_stats.pkl 路径（全局）')
    parser.add_argument('--before_moderate', type=str, default='', help='改进前中度拥堵 pkl')
    parser.add_argument('--after_moderate',  type=str, default='', help='改进后中度拥堵 pkl')
    parser.add_argument('--before_heavy', type=str, default='', help='改进前重度拥堵 pkl')
    parser.add_argument('--after_heavy',  type=str, default='', help='改进后重度拥堵 pkl')
    parser.add_argument('--out_dir', type=str, default='work_dirs/comparison_plots', help='输出目录')
    parser.add_argument('--label_before', type=str, default='改进前', help='改进前图例标签')
    parser.add_argument('--label_after',  type=str, default='改进后', help='改进后图例标签')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    DISTANCE_BINS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]

    # ========== 全局对比 ==========
    print(f'读取改进前(全局): {args.before}')
    before_stats = load_stats(args.before)
    print(f'读取改进后(全局): {args.after}')
    after_stats = load_stats(args.after)

    print_comparison_table(before_stats, after_stats, args.label_before, args.label_after, title='全局')

    plot_error_vs_distance_comparison(
        before_stats, after_stats, DISTANCE_BINS,
        args.label_before, args.label_after,
        os.path.join(args.out_dir, 'error_vs_distance_comparison.png')
    )
    plot_cdf_comparison(
        before_stats, after_stats,
        args.label_before, args.label_after,
        os.path.join(args.out_dir, 'cdf_ate_comparison.png')
    )

    # ========== 中度拥堵对比 ==========
    if args.before_moderate and args.after_moderate:
        print(f'读取改进前(中度): {args.before_moderate}')
        b_m = load_stats(args.before_moderate)
        print(f'读取改进后(中度): {args.after_moderate}')
        a_m = load_stats(args.after_moderate)
        print_comparison_table(b_m, a_m, args.label_before, args.label_after, title='中度拥堵')
        plot_error_vs_distance_comparison(
            b_m, a_m, DISTANCE_BINS,
            args.label_before, args.label_after,
            os.path.join(args.out_dir, 'error_vs_distance_comparison_moderate.png')
        )
        plot_cdf_comparison(
            b_m, a_m,
            args.label_before, args.label_after,
            os.path.join(args.out_dir, 'cdf_ate_comparison_moderate.png')
        )

    # ========== 重度拥堵对比 ==========
    if args.before_heavy and args.after_heavy:
        print(f'读取改进前(重度): {args.before_heavy}')
        b_h = load_stats(args.before_heavy)
        print(f'读取改进后(重度): {args.after_heavy}')
        a_h = load_stats(args.after_heavy)
        print_comparison_table(b_h, a_h, args.label_before, args.label_after, title='重度拥堵')
        plot_error_vs_distance_comparison(
            b_h, a_h, DISTANCE_BINS,
            args.label_before, args.label_after,
            os.path.join(args.out_dir, 'error_vs_distance_comparison_heavy.png')
        )
        plot_cdf_comparison(
            b_h, a_h,
            args.label_before, args.label_after,
            os.path.join(args.out_dir, 'cdf_ate_comparison_heavy.png')
        )

    print(f'\n✅ 全部完成，输出目录: {args.out_dir}')

if __name__ == '__main__':
    main()
