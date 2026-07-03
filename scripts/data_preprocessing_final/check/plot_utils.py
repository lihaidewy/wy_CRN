"""
论文级可视化工具 —— 用于 CRN 评估结果分析
=============================================
提供：
  1. 误差-距离曲线
  2. 累计分布函数曲线 (CDF)
  3. 分组 CDF（按距离分档）

风格：IEEE/CVPR 论文级
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm

# ==============================================================================
# 全局字体配置（支持 WSL / Windows / Linux）
# ==============================================================================
def _find_chinese_font():
    """按优先级搜索可用的中文字体文件（支持 WSL 路径）"""
    candidates = [
        # WSL 内访问 Windows 字体（最可能的情况）
        '/mnt/c/Windows/Fonts/simhei.ttf',
        '/mnt/c/Windows/Fonts/msyh.ttc',
        '/mnt/c/Windows/Fonts/simsun.ttc',
        '/mnt/c/Windows/Fonts/SIMHEI.TTF',
        # Windows 原生 Python
        r'C:\Windows\Fonts\simhei.ttf',
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\simsun.ttc',
        # Linux 常见 CJK 字体
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

SIMHEI_PATH = _find_chinese_font()

if SIMHEI_PATH:
    # 1) 注册到 matplotlib 字体管理器（强制刷新缓存）
    fm.fontManager.addfont(SIMHEI_PATH)
    # 2) 获取字体名称并设为全局默认
    _fp = fm.FontProperties(fname=SIMHEI_PATH)
    _font_name = _fp.get_name()
    plt.rcParams['font.family'] = [_font_name, 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    # 3) 创建不同用途的 FontProperties（把 weight='bold' 预置进去，避免 set_title 时冲突）
    FP_SIMHEI = FontProperties(fname=SIMHEI_PATH, size=11)
    FP_SIMHEI_TITLE = FontProperties(fname=SIMHEI_PATH, size=13, weight='bold')
    FP_SIMHEI_LABEL = FontProperties(fname=SIMHEI_PATH, size=12)
    FP_SIMHEI_LEGEND = FontProperties(fname=SIMHEI_PATH, size=9)
    print(f"[plot_utils] 中文字体已加载: {SIMHEI_PATH} (name={_font_name})")
else:
    FP_SIMHEI = None
    FP_SIMHEI_TITLE = None
    FP_SIMHEI_LABEL = None
    FP_SIMHEI_LEGEND = None
    print("[plot_utils] ⚠️ 未找到中文字体，中文将显示为方块！")
    print("           请确保 WSL 可访问 /mnt/c/Windows/Fonts/ 或安装 Linux CJK 字体。")

LANG = 'zh'
SAVE_PDF = False


# ==============================================================================
# 1. 全局论文风格设置
# ==============================================================================
def setup_paper_style():
    """设置 matplotlib 为论文级风格（字体已在模块顶部设置）"""
    # 逐个设置，避免 update() 覆盖字体配置
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    # 线条与标记
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['lines.linewidth'] = 1.8
    plt.rcParams['lines.markersize'] = 5
    # 刻度线向内
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['xtick.minor.size'] = 2.5
    plt.rcParams['ytick.minor.size'] = 2.5
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    # 输出
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.03
    plt.rcParams['savefig.transparent'] = False
    # 网格
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.25
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5

# 语言字典
def _t(zh_text, en_text):
    return zh_text if LANG == 'zh' else en_text


# ==============================================================================
# 2. 颜色与样式配置
# ==============================================================================
# 专业配色方案 (ColorBrewer Set1 + 自定义)
COLORS = {
    'lon': '#E41A1C',      # 红色 — 纵向误差
    'lat': '#377EB8',      # 蓝色 — 横向误差
    'ate': '#4DAF4A',      # 绿色 — 2D ATE
    'p90': '#FF7F00',      # 橙色 — P90 参考线
    'all': '#984EA3',      # 紫色 — 全部样本
}

# 距离分档颜色 (渐变)
DIST_COLORS = [
    '#1b9e77',  # 0-50m
    '#d95f02',  # 50-100m
    '#7570b3',  # 100-150m
    '#e7298a',  # 150-200m
    '#66a61e',  # >200m
]


# ==============================================================================
# 3. 误差-距离曲线
# ==============================================================================
def plot_error_vs_distance(stats_dict, title, save_path, distance_bins):
    """
    绘制误差随 GT 纵向距离变化的曲线。

    Args:
        stats_dict: 统计字典（含 bin_*_lon / bin_*_lat / bin_*_ate2d）
        title: 图表标题
        save_path: 保存路径
        distance_bins: 距离分段列表，如 [0, 20, 40, ..., 240]
    """
    setup_paper_style()

    x_centers = []
    lon_means, lat_means, ate_means = [], [], []
    counts = []

    for b in range(len(distance_bins) - 1):
        bin_name = f"{distance_bins[b]}-{distance_bins[b+1]}m"
        lon_list = stats_dict.get(f'bin_{bin_name}_lon', [])
        if not lon_list:
            continue
        lat_list = stats_dict.get(f'bin_{bin_name}_lat', [])
        ate_list = stats_dict.get(f'bin_{bin_name}_ate2d', [])

        x_centers.append((distance_bins[b] + distance_bins[b+1]) / 2)
        lon_means.append(np.mean(lon_list))
        lat_means.append(np.mean(lat_list))
        ate_means.append(np.mean(ate_list))
        counts.append(len(lon_list))

    if not x_centers:
        print(f"  [绘图跳过] {title}: 无有效数据")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.plot(x_centers, lon_means, 'o-', color=COLORS['lon'],
            linewidth=2.5, markersize=7, label=_t('纵向误差 (X)', 'Longitudinal Error (X)'))
    ax.plot(x_centers, lat_means, 's-', color=COLORS['lat'],
            linewidth=2.5, markersize=7, label=_t('横向误差 (Y)', 'Lateral Error (Y)'))
    ax.plot(x_centers, ate_means, '^-', color=COLORS['ate'],
            linewidth=2.5, markersize=7, label='2D ATE')

    # 标注样本数
    for x, lon, cnt in zip(x_centers, lon_means, counts):
        ax.annotate(f'n={cnt}', xy=(x, lon), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=7,
                    color='#555555')

    ax.set_xlabel(_t('GT 纵向距离 (m)', 'GT Longitudinal Distance (m)'), fontproperties=FP_SIMHEI_LABEL)
    ax.set_ylabel(_t('定位误差 (m)', 'Localization Error (m)'), fontproperties=FP_SIMHEI_LABEL)
    ax.set_title(title, fontproperties=FP_SIMHEI_TITLE)
    # 图例放左上角外侧，避免遮挡曲线
    ax.legend(loc='upper left', frameon=True, fancybox=False,
              edgecolor='gray', facecolor='white', bbox_to_anchor=(0, 1.02),
              prop=FP_SIMHEI_LEGEND)
    ax.set_xlim(0, max(x_centers) + 20)
    ax.set_ylim(0, max(max(lon_means), max(ate_means)) * 1.20)

    # 去掉上右边框（论文风格）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    if SAVE_PDF:
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, format='pdf')
        print(f"  [已保存] 误差-距离曲线: {save_path} + PDF")
    else:
        print(f"  [已保存] 误差-距离曲线: {save_path}")
    plt.close()


# ==============================================================================
# 4. CDF 曲线（总曲线 + 分组）
# ==============================================================================
def plot_cdf(errors_dict, title, save_path, xlabel='定位误差 (m)'):
    """
    绘制累计分布函数 (CDF) 曲线。

    Args:
        errors_dict: {label: error_list, ...}
                     label 如 'All', '0-50m', '50-100m', ...
        title: 图表标题
        save_path: 保存路径
        xlabel: 横轴标签
    """
    setup_paper_style()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    p90_summary = {}

    for idx, (label, errors) in enumerate(errors_dict.items()):
        if len(errors) == 0:
            continue

        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

        # 选择颜色
        if label == 'All':
            color = COLORS['all']
            linewidth = 3.0
            linestyle = '-'
            zorder = 10
        else:
            color = DIST_COLORS[idx % len(DIST_COLORS)]
            linewidth = 2.0
            linestyle = '-'
            zorder = 5

        ax.plot(sorted_errors, cdf, linewidth=linewidth, linestyle=linestyle,
                label=label, color=color, zorder=zorder)

        # 计算 P90
        p90_idx = np.searchsorted(cdf, 0.9, side='left')
        if p90_idx < len(sorted_errors):
            p90_val = sorted_errors[p90_idx]
            p90_summary[label] = p90_val

            # 标注 P90 点
            # All: 完整标注；分组: 只画点（数值见终端 P90 汇总），避免互相遮挡
            if label == 'All':
                ax.scatter([p90_val], [0.9], color=color, s=80, zorder=15,
                           edgecolors='white', linewidths=1.5)
                ax.annotate(f'P90={p90_val:.2f}m',
                            xy=(p90_val, 0.9), textcoords="offset points",
                            xytext=(0, -22), fontsize=10, color=color,
                            fontweight='bold')
            elif idx <= 3:
                # 分组只画圆点，不写文字
                ax.scatter([p90_val], [0.9], color=color, s=40, zorder=14,
                           edgecolors='white', linewidths=1.0)

    # P90 水平参考线
    ax.axhline(0.9, color=COLORS['p90'], linestyle='--', linewidth=1.2,
               alpha=0.6, label=_t('P90 阈值 (90%)', 'P90 Threshold (90%)'))

    ax.set_xlabel(xlabel, fontproperties=FP_SIMHEI_LABEL)
    ax.set_ylabel(_t('累计分布函数 (CDF)', 'Cumulative Distribution'), fontproperties=FP_SIMHEI_LABEL)
    ax.set_title(title, fontproperties=FP_SIMHEI_TITLE)
    # 分组数 > 4 时图例放图外右侧，避免遮挡曲线
    n_groups = len(errors_dict)
    if n_groups > 4:
        ax.legend(loc='center left', frameon=True, fancybox=False,
                  edgecolor='gray', facecolor='white',
                  bbox_to_anchor=(1.02, 0.5), ncol=1,
                  prop=FP_SIMHEI_LEGEND)
    else:
        ax.legend(loc='lower right', frameon=True, fancybox=False,
                  edgecolor='gray', facecolor='white', ncol=1,
                  prop=FP_SIMHEI_LEGEND)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.05)

    # 论文边框风格
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    if SAVE_PDF:
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, format='pdf')
        print(f"  [已保存] CDF 曲线: {save_path} + PDF")
    else:
        print(f"  [已保存] CDF 曲线: {save_path}")
    plt.close()

    # 打印 P90 汇总
    if p90_summary:
        print(f"\n  [P90] ({title}):")
        for label, val in sorted(p90_summary.items()):
            marker = " <--" if label == 'All' else ""
            print(f"      {label:<12}: {val:.3f} m{marker}")


# ==============================================================================
# 5. 便捷封装：一次性画所有图
# ==============================================================================
def plot_all(stats_dict, output_dir, distance_bins, prefix=''):
    """
    一次性绘制所有分析图表。

    Args:
        stats_dict: 统计字典
        output_dir: 输出目录
        distance_bins: 距离分段列表
        prefix: 文件名前缀（如 'frontal_' / 'oblique_'）
    """
    # 图表统一放到 output_dir/plots/ 子文件夹中
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # 1. 误差-距离曲线
    plot_error_vs_distance(
        stats_dict,
        title=_t('定位误差随距离变化', 'Localization Error vs. Distance'),
        save_path=os.path.join(plots_dir, f'{prefix}error_vs_distance.png'),
        distance_bins=distance_bins
    )

    # 2. 纵向误差 CDF（总曲线）
    lon_all = stats_dict.get('lon_error_list', [])
    if lon_all:
        plot_cdf(
            {'All': lon_all},
            title=_t('纵向误差 (X) 的 CDF', 'CDF of Longitudinal Error'),
            save_path=os.path.join(plots_dir, f'{prefix}cdf_lon.png'),
            xlabel=_t('纵向误差 (m)', 'Longitudinal Error (m)')
        )

    # 3. 横向误差 CDF（总曲线）
    lat_all = stats_dict.get('lat_error_list', [])
    if lat_all:
        plot_cdf(
            {'All': lat_all},
            title=_t('横向误差 (Y) 的 CDF', 'CDF of Lateral Error'),
            save_path=os.path.join(plots_dir, f'{prefix}cdf_lat.png'),
            xlabel=_t('横向误差 (m)', 'Lateral Error (m)')
        )

    # 4. 2D ATE CDF（不分组）
    ate_all = stats_dict.get('ate_2d_list', [])
    if ate_all:
        plot_cdf(
            {'All': ate_all},
            title=_t('二维定位误差 (2D ATE) 的 CDF', 'CDF of 2D ATE'),
            save_path=os.path.join(plots_dir, f'{prefix}cdf_ate.png'),
            xlabel='2D ATE (m)'
        )

    # 5. 2D ATE CDF（按距离分档分组）
    if ate_all:
        # 构建分组数据
        ate_grouped = {'All': ate_all}
        for b in range(len(distance_bins) - 1):
            low, high = distance_bins[b], distance_bins[b+1]
            bin_name = f"{low}-{high}m"
            ate_bin = stats_dict.get(f'bin_{bin_name}_ate2d', [])
            if ate_bin:
                label = f"{low}-{high}m" if high <= 240 else f">{low}m"
                ate_grouped[label] = ate_bin

        plot_cdf(
            ate_grouped,
            title=_t('2D ATE 分组 CDF（按距离分档）', '2D ATE CDF by Distance Range'),
            save_path=os.path.join(plots_dir, f'{prefix}cdf_ate_grouped.png'),
            xlabel='2D ATE (m)'
        )
