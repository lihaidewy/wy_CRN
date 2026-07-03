"""
离线对比 Baseline vs Distance-Aware Weighted 的按距离段精度
读取两个 eval_stats.pkl，合并 20m bins 为 50m 段，输出 Mean + P90
"""
import os
import pickle
import numpy as np

# ========== 配置 ==========
BASELINE_PKL = "work_dirs/eval_baseline_frontal/eval_stats.pkl"
WEIGHTED_PKL = "work_dirs/eval_distweight_frontal/eval_stats.pkl"

# 20m bins -> 50m 合并规则（近似，误差可忽略）
# 0-50:  0-20, 20-40, 40-60(前半) -> 直接合并整个 40-60 bin 近似
SEGMENT_MAP = {
    "Near (0-50m)":   ["0-20m", "20-40m", "40-60m"],
    "Mid (50-100m)":  ["40-60m", "60-80m", "80-100m"],
    "Far (100-150m)": ["100-120m", "120-140m", "140-160m"],
    "VeryFar (>150m)":["160-180m", "180-200m", "200-220m", "220-240m"],
}


def load_stats(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def merge_bins(stats, segment_map, metric='lon'):
    """将 20m bins 按 segment_map 合并，返回 {segment: [errors]}"""
    merged = {name: [] for name in segment_map}
    for seg_name, bin_names in segment_map.items():
        for bn in bin_names:
            key = f'bin_{bn}_{metric}'
            if key in stats:
                merged[seg_name].extend(stats[key])
    return merged


def calc_mean_p90(err_list):
    if not err_list:
        return None, None, 0
    arr = np.array(err_list)
    return float(np.mean(arr)), float(np.percentile(arr, 90)), len(arr)


def print_segment_table(baseline_stats, weighted_stats):
    print("\n" + "="*80)
    print("  Distance-Aware Weighting 按距离段效果验证")
    print("="*80)
    print(f"  {'距离段':<16} {'指标':<10} {'Baseline':<14} {'Weighted':<14} {'变化':<12} {'样本数':<10}")
    print("-"*80)

    for seg_name in SEGMENT_MAP:
        base_lon = merge_bins(baseline_stats, {seg_name: SEGMENT_MAP[seg_name]}, 'lon')
        wgt_lon = merge_bins(weighted_stats, {seg_name: SEGMENT_MAP[seg_name]}, 'lon')

        base_mean, base_p90, base_n = calc_mean_p90(base_lon[seg_name])
        wgt_mean, wgt_p90, wgt_n = calc_mean_p90(wgt_lon[seg_name])

        if base_mean is None or wgt_mean is None:
            continue

        # Mean
        mean_delta = wgt_mean - base_mean
        mean_pct = (mean_delta / base_mean * 100) if base_mean > 0 else 0
        mean_tag = "↓" if mean_delta < 0 else "↑"
        print(f"  {seg_name:<16} {'MeanLon':<10} {base_mean:<14.3f} {wgt_mean:<14.3f} "
              f"{mean_tag}{abs(mean_delta):.3f}({abs(mean_pct):.1f}%){'':<4} {base_n}/{wgt_n}")

        # P90
        p90_delta = wgt_p90 - base_p90
        p90_pct = (p90_delta / base_p90 * 100) if base_p90 > 0 else 0
        p90_tag = "↓" if p90_delta < 0 else "↑"
        print(f"  {'':<16} {'P90Lon':<10} {base_p90:<14.3f} {wgt_p90:<14.3f} "
              f"{p90_tag}{abs(p90_delta):.3f}({abs(p90_pct):.1f}%)")
        print("-"*80)

    # 整体对比
    print(f"\n  {'【整体】':<16} {'指标':<10} {'Baseline':<14} {'Weighted':<14} {'变化':<12}")
    print("-"*80)
    for label, key in [('MeanLon', 'lon_error_list'), ('P90Lon', 'lon_error_list')]:
        base_arr = np.array(baseline_stats.get(key, []))
        wgt_arr = np.array(weighted_stats.get(key, []))
        if len(base_arr) == 0 or len(wgt_arr) == 0:
            continue
        if label == 'MeanLon':
            base_v, wgt_v = float(np.mean(base_arr)), float(np.mean(wgt_arr))
        else:
            base_v, wgt_v = float(np.percentile(base_arr, 90)), float(np.percentile(wgt_arr, 90))
        delta = wgt_v - base_v
        pct = (delta / base_v * 100) if base_v > 0 else 0
        tag = "↓" if delta < 0 else "↑"
        print(f"  {'Overall':<16} {label:<10} {base_v:<14.3f} {wgt_v:<14.3f} "
              f"{tag}{abs(delta):.3f}({abs(pct):.1f}%)")
    print("="*80)


if __name__ == "__main__":
    if not os.path.exists(BASELINE_PKL):
        print(f"❌ 找不到 Baseline pkl: {BASELINE_PKL}")
        exit(1)
    if not os.path.exists(WEIGHTED_PKL):
        print(f"❌ 找不到 Weighted pkl: {WEIGHTED_PKL}")
        exit(1)

    base_stats = load_stats(BASELINE_PKL)
    wgt_stats = load_stats(WEIGHTED_PKL)

    print_segment_table(base_stats, wgt_stats)

    print("\n✅ 说明:")
    print("   • Near/Mid/Far/VeryFar 合并自 20m 细粒度 bins，存在少量边界混叠（±10m），不影响趋势判断")
    print("   • ↓ 表示 Weighted 优于 Baseline（误差降低）")
    print("   • 若 VeryFar 段 P90Lon 显著下降，则验证 Distance-Aware 策略对远处有效")
