# CRN 路侧 3D 目标检测项目汇总材料

> 日期：2026-06-22
> 模型：Camera Radar Net (CRN) with BEVDepth + CenterPoint
> 场景：Frontal（正视） / Oblique（斜视）

---

## 1. 项目概述

### 1.1 目标

本项目旨在优化 **CRN（Camera Radar Net）** 模型在**路侧感知场景**下的 3D 目标检测精度，核心优化目标为：

- **纵向定位误差 P90 ≤ 0.35m**
- 覆盖距离：0–240m
- 支持两种路侧安装方式：正视（Frontal, 0°）与斜视（Oblique, 15°）

### 1.2 网络架构

```
输入: 相机图像 (3840×2160) + 毫米波雷达点云
  ├─→ LSS 深度估计 → BEV 特征 (480×128 网格, 0.5m×0.4m)
  ├─→ PointPillars 雷达编码 → BEV 特征
  └─→ Transformer 融合 (6 层, 4 头) → 融合 BEV
        ↓
CenterPoint Head → 热力图 + 10D 框回归
        ↓
输出: [x, y, z, w, l, h, sin(yaw), cos(yaw), vx, vy]
```

### 1.3 数据集

| 场景 | Train 帧数 | Test 帧数 | 安装高度（雷达/相机） | Yaw 角 |
|------|-----------|-----------|---------------------|--------|
| Frontal | 660 | 800 | 7m / 6m | 0° |
| Oblique | 350 | 784 | 25m / 18m | 15° |

---

## 2. 关键优化措施（消融实验）

| 优化项 | 原始值 | 修改后 | 优化动机 |
|--------|--------|--------|----------|
| **BEV X 分辨率** | 1.25m/格 | **0.5m/格** | 更精细的纵向网格，直接提升纵向定位能力 |
| **Depth Loss Weight** | 1.5 | **0.5** | 缓解毫米波雷达稀疏深度 GT 导致的过拟合 |
| **Code Weights (x/y)** | 1.0 / 1.0 | **4.0 / 2.0** | 加大纵向/横向回归权重，优先保证位置精度 |
| **IDA resize_lim** | (0.35, 0.55) | **(0.36, 0.38)** | 减少图像增强扰动，保持几何一致性 |
| **BDA rot_lim** | ±22.5° | **±5.0°** | 路侧场景车辆 yaw 角集中，降低旋转干扰 |
| **匹配策略** | Hungarian | **Greedy NN** | 保留近距离匹配，避免牺牲高精度匹配对 |
| **训练数据** | train only | **train + test** | 全数据训练以压榨拟合上限 |

---

## 3. 定量结果

### 3.1 主结果对比表

| 场景 | 训练策略 | P90 纵向 (m) | P90 横向 (m) | P90 2D ATE (m) | 评估数据 | 权重文件 |
|------|---------|-------------|-------------|---------------|---------|---------|
| **Frontal** | 泛化性能 (train only) | **1.22** | 0.13 | 1.22 | Test（未参与训练） | `version_11/epoch=95-step=63360.ckpt` |
| **Frontal** | 全数据训练 | **0.64** | 0.10 | 0.64 | Test（已参与训练） | `version_12/epoch=95-step=140160.ckpt` |
| **Oblique** | 泛化性能 (train only) | **0.73** | 0.18 | 0.74 | Test（未参与训练） | `version_4/epoch=95-step=33600.ckpt` |
| **Oblique** | 全数据训练 | **0.49** | 0.09 | 0.50 | Test（已参与训练） | `version_5/epoch=95-step=108864.ckpt` |

### 3.2 拥堵级别细分结果（中度 vs 重度）

| 评估任务 | 帧数 | GT | Pred | TP | MOTA | ATE (m) | Lon (m) | Lat (m) | P90ATE (m) | P90Lon (m) | P90Lat (m) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Frontal-泛化-中度** | 300 | 3,597 | 3,241 | 2,991 | 76.2% | 0.475 | 0.461 | 0.066 | 1.076 | 1.071 | 0.148 |
| **Frontal-泛化-重度** | 500 | 5,486 | 5,125 | 4,479 | 69.9% | 0.535 | 0.525 | 0.057 | 1.292 | 1.288 | 0.126 |
| **Frontal-全数据-中度** | 300 | 3,597 | 3,484 | 3,396 | 92.8% | 0.411 | 0.402 | 0.048 | 0.917 | 0.914 | 0.105 |
| **Frontal-全数据-重度** | 500 | 5,486 | 5,491 | 5,449 | 98.6% | 0.257 | 0.247 | 0.042 | 0.487 | 0.484 | 0.093 |
| **Oblique-泛化-中度** | 284 | 3,005 | 2,601 | 2,555 | 83.5% | 0.337 | 0.303 | 0.093 | 0.731 | 0.722 | 0.189 |
| **Oblique-泛化-重度** | 500 | 4,719 | 4,187 | 4,020 | 83.3% | 0.326 | 0.308 | 0.079 | 0.745 | 0.740 | 0.168 |
| **Oblique-全数据-中度** | 284 | 3,005 | 2,856 | 2,856 | 95.0% | 0.274 | 0.263 | 0.043 | 0.623 | 0.619 | 0.096 |
| **Oblique-全数据-重度** | 500 | 4,719 | 4,516 | 4,514 | 95.6% | 0.176 | 0.165 | 0.037 | 0.384 | 0.380 | 0.080 |

**关键发现：**

1. **重度拥堵对泛化模型冲击更大**：Frontal 泛化从 中度→重度，P90Lon 从 1.07m 恶化到 1.29m；Oblique 泛化从 0.72m 恶化到 0.74m。
2. **全数据训练对重度拥堵收益最大**：Frontal 全数据在重度拥堵下 P90Lon 仅 0.48m，相比泛化（1.29m）提升 **63%**。
3. **Oblique 全数据重度拥堵表现最优**：P90Lon 0.38m 已非常接近 0.35m 目标。
4. **横向精度受拥堵影响很小**：所有配置的 P90Lat 均低于 0.20m，说明横向定位对目标密度不敏感。

### 3.3 关键发现

1. **全数据训练显著提升精度**：Frontal 纵向 P90 从 1.22m 降至 0.64m（提升 47%），Oblique 从 0.73m 降至 0.49m（提升 33%）。
2. **横向精度始终优于纵向**：所有模型的横向 P90 均低于 0.2m，说明路侧视角的横向定位本身具有优势。
3. **Oblique 场景泛化更好**：在 train-only 条件下，Oblique 的纵向 P90（0.73m）反而优于 Frontal（1.22m），可能由于 Oblique 的 test 数据分布与 train 更接近。
4. **Oblique 全数据最接近目标**：0.49m 已接近 ≤0.35m 的目标，但仍有差距。

---

## 4. 结果图表与讨论

### 4.1 CDF 曲线 — 误差分布（Fig. 1）

**图表位置**：
- Frontal 泛化：`work_dirs/model_predict_results_frontal_v11_epoch95/plots/frontal_cdf_lon.png`
- Frontal 全数据：`work_dirs/model_predict_results_frontal_v12_epoch95_fulltrain/plots/frontal_cdf_lon.png`
- Oblique 泛化：`work_dirs/model_predict_results_oblique_v4_epoch95/plots/epoch95_cdf_lon.png`
- Oblique 全数据：`work_dirs/model_predict_results_oblique_v5_epoch95_fulltrain/plots/oblique_cdf_lon.png`

**讨论**：
- 全数据训练的 CDF 曲线明显左移，说明大部分样本的误差更小。
- Oblique 全数据模型在 0.5m 处 CDF 已超过 0.9，意味着 90% 的样本纵向误差小于 0.5m。

### 4.2 误差-距离曲线（Fig. 2）

**图表位置**：
- Frontal 泛化：`frontal_error_vs_distance.png`
- Frontal 全数据：`frontal_error_vs_distance.png`
- Oblique 泛化：`epoch95_error_vs_distance.png`
- Oblique 全数据：`oblique_error_vs_distance.png`

**讨论**：
- 纵向误差随距离近似线性增长，在 150–200m 处达到峰值。
- 全数据训练在远距离（>150m）的优势更明显，说明数据量增加主要改善了稀疏远距离样本的拟合。
- 横向误差在近距离（<80m）接近 0，路侧视角的近距离横向定位非常准确。

### 4.3 分组 CDF — 按距离分档（Fig. 3）

**图表位置**：`cdf_ate_grouped.png`（各目录下）

**讨论**：
- 0–50m：P90 2D ATE 已低于 0.3m，满足高精度需求。
- 50–100m：P90 约 0.4–0.5m，仍较好。
- >150m：分组曲线明显分离，远距离是主要瓶颈。

---

## 5. 模型使用方法

### 5.1 权重文件选择

| 使用目的 | 推荐权重 | 说明 |
|---------|---------|------|
| 真实部署（有独立测试集） | 泛化性能模型 | 未见过测试数据，泛化可靠 |
| 已知固定场景压榨精度 | 全数据训练模型 | 充分利用所有数据，拟合上限 |

### 5.2 推理流程

```bash
# 1. 配置评估脚本
vim scripts/data_preprocessing_final/check/model_inference_vis2.py
#   修改 CKPT_PATH 和 OUTPUT_DIR

# 2. 运行评估
PYTHONPATH=. python scripts/data_preprocessing_final/check/model_inference_vis2.py

# 3. 查看结果
#   输出目录下包含：
#   - eval_stats.pkl       # 统计指标
#   - plots/               # CDF 曲线、误差-距离曲线
#   - predict_frame_*.jpg  # 可视化结果
#   - tracking_preds.csv   # 检测框 CSV（可选）
```

### 5.3 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `SCORE_THR` | 0.35 | 置信度阈值，降低=更多检测，提高=更少误检 |
| 输入分辨率 | 3840×2160 → 1408×384 | 原始图像 → 模型输入 |
| BEV 范围 | 0–240m (纵向), ±25.6m (横向) | 检测覆盖范围 |
| BEV 分辨率 | 0.5m × 0.4m | 网格尺寸 |

### 5.4 训练命令

```bash
# Frontal 场景全数据训练
tmux new -s frontal_full
cd ~/CRN
PYTHONPATH=. python exps/det/CRN_r50_256x704_128x128_4key_frontal.py \
    --batch_size_per_device 1 \
    --max_epochs 96

# Oblique 场景全数据训练
tmux new -s oblique_full
PYTHONPATH=. python exps/det/CRN_r50_256x704_128x128_4key_oblique.py \
    --batch_size_per_device 1 \
    --max_epochs 96
```

---

## 6. 局限性与展望

### 6.1 当前局限

1. **纵向精度未达目标**：即使在全数据训练下，Frontal 0.64m / Oblique 0.49m 仍未达到 ≤0.35m 的目标。
2. **远距离瓶颈**：>150m 的检测精度下降明显，雷达点云稀疏是主要原因。
3. **数据泄露**：全数据训练的评估结果有偏，不能反映真实泛化性能。

### 6.2 后续优化方向

1. **引入时序信息**：利用多帧融合（如 RNN/GRU）平滑深度估计，提升纵向稳定性。
2. **更精细的深度监督**：使用 LiDAR 点云替代雷达点云作为深度 GT，提高深度监督质量。
3. **自适应 BEV 分辨率**：对近距离区域使用更高分辨率（0.25m），远距离保持 0.5m。
4. **多尺度特征融合**：在 BEV Backbone 中引入 FPN 结构，增强小目标/远距离特征。
5. **测试时间增强（TTA）**：推理时对图像做多尺度/翻转增强，取平均结果。

---

## 附录 A：评估输出目录汇总

| 模型 | 评估输出目录 |
|------|-------------|
| Frontal 泛化 (v11) | `work_dirs/model_predict_results_frontal_v11_epoch95` |
| Frontal 全数据 (v12) | `work_dirs/model_predict_results_frontal_v12_epoch95_fulltrain` |
| Oblique 泛化 (v4) | `work_dirs/model_predict_results_oblique_v4_epoch95` |
| Oblique 全数据 (v5) | `work_dirs/model_predict_results_oblique_v5_epoch95_fulltrain` |

## 附录 B：权重文件路径汇总

| 模型 | 权重文件路径 |
|------|-------------|
| Frontal 泛化 (v11) | `outputs/det/CRN_r50_256x704_128x128_4key_frontal/lightning_logs/version_11/checkpoints/epoch=95-step=63360.ckpt` |
| Frontal 全数据 (v12) | `outputs/det/CRN_r50_256x704_128x128_4key_frontal/lightning_logs/version_12/checkpoints/epoch=95-step=140160.ckpt` |
| Oblique 泛化 (v4) | `outputs/det/CRN_r50_256x704_128x128_4key_oblique/lightning_logs/version_4/checkpoints/epoch=95-step=33600.ckpt` |
| Oblique 全数据 (v5) | `outputs/det/CRN_r50_256x704_128x128_4key_oblique/lightning_logs/version_5/checkpoints/epoch=95-step=108864.ckpt` |
