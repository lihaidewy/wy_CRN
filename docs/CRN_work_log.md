# CRN 路侧项目 — 完整工作日志

> 最后更新：2026-07-02（新增 §7 命令全集模块）  
> 工作原则：**面向文档工作**，所有配置变更、实验设计、评估结果均记录在此文件中  
> 关联文档：`docs/CRN_optimization_history.md`（历史版本详细记录）、`docs/CRN_optimization_history_detailed.md`（细节记录）

---

## 目录

1. [项目总览](#1-项目总览)
2. [数据结构](#2-数据结构)
3. [配置矩阵](#3-配置矩阵)
4. [优化历程时间线](#4-优化历程时间线)
5. [实验结果汇总](#5-实验结果汇总)
6. [当前运行状态](#6-当前运行状态)
7. [命令速查](#7-命令速查)
8. [待办事项](#8-待办事项)
9. [经验教训](#9-经验教训)

---

## 1. 项目总览

### 1.1 项目目标

将 CRN（Camera Radar Net）从 nuScenes 车端自动驾驶适配到**路侧交通杆感知**场景：
- 单相机（3840×2160 → 384×1408）+ 单毫米波雷达
- BEV 感知范围：0~240m（纵向）× ±25.6m（横向）
- **核心指标**：纵向定位误差 P90 ≤ 0.35m

### 1.2 场景定义

| 属性 | 场景一（Frontal） | 场景二（Oblique） |
|------|------------------|------------------|
| 安装方式 | 正向安装（yaw=0°） | 斜向安装（yaw=15°） |
| 雷达高度 | 7m | 25m |
| 相机高度 | 6m | 18m |
| 训练帧数 | 660 帧 | 606 帧 |
| 测试帧数 | 800 帧 | 784 帧 |
| 核心难点 | 远端目标 depth 误差大，纵向定位差 | 高安装角导致投影畸变，数据量少 |

### 1.3 网络架构

```
输入: 相机图像 (3840×2160 → 384×1408) + 毫米波雷达点云
  ├─→ RVTLSSFPN (ResNet-50 + DepthNet) → BEV 图像特征
  ├─→ PtsBackbone (PointPillars + SECOND) → BEV 雷达特征
  └─→ MFAFuser (6层 Transformer, 4头, embed=128) → 融合 BEV
        ↓
BEVDepthHead (CenterPoint 风格, 6 组分类任务) → 热力图 + 10D 框回归
        ↓
输出: [x, y, z, w, l, h, sin(yaw), cos(yaw), vx, vy]
```

### 1.4 Code Weights（10 维回归权重）

| 索引 | 维度 | 当前权重 | 含义 |
|------|------|---------|------|
| 0 | x（纵向） | **4.0** | 核心指标，最高权重 |
| 1 | y（横向） | **2.0** | 次于纵向 |
| 2 | z（高度） | 0.5 | 路侧高度变化小 |
| 3-4 | w, l（尺寸） | 1.5 | 适中 |
| 5 | h（高度尺寸） | 0.5 | 变化小 |
| 6-7 | sin/cos(yaw) | **3.0** | 朝向影响框形状 |
| 8-9 | vx, vy（速度） | 0.2 | 雷达自带速度 |

---

## 2. 数据结构

### 2.1 数据预处理管线

- 脚本：`scripts/data_preprocessing_final/pipeline.py`（4 阶段管线）
- 配置：`scripts/data_preprocessing_final/config.py`
- 划分模式：`train_test`（按 offset 批次严格划分）

### 2.2 Offset 批次划分

| Offset | 场景 | Split | 帧数 | 拥堵级别 | 用途 |
|--------|------|-------|------|----------|------|
| 0 | Frontal | **train** | 100 | 中度 | 训练 |
| 100000 | Frontal | **train** | 260 | 重度 | 训练 |
| 200000 | Frontal | test | 400 | 中度 | 泛化评估 |
| 300000 | Frontal | test | 400 | 重度 | 泛化评估 |
| 400000 | Oblique | **train** | 300 | 中度 | 训练 |
| 500000 | Oblique | test | 400 | 中度 | 泛化评估 |
| 600000 | Oblique | test | 384 | 重度 | 泛化评估 |
| 700000 | Oblique | **train** | 6 | 重度 | 训练 |
| 800000 | Frontal | **train** | 300 | 中度 | 训练 |
| 900000 | Oblique | **train** | 300 | 中度 | 训练 |

### 2.3 PKL 文件帧数统计

| PKL 文件 | 帧数 | 内容 |
|----------|------|------|
| `nuscenes_infos_frontal_train.pkl` | 660 | Frontal 训练集（offset 0, 100000, 800000） |
| `nuscenes_infos_frontal_test.pkl` | 800 | Frontal 测试集（offset 200000, 300000） |
| `nuscenes_infos_oblique_train.pkl` | 606 | Oblique 训练集（offset 400000, 700000, 900000） |
| `nuscenes_infos_oblique_test.pkl` | 784 | Oblique 测试集（offset 500000, 600000） |
| `nuscenes_infos_train.pkl` | 1266 | 全局训练集（frontal 660 + oblique 606） |
| `nuscenes_infos_test.pkl` | 1584 | 全局测试集（frontal 800 + oblique 784） |

### 2.4 训练策略

- **分场景独立训练**：Frontal 和 Oblique 各自独立训练，不混合
- **仅用 train 训练**：`train_info_paths` 只包含 `*_train.pkl`，test 数据不参与训练
- **评估方式**：
  - `train` split → 输出到 `非泛化/`（训练集性能）
  - `val` split → 输出到 `泛化/`（泛化性能）

---

## 3. 配置矩阵

### 3.1 所有配置文件一览

| 配置文件 | BEV 分辨率 | 场景 | 训练数据 | 用途 |
|----------|-----------|------|----------|------|
| `CRN_r50_256x704_128x128_4key.py` | **960 格** (0.25m) | 全场景 | 全局 train（1266帧） | Route B 基类（已废弃用于训练）|
| `CRN_r50_256x704_128x128_4key_frontal.py` | **960 格** | Frontal | frontal_train（660帧） | 960 格 Frontal 训练 |
| `CRN_r50_256x704_128x128_4key_oblique.py` | **960 格** | Oblique | oblique_train（606帧） | 960 格 Oblique 训练 |
| `CRN_frontal_ultimate_finetune_closed.py` | **960 格** | Frontal | frontal_train（660帧） | 闭卷微调（含 version_11 权重加载）|
| `CRN_r50_256x704_128x128_4key_480.py` | **480 格** (0.5m) | 通用 | — | 480 格基类 |
| `CRN_r50_256x704_128x128_4key_frontal_480.py` | **480 格** | Frontal | frontal_train（660帧） | **← 当前主力训练配置** |
| `CRN_r50_256x704_128x128_4key_oblique_480.py` | **480 格** | Oblique | oblique_train（606帧） | 480 格 Oblique 训练 |

### 3.2 评估脚本一览

| 脚本 | BEV 分辨率 | 场景 | 备注 |
|------|-----------|------|------|
| `eval_frontal.py` | 960 格 | Frontal + Oblique | 通用评估（含可视化） |
| `eval_frontal_480.py` | 480 格 | Frontal | 含 gt_token 列 |
| `eval_oblique_480.py` | 480 格 | Oblique | 含 gt_token 列 |
| `export_tracking_csv.py` | 480/960 | Frontal + Oblique | **独立 CSV 导出**（无可视化，自动分拥堵） |
| `export_all_csv.sh` | 480 | 两个场景 | 一键导出全部 CSV |

### 3.3 BEV 分辨率演变

| 阶段 | 分辨率 | 网格 | 感知范围 |
|------|--------|------|----------|
| A（早期） | 1.25m/格 | 128×176 | 0~160m |
| B | 0.625m/格 | 128×352 | 0~220m |
| **C（当前主力）** | **0.5m/格** | **128×480** | **0~240m** |
| D（Route B） | 0.25m/格 | 128×960 | 0~240m |

---

## 4. 优化历程时间线

### 4.1 全局基础优化

| 日期 | 改动 | 文件 | 效果 |
|------|------|------|------|
| 早期 | 相机 6→1，分辨率 256×704→384×1408 | `CRN_r50_...4key.py` | 适配路侧单相机 |
| 早期 | BDA rot_lim 22.5°→5° | `CRN_r50_...4key.py` | 路侧 yaw 集中，降旋转干扰 |
| 早期 | BEV X 多次提升: 1.25→0.625→0.5→0.25m | 多处 | 精度逐步提升 |
| 中期 | code_weights 调优: x=4.0→6.0→4.0, y=1.0→2.0 | `CRN_r50_...4key.py` | 平衡纵向和横向 |
| 中期 | SmoothL1 beta 0.11→0.05 | `CRN_r50_...4key.py` | 大误差惩罚更严 |
| 中期 | depth_loss weight 3.0→1.5→0.5 | `CRN_r50_...4key.py` | 降低 depth 噪声影响 |

### 4.2 场景一（Frontal）专属优化

| 日期 | 改动 | 策略 | 效果 |
|------|------|------|------|
| 2026-06 | 分场景训练 + train/test 分离 | 数据划分 | 消除数据泄露评估偏置 |
| 2026-06 | Distance-Aware BBox Weight | 中间段(50-100m) 2.0× 重点保护 | 中间段纵向改善 |
| 2026-06 | Gaussian Radius 分段扩大 | 1.2×/1.5×/2.0× | 远端 depth 容错 |
| 2026-06 | 微调实验组：Baseline / DistWeight / Ultimate / Closed | 对照验证各策略 | 全量解冻最佳 |
| 2026-06 | Route B (0.25m BEV, 960格) | 激进的 4.0× radius + 更高分辨率 | 显存极限，速度慢，搁置 |
| **2026-07-02** | **远端 BBox 加码** | 150m+: 1.2→2.5, 100-150m: 1.5→2.0 | **进行中** |
| **2026-07-02** | **Radius 保守化** | 4.0×/2.5×→2.0×/1.5×（回到 480 格标准） | **进行中** |
| **2026-07-02** | **Depth Loss 距离感知** | 远端像素 2.0×, 中端 1.0×, 近端 0.5× | **进行中** |

### 4.3 场景二（Oblique）专属优化

| 日期 | 改动 | 策略 | 效果 |
|------|------|------|------|
| 2026-06 | Distance-Aware Weighting | 远处重点（150m+: 3.0×） | 远端大幅改善 |
| 2026-06 | Heatmap Weight 启用 | 远处 heatmap 中心加权 | 数据稀缺场景有效 |
| 2026-06 | 微调实验组：Ultimate / Closed | 全量解冻 + 闭卷 | — |

### 4.4 训练基础设施改进

| 日期 | 改动 | 说明 |
|------|------|------|
| 2026-07-01 | 480 格永久评估方案 | 创建独立 480 格配置 + 评估脚本，不临时改配置 |
| 2026-07-01 | tmux 训练 + Windows Terminal | 解决 VS Code 终端关闭导致训练中断 |
| 2026-07-01 | `--ckpt_path` 续训 | 支持中断后恢复 optimizer/scheduler/epoch 状态 |
| 2026-07-02 | patience 10→5 | 更早触发 EarlyStopping |
| 2026-07-02 | `check_val_every_n_epoch` 1→2 | 减少验证频率，节省 ~15% 训练时间 |
| **2026-07-02** | **`export_tracking_csv.py`** | 独立 CSV 导出脚本，按拥堵级别自动拆分 |
| **2026-07-02** | **`TQDM_MINITERS=1` + `python -u`** | 进度条实时刷新 |

### 4.5 关键 Bug 修复

| Bug | 原因 | 解决 | 影响 |
|-----|------|------|------|
| `register_buffer` 静默覆盖 | `load_state_dict(strict=False)` 仍覆盖同名 buffer | 手动 skip `voxel_num/voxel_size/voxel_coord/frustum` | Route B 权重加载 |
| 训练集混入 test 数据 | 960 格 config 中 `train_info_paths` 包含 test | 改为仅 `*_train.pkl` | 闭卷训练 |
| WSAD 训练全场景混合 | 误用了基类 `CRN_r50_...4key.py`（全场景 1266 帧） | 改用分场景配置 | 已修正 |
| EarlyStopping resume 报错 | `val/detection` metric 在 resume 时不存在 | `check_on_train_epoch_end=False` | 已修复 |
| VS Code 终端关闭杀 tmux | WSL 闲置自动关闭 | 用 Windows Terminal 保持连接 | 已解决 |
| tmux socket 丢失 | WSL 虚拟机 shutdown | 重启训练 | 临时 |

---

## 5. 实验结果汇总

### 5.1 主结果

| 场景 | 训练策略 | P90 纵向 (m) | P90 横向 (m) | P90 2D ATE (m) | 评估数据 |
|------|---------|-------------|-------------|---------------|---------|
| **Frontal** | **泛化 (train only)** | **1.22** | 0.13 | 1.22 | Test（未参与训练） |
| Frontal | 全数据训练 | 0.64 | 0.10 | 0.64 | Test（已参与训练） |
| Oblique | 泛化 (train only) | 0.73 | 0.18 | 0.74 | Test（未参与训练） |
| Oblique | 全数据训练 | 0.49 | 0.09 | 0.50 | Test（已参与训练） |

### 5.2 拥堵级别细分

| 评估任务 | MOTA | ATE (m) | P90Lon (m) | P90Lat (m) |
|:---|---:|---:|---:|---:|
| **Frontal-泛化-中度** | 76.2% | 0.475 | 1.071 | 0.148 |
| **Frontal-泛化-重度** | 69.9% | 0.535 | **1.288** | 0.126 |
| Frontal-全数据-中度 | 92.8% | 0.411 | 0.914 | 0.105 |
| Frontal-全数据-重度 | 98.6% | 0.257 | 0.484 | 0.093 |
| Oblique-泛化-中度 | 83.5% | 0.337 | 0.722 | 0.189 |
| Oblique-泛化-重度 | 83.3% | 0.326 | 0.740 | 0.168 |
| Oblique-全数据-中度 | 95.0% | 0.274 | 0.619 | 0.096 |
| Oblique-全数据-重度 | 95.6% | 0.176 | 0.380 | 0.080 |

### 5.3 关键发现

1. **泛化能力是瓶颈**：Frontal 泛化 P90 纵向 1.22m，距离目标 0.35m 差距很大
2. **重度拥堵恶化严重**：Frontal 泛化从中度→重度，P90Lon 从 1.07m→1.29m
3. **远端漏检是核心问题**：重度拥堵（offset 300000）MOTA 仅 69.9%
4. **横向精度始终优于纵向**：所有模型 P90 横向 < 0.2m
5. **Oblique 泛化优于 Frontal**：Oblique 0.73m vs Frontal 1.22m，数据分布差异更小

---

## 6. 当前运行状态

### 6.1 正在进行的训练

| 项 | 值 |
|----|-----|
| **配置文件** | `exps/det/CRN_r50_256x704_128x128_4key_frontal_480.py` |
| **BEV 分辨率** | **480 格** (0.5m/格) |
| **场景** | Frontal（场景一） |
| **训练数据** | `frontal_train.pkl`（660 帧，闭卷） |
| **验证数据** | `frontal_test.pkl`（800 帧） |
| **策略** | 远端 BBox 2.5× + Radius 保守 + Depth Loss 距离感知 |
| **优化器** | AdamW lr=5e-5 (若未加载 ckpt 则为 1e-4) |
| **patience** | 5 |
| **check_val_every_n_epoch** | 2 |

### 6.2 权重文件路径

| 权重 | 路径 | 分辨率 | 场景 | 说明 |
|------|------|--------|------|------|
| version_1 | `outputs/det/CRN_frontal_ultimate_finetune_closed/lightning_logs/version_1/checkpoints/last.ckpt` | 480 格 | Frontal | 旧策略训练的最终权重 |
| version_0 | `outputs/det/CRN_oblique_ultimate_finetune_closed/lightning_logs/version_0/checkpoints/last.ckpt` | 480 格 | Oblique | Oblique 旧权重 |
| version_6 | `outputs/det/CRN_frontal_ultimate_finetune_closed/lightning_logs/version_6/checkpoints/last.ckpt` | 960 格 | Frontal | Route B 960 格中断点（已废弃） |

### 6.3 当前生效的优化策略

```yaml
# bev_depth_head_det.py — Frontal 策略
BBox Regression Weight:
  x > 150m:        2.5   # 远端加码
  100m < x ≤ 150m: 2.0   # 过渡段加强
  50m < x ≤ 100m:  2.0   # 中间段重点
  x ≤ 50m:         1.2

Gaussian Radius (保守):
  x > 150m:        2.0×
  100m < x ≤ 150m: 1.5×
  50m < x ≤ 100m:  1.2×

Heatmap Weight:    不加权 (hw=ones, 避免虹吸)

Depth Loss (距离感知):
  远端 (162-240m): 2.0×
  中端 (82-160m):  1.0×
  近端 (2-80m):    0.5×
```

---

## 7. 命令全集（Command Reference）

> 所有命令均以 `cd /home/wy666/CRN && export PYTHONPATH=.` 为前提，标注 `(base)` 的行表示每次新开终端需先执行。

### 7.1 训练命令

#### 7.1.1 Frontal 场景（场景一）

```bash
# ========== 480 格（当前主力，推荐） ==========

# (base) 前置环境
cd /home/wy666/CRN && export PYTHONPATH=.
export TQDM_MINITERS=1          # 进度条实时刷新

# 全新训练（从零开始）
python -u exps/det/CRN_r50_256x704_128x128_4key_frontal_480.py -b 1

# 续训 / 微调（从 checkpoint 恢复 optimizer/scheduler/epoch 状态）
python -u exps/det/CRN_r50_256x704_128x128_4key_frontal_480.py -b 1 \
    --ckpt_path outputs/det/CRN_frontal_ultimate_finetune_closed/lightning_logs/version_1/checkpoints/last.ckpt

# 指定 seed
python -u exps/det/CRN_r50_256x704_128x128_4key_frontal_480.py -b 1 --seed 42


# ========== 960 格（Route B，RTX 5060 不推荐，已搁置） ==========

# 全新训练
python -u exps/det/CRN_r50_256x704_128x128_4key_frontal.py -b 1

# 闭卷微调版（含 version_11 权重加载逻辑）
python -u exps/det/CRN_frontal_ultimate_finetune_closed.py -b 1


# ========== 通用参数说明 ==========
#   -b N / --batch_size_per_device N    batch size（默认 1）
#   --ckpt_path <path>                  续训起点，恢复权重+优化器+scheduler+epoch
#   --seed N                            随机种子（默认 0）
#   -e / --evaluate                     仅评估验证集（不训练）
#   -p / --predict                      仅推理测试集（不训练）
```

#### 7.1.2 Oblique 场景（场景二）

```bash
# ========== 480 格 ==========
cd /home/wy666/CRN && export PYTHONPATH=.
export TQDM_MINITERS=1

# 全新训练
python -u exps/det/CRN_r50_256x704_128x128_4key_oblique_480.py -b 1

# 续训
python -u exps/det/CRN_r50_256x704_128x128_4key_oblique_480.py -b 1 \
    --ckpt_path <path_to_last.ckpt>


# ========== 960 格 ==========
python -u exps/det/CRN_r50_256x704_128x128_4key_oblique.py -b 1
```

#### 7.1.3 全场景训练（已废弃，仅参考）

```bash
# 全局 480 格（不推荐 — 1266 帧混合训练）
python -u exps/det/CRN_r50_256x704_128x128_4key_480.py -b 1

# 全局 960 格
python -u exps/det/CRN_r50_256x704_128x128_4key.py -b 1
```

---

### 7.2 评估命令（MOTA / P90 / ATE 指标）

#### 7.2.1 Frontal 480 格评估

```bash
cd /home/wy666/CRN && export PYTHONPATH=.

# 测试集评估（泛化性能，800 帧）
python scripts/data_preprocessing_final/check/eval_frontal_480.py \
    --ckpt <ckpt_path> \
    --out_dir work_dirs/eval_frontal_newstrat_test \
    --scene frontal \
    --eval_split val

# 训练集评估（非泛化性能，660 帧）
python scripts/data_preprocessing_final/check/eval_frontal_480.py \
    --ckpt <ckpt_path> \
    --out_dir work_dirs/eval_frontal_newstrat_train \
    --scene frontal \
    --eval_split train

# 仅评估重度拥堵子集
python scripts/data_preprocessing_final/check/eval_frontal_480.py \
    --ckpt <ckpt_path> \
    --out_dir work_dirs/eval_frontal_heavy \
    --scene frontal --eval_split val --congestion heavy

# 仅评估中度拥堵子集
python scripts/data_preprocessing_final/check/eval_frontal_480.py \
    --ckpt <ckpt_path> \
    --out_dir work_dirs/eval_frontal_moderate \
    --scene frontal --eval_split val --congestion moderate

# 限制帧数（快速验证）
python scripts/data_preprocessing_final/check/eval_frontal_480.py \
    --ckpt <ckpt_path> \
    --out_dir work_dirs/eval_quick \
    --scene frontal --eval_split val --max_frames 100

# 调高置信度阈值（默认 0.35）
python scripts/data_preprocessing_final/check/eval_frontal_480.py \
    --ckpt <ckpt_path> \
    --out_dir work_dirs/eval_highconf \
    --scene frontal --eval_split val --score_thr 0.5
```

#### 7.2.2 Oblique 480 格评估

```bash
cd /home/wy666/CRN && export PYTHONPATH=.

# 测试集评估（泛化性能，784 帧）
python scripts/data_preprocessing_final/check/eval_oblique_480.py \
    --ckpt <ckpt_path> \
    --out_dir work_dirs/eval_oblique_test \
    --scene oblique \
    --eval_split val

# 训练集评估（非泛化性能，606 帧）
python scripts/data_preprocessing_final/check/eval_oblique_480.py \
    --ckpt <ckpt_path> \
    --out_dir work_dirs/eval_oblique_train \
    --scene oblique \
    --eval_split train
```

#### 7.2.3 960 格评估（旧版，含可视化）

```bash
cd /home/wy666/CRN && export PYTHONPATH=.

# Frontal 960 格
python scripts/data_preprocessing_final/check/eval_frontal.py \
    --ckpt <ckpt_path> \
    --out_dir work_dirs/eval_frontal_960 \
    --scene frontal --eval_split val

# Oblique 960 格
python scripts/data_preprocessing_final/check/eval_frontal.py \
    --ckpt <ckpt_path> \
    --out_dir work_dirs/eval_oblique_960 \
    --scene oblique --eval_split val
```

#### 7.2.4 评估脚本通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ckpt` | str | **必填** | checkpoint 路径 |
| `--out_dir` | str | **必填** | 输出目录 |
| `--scene` | str | `frontal` | `frontal` / `oblique` |
| `--eval_split` | str | `val` | `train`=非泛化 / `val`=泛化 |
| `--score_thr` | float | `0.35` | 置信度阈值 |
| `--congestion` | str | `all` | `all` / `moderate` / `heavy` |
| `--max_frames` | int | `999999` | 最大评估帧数（快速验证用） |

---

### 7.3 CSV 导出命令（追踪格式，按拥堵自动拆分）

```bash
cd /home/wy666/CRN && export PYTHONPATH=.

# ========== 单条导出 ==========

# 场景一 Frontal — 测试集（→ 输出到 泛化/ 子目录）
python scripts/data_preprocessing_final/check/export_tracking_csv.py \
    --ckpt outputs/det/CRN_frontal_ultimate_finetune_closed/lightning_logs/version_1/checkpoints/last.ckpt \
    --out_dir work_dirs/export_frontal_v1_480 \
    --scene frontal --resolution 480 --eval_split val

# 场景一 Frontal — 训练集（→ 输出到 非泛化/ 子目录）
python scripts/data_preprocessing_final/check/export_tracking_csv.py \
    --ckpt outputs/det/CRN_frontal_ultimate_finetune_closed/lightning_logs/version_1/checkpoints/last.ckpt \
    --out_dir work_dirs/export_frontal_v1_480 \
    --scene frontal --resolution 480 --eval_split train

# 场景二 Oblique — 测试集
python scripts/data_preprocessing_final/check/export_tracking_csv.py \
    --ckpt outputs/det/CRN_oblique_ultimate_finetune_closed/lightning_logs/version_0/checkpoints/last.ckpt \
    --out_dir work_dirs/export_oblique_v0_480 \
    --scene oblique --resolution 480 --eval_split val

# 场景二 Oblique — 训练集
python scripts/data_preprocessing_final/check/export_tracking_csv.py \
    --ckpt outputs/det/CRN_oblique_ultimate_finetune_closed/lightning_logs/version_0/checkpoints/last.ckpt \
    --out_dir work_dirs/export_oblique_v0_480 \
    --scene oblique --resolution 480 --eval_split train


# ========== 一键全部导出 ==========
bash scripts/data_preprocessing_final/check/export_all_csv.sh
```

#### export_tracking_csv.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ckpt` | str | **必填** | checkpoint 路径 |
| `--out_dir` | str | **必填** | 输出根目录（自动创建 泛化/非泛化 子目录） |
| `--scene` | str | `frontal` | `frontal` / `oblique` |
| `--resolution` | str | `960` | `480` / `960`（决定用哪个模型类） |
| `--eval_split` | str | `val` | `train`→非泛化 / `val`→泛化 |
| `--score_thr` | float | `0.35` | 置信度阈值 |
| `--match_dist` | float | `2.0` | Hungarian 匹配距离阈值 (m) |
| `--max_frames` | int | `999999` | 最大帧数 |

#### 输出文件命名规则

| 拥堵级别 | 泛化目录 | 非泛化目录 | 全量pred CSV | TP+GT CSV |
|----------|----------|------------|-------------|-----------|
| 中度 | `泛化/S1m.csv` | `非泛化/S1m.csv` | `frame,x,y,z,w,l,h,yaw,vx,vy` | `frame,gt_id,gt_x,gt_y,x,y,z,w,l,h,yaw,vx,vy` |
| 重度 | `泛化/S1s.csv` | `非泛化/S1s.csv` | 同上 | 同上 |
| 中度 | — | — | `S2m.csv` | `S2m.csv` |
| 重度 | — | — | `S2s.csv` | `S2s.csv` |

---

### 7.4 数据预处理命令

```bash
cd /home/wy666/CRN && export PYTHONPATH=.

# ========== 查看当前预处理配置 ==========
python scripts/data_preprocessing_final/pipeline.py --show-config


# ========== 阶段1: 雷达 CSV → 18通道二进制 ==========
# 每批数据执行一次（按 offset + csv 确定唯一批次）

# Frontal offset=0 (中度, 100帧)
python scripts/data_preprocessing_final/pipeline.py \
    --scenario frontal --offset 0 \
    --csv data/radar_data/S1m_flag.csv \
    --stage radar2bin

# Frontal offset=100000 (重度, 260帧)
python scripts/data_preprocessing_final/pipeline.py \
    --scenario frontal --offset 100000 \
    --csv data/radar_data/S1s_flag.csv \
    --stage radar2bin

# Frontal offset=200000 (中度, 400帧)
python scripts/data_preprocessing_final/pipeline.py \
    --scenario frontal --offset 200000 \
    --csv data/radar_data/S2m_flag.csv \
    --stage radar2bin

# Oblique offset=400000 (中度, 300帧)
python scripts/data_preprocessing_final/pipeline.py \
    --scenario oblique --offset 400000 \
    --csv data/radar_data/oblique_S1m_flag.csv \
    --stage radar2bin

# Oblique offset=700000 (重度, 6帧)
python scripts/data_preprocessing_final/pipeline.py \
    --scenario oblique --offset 700000 \
    --csv data/radar_data/oblique_S1s_flag.csv \
    --stage radar2bin


# ========== 阶段2: 生成 BEV + PV 衍生特征（自动处理所有已存在的bin） ==========
python scripts/data_preprocessing_final/pipeline.py --stage derivatives


# ========== 阶段3: 生成深度真值（自动处理所有场景） ==========
python scripts/data_preprocessing_final/pipeline.py --stage depth_gt


# ========== 阶段4: 构建 nuScenes info PKL（自动合并所有场景） ==========
python scripts/data_preprocessing_final/pipeline.py --stage build_infos

# 输出文件：
#   data/my_formatted_data/nuscenes_infos_train.pkl      (全量训练集)
#   data/my_formatted_data/nuscenes_infos_test.pkl       (全量测试集)
#   data/my_formatted_data/nuscenes_infos_frontal_train.pkl
#   data/my_formatted_data/nuscenes_infos_frontal_test.pkl
#   data/my_formatted_data/nuscenes_infos_oblique_train.pkl
#   data/my_formatted_data/nuscenes_infos_oblique_test.pkl


# ========== 阶段2b: 合并新批次数据 ==========
python scripts/data_preprocessing_final/pipeline.py \
    --stage merge_new_data --offset <新offset>
```

#### pipeline.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--scenario` | str | `frontal` | 场景名（决定雷达高度 H 和安装角 yaw） |
| `--stage` | str | 无 | 阶段名：`radar2bin` / `derivatives` / `depth_gt` / `build_infos` / `merge_new_data` |
| `--csv` | str | 无 | 雷达 CSV 路径（radar2bin 阶段必需） |
| `--offset` | int | 无 | 帧号偏移量（radar2bin / merge_new_data 阶段必需） |
| `--show-config` | — | — | 打印当前配置并退出 |

---

### 7.5 数据验证命令

```bash
cd /home/wy666/CRN && export PYTHONPATH=.

# ========== 统一验证（推荐） ==========
# 随机抽查 5 帧，显示雷达点云 + 3D 框
python scripts/data_preprocessing_final/check/verify_data.py

# 指定场景 + 帧数
python scripts/data_preprocessing_final/check/verify_data.py \
    --scenario frontal --num 10

# 指定单帧（如 offset=0 的第 50 帧 → frame 50）
python scripts/data_preprocessing_final/check/verify_data.py \
    --scenario oblique --frame 400050 --num 1

# 不显示雷达点云（只看 3D 框）
python scripts/data_preprocessing_final/check/verify_data.py --no-radar

# 不显示 3D 框（只看雷达点云）
python scripts/data_preprocessing_final/check/verify_data.py --no-boxes

# 交互模式（逐帧翻页）
python scripts/data_preprocessing_final/check/verify_data.py --interactive

# 指定帧范围
python scripts/data_preprocessing_final/check/verify_data.py --range 0 100


# ========== 预处理管线分步验证 ==========
# Stage 1: 检查雷达 CSV → 18ch bin 是否正确
python scripts/data_preprocessing_final/check/check_step1.py

# Stage 2: 检查 BEV/PV 衍生特征
python scripts/data_preprocessing_final/check/check_step2.py

# Stage 3: 检查深度 GT
python scripts/data_preprocessing_final/check/check_step3.py

# Stage 4: 检查最终 PKL 加载
python scripts/data_preprocessing_final/check/check_step4.py

# 批量检查 step4（多帧）
python scripts/data_preprocessing_final/check/check_step4_batch.py


# ========== 模型属性探测 ==========
python scripts/data_preprocessing_final/check/probe_model_attrs.py
python scripts/data_preprocessing_final/check/probe.py
```

#### verify_data.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--frame` | int | 无 | 指定帧号（如 `50` = offset0的第50帧） |
| `--scenario` | str | 无 | `frontal` / `oblique`（过滤场景） |
| `--num` | int | `5` | 随机抽查帧数 |
| `--no-radar` | flag | False | 不显示雷达点云 |
| `--no-boxes` | flag | False | 不显示 3D 标注框 |
| `--interactive` | flag | False | 逐帧翻页模式 |
| `--range` | int int | 无 | 帧范围 `START END`（含两端） |

---

### 7.6 分析与对比命令

```bash
cd /home/wy666/CRN && export PYTHONPATH=.

# ========== 改进前后误差曲线对比 ==========
python scripts/data_preprocessing_final/check/compare_error_curves.py \
    --before work_dirs/eval_baseline_test/eval_stats.pkl \
    --after work_dirs/eval_newstrat_test/eval_stats.pkl \
    --before_moderate work_dirs/eval_baseline_moderate/eval_stats.pkl \
    --after_moderate work_dirs/eval_newstrat_moderate/eval_stats.pkl \
    --before_heavy work_dirs/eval_baseline_heavy/eval_stats.pkl \
    --after_heavy work_dirs/eval_newstrat_heavy/eval_stats.pkl \
    --out_dir work_dirs/comparison_plots \
    --label_before "改进前(version_1)" \
    --label_after "改进后(新策略)"

# ========== 距离分段误差对比 ==========
python scripts/data_preprocessing_final/check/compare_dist_segments.py


# ========== GT/Pred 逐帧对比可视化 ==========
python scripts/data_preprocessing_final/check/compare_gt_pred.py


# ========== 拥堵级别批量评估 ==========
python scripts/data_preprocessing_final/check/batch_eval_congestion.py
```

#### compare_error_curves.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--before` | str | **必填** | 改进前全局 eval_stats.pkl |
| `--after` | str | **必填** | 改进后全局 eval_stats.pkl |
| `--before_moderate` | str | `''` | 改进前中度拥堵 pkl |
| `--after_moderate` | str | `''` | 改进后中度拥堵 pkl |
| `--before_heavy` | str | `''` | 改进前重度拥堵 pkl |
| `--after_heavy` | str | `''` | 改进后重度拥堵 pkl |
| `--out_dir` | str | `work_dirs/comparison_plots` | 输出目录 |
| `--label_before` | str | `改进前` | 改进前图例标签 |
| `--label_after` | str | `改进后` | 改进后图例标签 |

---

### 7.7 模型推理可视化命令

```bash
cd /home/wy666/CRN && export PYTHONPATH=.

# ========== 单帧推理可视化 ==========
python scripts/data_preprocessing_final/check/model_inference_vis.py

# ========== 多帧推理可视化（v2） ==========
python scripts/data_preprocessing_final/check/model_inference_vis2.py
```

---

### 7.8 系统运维命令

```bash
# ========== 进程管理 ==========
ps aux | grep python | grep -v grep        # 查看所有 Python 训练进程
pkill -9 -f python                          # 强制杀掉所有 Python 进程
pkill -9 -f CRN_frontal                     # 精准杀掉 Frontal 训练


# ========== GPU 监控 ==========
nvidia-smi                                   # 一次性查看 GPU 状态
watch -n 1 nvidia-smi                        # 每秒刷新 GPU 状态
nvidia-smi dmon -s u                         # GPU 利用率实时流


# ========== Checkpoint 管理 ==========
# 按时间排序查看所有权重文件
ls -lht outputs/det/*/lightning_logs/version_*/checkpoints/

# 查看特定版本
ls -lht outputs/det/CRN_frontal_ultimate_finetune_closed/lightning_logs/version_1/checkpoints/

# 查看 checkpoint 文件大小趋势
du -sh outputs/det/*/lightning_logs/version_*/checkpoints/last.ckpt


# ========== tmux 会话管理（防训练中断） ==========
tmux new -s train                            # 新建名为 train 的会话
tmux attach -t train                         # 挂载到 train 会话
tmux ls                                      # 查看所有会话
# Ctrl+B, D                                  # 分离当前会话（不杀进程）
# Ctrl+B, [                                  # 进入滚动模式（q 退出）
tmux kill-session -t train                   # 彻底关闭 train 会话


# ========== 磁盘与文件 ==========
du -sh data/                                 # 数据目录总大小
du -sh outputs/                              # 输出目录总大小
find outputs/ -name "*.ckpt" -mtime +7       # 找出一周前的旧权重


# ========== 推理输出验证 ==========
# 检查评估是否生成了关键输出文件
ls work_dirs/<eval_dir>/*.csv work_dirs/<eval_dir>/*.pkl
cat work_dirs/<eval_dir>/summary.txt
```

---

### 7.9 批量操作速查（最常用）

```bash
# ═══════════════════════════════════════════════════════════════
# 新权重完整评估流水线（以 Frontal 新策略为例）
# ═══════════════════════════════════════════════════════════════

CKPT="outputs/det/CRN_r50_256x704_128x128_4key_frontal_480/lightning_logs/version_0/checkpoints/last.ckpt"

# Step 1: 评估测试集（泛化）
python scripts/data_preprocessing_final/check/eval_frontal_480.py \
    --ckpt $CKPT --out_dir work_dirs/eval_final_test \
    --scene frontal --eval_split val

# Step 2: 评估训练集（非泛化）
python scripts/data_preprocessing_final/check/eval_frontal_480.py \
    --ckpt $CKPT --out_dir work_dirs/eval_final_train \
    --scene frontal --eval_split train

# Step 3: 导出 CSV
python scripts/data_preprocessing_final/check/export_tracking_csv.py \
    --ckpt $CKPT --out_dir work_dirs/export_final \
    --scene frontal --resolution 480 --eval_split val

python scripts/data_preprocessing_final/check/export_tracking_csv.py \
    --ckpt $CKPT --out_dir work_dirs/export_final \
    --scene frontal --resolution 480 --eval_split train

# Step 4: 对比改进前后
python scripts/data_preprocessing_final/check/compare_error_curves.py \
    --before work_dirs/eval_v1_test/eval_stats.pkl \
    --after work_dirs/eval_final_test/eval_stats.pkl \
    --out_dir work_dirs/comparison_final_vs_v1
```


---

## 8. 待办事项

### 8.1 高优先级

- [ ] **完成 480 格 Frontal 新策略训练**（远端 BBox 2.5× + Radius 保守 + Depth 距离感知）
- [ ] **评估新策略在 train/test 上的性能**，对比 version_1 基线
- [ ] 导出新权重的 CSV，分析远端检测率改善程度
- [ ] 如果远端改善不明显，尝试中端-远端分段 BBox weight 进一步调优

### 8.2 中优先级

- [ ] 480 格 Oblique 同理优化（远端 BBox + Depth 距离感知）
- [ ] 评估 Oblique version_0 在 train/test 上的性能
- [ ] 导出 Oblique CSV

### 8.3 低优先级

- [ ] 远端样本过采样 DataLoader
- [ ] 分阶段训练（先全量→冻结 Backbone→再全量）
- [ ] RTX 5060 升级方案评估（显存瓶颈）

---

## 9. 经验教训

1. **训练前确认数据划分**：多次出问题都是因为 `train_info_paths` 混入了 test 或用错了配置文件
2. **BEV 分辨率不是越高越好**：960 格在 5060 上太慢（~0.3 step/s），480 格（~2 step/s）更实用
3. **ctmd 独立终端**：VS Code 内置终端关闭会杀 WSL → 训练中断 → 必须用 Windows Terminal + tmux
4. **配置继承链要清晰**：基类→子类→闭卷微调类，每层覆盖了什么要明确注释
5. **评估脚本要永久化**：不要临时改配置做评估，创建独立的 480 格评估方案
6. **面向文档工作**：每次改动后更新此文件，避免记忆丢失

---

> **此文件随项目同步更新。每次完成一个实验或配置变更后，请更新对应章节。**
