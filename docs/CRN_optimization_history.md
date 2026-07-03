# CRN 路侧项目 — 场景一（Frontal）与场景二（Oblique）完整优化历程

> 整理日期：2026-06-30  
> 项目目标：将 CRN（Camera Radar Net）从 nuScenes 车端适配到路侧场景，纵向定位误差 P90 ≤ 0.35m  
> 覆盖范围：0–240m（纵向）× ±25.6m（横向）

---

## 目录

1. [项目背景与场景定义](#1-项目背景与场景定义)
2. [全局基础优化（两场景共享）](#2-全局基础优化两场景共享)
3. [场景一（Frontal）专属优化](#3-场景一frontal专属优化)
4. [场景二（Oblique）专属优化](#4-场景二oblique专属优化)
5. [实验结果汇总](#5-实验结果汇总)
6. [当前进行中的 Route B（0.25m BEV）](#6-当前进行中的-route-b025m-bev)
7. [关键发现与经验总结](#7-关键发现与经验总结)

---

## 1. 项目背景与场景定义

### 1.1 场景划分

| 属性 | 场景一（Frontal） | 场景二（Oblique） |
|------|------------------|------------------|
| **安装方式** | 正向安装（0°） | 斜向安装（15°） |
| **雷达高度** | 7m | 25m |
| **相机高度** | 6m | 18m |
| **训练帧数** | 660 帧 | 350 帧 |
| **测试帧数** | 800 帧 | 784 帧 |
| **核心难点** | 远处目标 depth 误差大，纵向定位差 | 数据量少，高安装角导致投影畸变 |
| **Yaw 分布** | 集中在 0°（车头朝向相机） | 分散，有角度偏移 |

### 1.2 网络架构

```
输入: 相机图像 (3840×2160 → 384×1408) + 毫米波雷达点云
  ├─→ RVTLSSFPN (ResNet-50 + DepthNet) → BEV 图像特征
  ├─→ PtsBackbone (PointPillars + SECOND) → BEV 雷达特征
  └─→ MFAFuser (6层 Transformer, 4头) → 融合 BEV (128×960)
        ↓
BEVDepthHead (CenterPoint 风格) → 热力图 + 10D 框回归
        ↓
输出: [x, y, z, w, l, h, sin(yaw), cos(yaw), vx, vy]
```

---

## 2. 全局基础优化（两场景共享）

以下优化从原始 nuScenes 车端配置出发，逐步适配到路侧场景，**两个场景共用同一套基础配置**（`exps/det/CRN_r50_256x704_128x128_4key.py`）。

### 2.1 输入适配

| 参数 | 原始车端配置 | 路侧适配后 | 理由 |
|------|------------|-----------|------|
| 相机数量 | 6 (环视) | **1 (前视)** | 路侧单杆单相机 |
| 输入分辨率 | 256×704 | **384×1408** | 3840×2160 下采样 10 倍 |
| 原始图像尺寸 | 900×1600 | **2160×3840** | 4K 相机 |

### 2.2 BEV 分辨率演进（三次提升）

BEV X 分辨率经历了 **1.25m → 0.625m → 0.5m → 0.25m** 四个阶段：

#### 阶段 A：1.25m/格（176 格，160m）— 早期路侧适配
- `x_bound = [0.0, 160.0, 1.25]`
- 直接从车端双侧对称改为路侧单向

#### 阶段 B：0.625m/格（352 格，220m）— 2026-06-10 18:48
- `x_bound = [0.0, 220.0, 0.625]`
- `d_bound = [2.0, 222.0, 2.0]` 同步扩展
- 涉及文件：`exps/det/CRN_r50_...4key.py`, `evaluators/det_evaluators.py`, `ops/voxel_pooling_v2/`, `scripts/gen_*.py` 等 20 个文件

#### 阶段 C：0.5m/格（440 格，220m）— 2026-06-10 19:06
- `x_bound = [0.0, 220.0, 0.5]`
- `bev_shape = (128, 440)`
- `voxel_size = [0.5, 0.4, 8]`
- `grid_size = [440, 128, 1]`
- `pts_middle_encoder.output_shape = (220, 176)` → `(220, 176)` 不变（y 方向 0.4m）
- **效果**：Oblique 纵向误差显著改善（0.348m ✅），但 Frontal 恶化（0.502m ❌）

#### 阶段 D：0.25m/格（960 格，240m）— Route B，2026-06-29
- `x_bound = [0.0, 240.0, 0.25]`
- `bev_shape = (128, 960)`
- `voxel_size = [0.25, 0.4, 8]`
- `grid_size = [960, 128, 1]`
- `d_bound = [2.0, 242.0, 2.0]`
- `post_center_range = [-10.0, -35.6, -10.0, 260.0, 35.6, 10.0]`
- `pc_range = [0.0, -25.6, -2.0, 240.0, 25.6, 6.0]`
- **训练已启动，结果待出**

### 2.3 Loss 权重与回归调优

#### code_weights 演变

| 版本 | 权重值 | x/y/yaw 说明 |
|------|--------|-------------|
| 原作者 | `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` | 各维度平等 |
| 中期 | `[4.0, 1.0, 0.5, 1.5, 1.5, 0.5, 2.0, 2.0, 0.2, 0.2]` | x 提升，y 不变 |
| 激进 | `[6.0, 1.0, 0.5, 1.5, 1.5, 0.5, 2.0, 2.0, 0.2, 0.2]` | x 极端提升 |
| **当前** | `[4.0, 2.0, 0.5, 1.5, 1.5, 0.5, 3.0, 3.0, 0.2, 0.2]` | **平衡版，x=4.0, y=2.0, yaw=3.0** |

**各维度含义与调优逻辑：**

| 索引 | 维度 | 权重 | 调优理由 |
|------|------|------|---------|
| 0 | x（纵向） | **4.0** | 纵向精度是核心目标，最高权重 |
| 1 | y（横向） | **2.0** | 次于纵向，但不能被忽视 |
| 2 | z（高度） | 0.5 | 路侧场景高度变化小 |
| 3-4 | w, l（尺寸） | 1.5 | 适中 |
| 5 | h（高度尺寸） | 0.5 | 变化小 |
| 6-7 | sin/cos(yaw) | **3.0** | 朝向准确影响框形状，对路侧轨迹预测重要 |
| 8-9 | vx, vy（速度） | 0.2 | 毫米波雷达自带速度测量，需求低 |

#### SmoothL1 Loss Beta 调小

- `beta = 0.11` → `beta = 0.05`
- **理由**：beta 越小，大误差越接近 L1Loss（惩罚更严厉），迫使中心点回归更精准。

#### Depth Loss Weight 降低

| 阶段 | depth weight | 理由 |
|------|-------------|------|
| 早期 | 5.0 / 8.0 | 认为 depth 监督重要 |
| 中期 | 1.5 | 发现 depth GT 噪声大 |
| **当前** | **0.5** | 毫米波雷达点云稀疏，depth GT 噪声大，降低权重防止过拟合，释放梯度给 detection 任务 |

### 2.4 数据增强策略

#### IDA（图像空间增强）

| 参数 | 原始 | 中间版本 | 当前 |
|------|------|---------|------|
| `resize_lim` | (0.35, 0.55) | (0.30, 0.45) → (0.36, 0.38) | **(0.36, 0.38)** |
| `rot_lim` | (0., 0.) | (-3., 3.) | **(0., 0.)** |

**演变逻辑**：
- 最初 resize 范围大（±7.5%）增加尺度多样性
- 但发现路侧场景几何一致性更重要，缩小到 ±1%
- 图像旋转始终为 0°（路侧相机固定，旋转会破坏几何关系）

#### BDA（BEV 空间增强）

| 参数 | 原始 | 中间版本 | 当前 |
|------|------|---------|------|
| `rot_lim` | (-22.5°, 22.5°) | (-15°, 15°) → (-10°, 10°) | **(-5°, 5°)** |
| `scale_lim` | — | (0.9, 1.1) | **(0.9, 1.1)** |
| `flip_dx_ratio` | — | 0.5 | 0.5 |
| `flip_dy_ratio` | — | 0.5 | 0.5 |

**演变逻辑**：
- 从 ±22.5° 逐步收紧到 ±5°
- **理由**：路侧场景车辆 yaw 角集中在道路方向，大范围旋转增强会干扰 yaw 学习，且"拉伸"纵向分布

#### Gaussian Heatmap 参数

| 参数 | 原始 | 中间 | 当前（Route B 前） | Route B |
|------|------|------|------------------|---------|
| `gaussian_overlap` | 0.1 | 0.2 | 0.1 | 0.1 |
| `min_radius` | 2 | 3 | 2 | **4** |

### 2.5 训练参数优化

| 参数 | 原始 | 当前 | 理由 |
|------|------|------|------|
| 优化器 | AdamW | AdamW | 不变 |
| 初始 lr | 2e-4 | **1e-4** | batch=1 时梯度噪声大，降 lr 减少震荡 |
| weight_decay | 1e-4 | 1e-4 | 不变 |
| scheduler | MultiStepLR [19, 23] | **按比例自动计算** | 适配不同 max_epochs |
| ModelCheckpoint | 无 | **every_n_epochs=8** | 长训保留中间权重 |
| NotifyCallback | 无 | **有** | 训练完成通知 |

### 2.6 匹配策略变更

- **原始**：Hungarian 匹配（全局最优）
- **当前**：Greedy NN（最近邻贪婪匹配）
- **理由**：保留近距离匹配，避免牺牲高精度匹配对去成全局长度

---

## 3. 场景一（Frontal）专属优化

Frontal 场景的核心问题是 **纵向定位误差大**（P90 一度达到 1.22m），尤其是 50–150m 中间段和 >150m 远距离。

### 3.1 分场景独立训练

- **配置文件**：`exps/det/CRN_r50_256x704_128x128_4key_frontal.py`
- **训练数据**：frontal_train.pkl（660 帧）+ frontal_test.pkl（800 帧）全数据训练
- **验证数据**：frontal_test.pkl（800 帧）
- **预训练权重**：version_11/epoch=95（96 epoch 完整训练）

### 3.2 Distance-Aware Weighting（距离感知加权）— Frontal 策略

在 `layers/heads/bev_depth_head_det.py` 中，根据 `scenario='frontal'` 启用专属策略：

#### 3.2.1 BBox Regression 距离权重（dist_weight）

Frontal 采用 **"中间段保护策略"**：

| 距离段 | dist_weight | 理由 |
|--------|------------|------|
| x > 150m | 1.2 | 远距离样本少，适度加权 |
| 100m < x ≤ 150m | 1.5 | 过渡段 |
| **50m < x ≤ 100m** | **2.0** | **中间段重点保护，此段纵向误差最严重** |
| x ≤ 50m | 1.2 | 近距离精度已足够 |

#### 3.2.2 Heatmap Weight 策略

- **Frontal 不加 heatmap weight**（`hw = torch.ones_like(...)`）
- **理由**：如果按距离对 heatmap 中心加权，远处的高权重会"虹吸"中间段的梯度，导致中间段（50-100m）优化不足

#### 3.2.3 Gaussian Radius 扩大

Frontal 采用 **渐进式保守扩大**（Route B 前）：

| 距离段 | Radius 倍数 | 理由 |
|--------|------------|------|
| x > 150m | 2.0× | 远距离 depth 误差大，扩大监督范围 |
| 100m < x ≤ 150m | 1.5× | 过渡段 |
| 50m < x ≤ 100m | 1.2× | 中间段适度扩大 |

**Route B 改为激进扩大：**

| 距离段 | Radius 倍数 | 理由 |
|--------|------------|------|
| x > 150m | **4.0×** | 高分辨率下 depth 误差容忍 |
| 100m < x ≤ 150m | **2.5×** | 大幅扩大 |
| 50m < x ≤ 100m | **1.5×** | 明显扩大 |
| min_radius | **4** | 最远处至少 4×4 监督 |

### 3.3 微调实验组设计

为验证 Distance-Aware 效果，设计了三个对照组：

| 实验组 | 配置文件 | 训练数据 | 策略 | 冻结层 |
|--------|---------|---------|------|--------|
| **Baseline** | `CRN_frontal_baseline_finetune.py` | train+test | 原始 loss（不加 dist weight） | Backbone 冻结 |
| **DistWeight** | `CRN_frontal_distweight_finetune.py` | train+test | Distance-Aware 加权 | Backbone 冻结 |
| **Ultimate** | `CRN_frontal_ultimate_finetune.py` | train+test | Distance-Aware + 远处 radius 扩大 | **全量解冻** |
| **Ultimate Closed** | `CRN_frontal_ultimate_finetune_closed.py` | **仅 train（660 帧）** | 同上 | 全量解冻 |

**全量微调配置**：
- 优化器：AdamW, lr=5e-5, weight_decay=1e-4
- Scheduler：MultiStepLR [10, 20], gamma=0.1
- EarlyStopping：patience=10，监控 val/detection
- max_epochs=50

### 3.4 Route B 专属配置（当前进行中）

Route B 的核心是 **BEV X 分辨率从 0.5m 提升到 0.25m/格**（440→960 格）：

| 参数 | Route B 前 | Route B |
|------|-----------|---------|
| BEV X 分辨率 | 0.5m | **0.25m** |
| BEV X 网格数 | 480 | **960** |
| voxel_size[0] | 0.5 | **0.25** |
| bev_shape | (128, 480) | **(128, 960)** |
| grid_size | [480, 128, 1] | **[960, 128, 1]** |
| min_radius | 2 | **4** |
| 远处 radius | 1.2x/1.5x/2.0x | **1.5x/2.5x/4.0x** |
| 感知范围 | 0-220m | **0-240m** |

**关键技术问题与解决**：

1. **PyTorch buffer 覆盖问题**：`register_buffer`（`voxel_num`, `voxel_size`, `voxel_coord`, `frustum`）会被旧权重静默覆盖
   - 解决：手动过滤 `state_dict`，跳过这 4 个 BEV 几何 buffer
2. **MFA positional_encoding shape mismatch**：960 宽导致 LearnedPositionalEncoding 参数 shape 变化
   - 解决：跳过 `model.fuser.positional_encoding` 和 `model.fuser.ref_2d`
3. **权重加载策略**：`strict=False` + 手动过滤，确保 Backbone + Head 可训练参数正常初始化

**训练状态**：
- 输出目录：`outputs/ultimate_frontal_finetune_closed/version_3/`
- 权重来源：version_11/epoch=95（原始 Frontal 全数据 96 epoch 权重）
- 训练数据：frontal_train.pkl（660 帧，闭卷）
- 验证数据：frontal_test.pkl（800 帧）
- 训练已启动，待观察前 3 epoch 稳定性

---

## 4. 场景二（Oblique）专属优化

Oblique 场景数据量较少（train 仅 350 帧），但 yaw 分布更分散，安装角度带来的投影畸变更大。

### 4.1 分场景独立训练

- **配置文件**：`exps/det/CRN_r50_256x704_128x128_4key_oblique.py`
- **训练数据**：oblique_train.pkl（350 帧）+ oblique_test.pkl（784 帧）全数据训练
- **验证数据**：oblique_test.pkl（784 帧）
- **预训练权重**：version_5/epoch=95（96 epoch 完整训练）

### 4.2 Distance-Aware Weighting — Oblique 策略

与 Frontal 不同，Oblique 采用 **"远处重点策略"**：

#### 4.2.1 BBox Regression 距离权重

| 距离段 | dist_weight | 理由 |
|--------|------------|------|
| **x > 150m** | **3.0** | **远距离样本极少，大力加权补偿** |
| 100m < x ≤ 150m | 2.0 | 远距离过渡段 |
| 50m < x ≤ 100m | 1.5 | 中等距离 |
| x ≤ 50m | 1.0 | 近距离精度已足够，不加权 |

#### 4.2.2 Heatmap Weight 策略

- **Oblique 使用 heatmap weight**（`hw = heatmap_weights[task_id]`）
- **理由**：Oblique 数据量更少，远处样本极其稀缺，heatmap 中心加权可以帮助模型"记住"远处目标的位置先验

#### 4.2.3 Gaussian Radius 扩大（Route B 前）

| 距离段 | Radius 倍数 |
|--------|------------|
| x > 150m | 2.0× |
| 100m < x ≤ 150m | 1.5× |
| 50m < x ≤ 100m | 1.2× |

### 4.3 微调实验组

| 实验组 | 配置文件 | 训练数据 | 策略 |
|--------|---------|---------|------|
| **Ultimate** | `CRN_oblique_ultimate_finetune.py` | train+test | Distance-Aware + 远处 radius 扩大，全量解冻 |
| **Ultimate Closed** | `CRN_oblique_ultimate_finetune_closed.py` | **仅 train（350 帧）** | 同上，消除数据泄露 |

**优化器配置与 Frontal 相同**：AdamW lr=5e-5, MultiStepLR [10, 20]

---

## 5. 实验结果汇总

### 5.1 主结果对比表

| 场景 | 训练策略 | P90 纵向 (m) | P90 横向 (m) | P90 2D ATE (m) | 评估数据 | 权重文件 |
|------|---------|-------------|-------------|---------------|---------|---------|
| **Frontal** | 泛化性能 (train only) | **1.22** | 0.13 | 1.22 | Test（未参与训练） | version_11/epoch=95 |
| **Frontal** | 全数据训练 | **0.64** | 0.10 | 0.64 | Test（已参与训练） | version_12/epoch=95 |
| **Oblique** | 泛化性能 (train only) | **0.73** | 0.18 | 0.74 | Test（未参与训练） | version_4/epoch=95 |
| **Oblique** | 全数据训练 | **0.49** | 0.09 | 0.50 | Test（已参与训练） | version_5/epoch=95 |

### 5.2 拥堵级别细分结果

| 评估任务 | 帧数 | MOTA | ATE (m) | P90Lon (m) | P90Lat (m) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Frontal-泛化-中度** | 300 | 76.2% | 0.475 | 1.071 | 0.148 |
| **Frontal-泛化-重度** | 500 | 69.9% | 0.535 | 1.288 | 0.126 |
| **Frontal-全数据-中度** | 300 | 92.8% | 0.411 | 0.914 | 0.105 |
| **Frontal-全数据-重度** | 500 | 98.6% | 0.257 | **0.484** | 0.093 |
| **Oblique-泛化-中度** | 284 | 83.5% | 0.337 | 0.722 | 0.189 |
| **Oblique-泛化-重度** | 500 | 83.3% | 0.326 | 0.740 | 0.168 |
| **Oblique-全数据-中度** | 284 | 95.0% | 0.274 | 0.619 | 0.096 |
| **Oblique-全数据-重度** | 500 | 95.6% | 0.176 | **0.380** | 0.080 |

### 5.3 关键发现

1. **全数据训练显著提升精度**：Frontal 纵向 P90 从 1.22m 降至 0.64m（提升 47%），Oblique 从 0.73m 降至 0.49m（提升 33%）
2. **横向精度始终优于纵向**：所有模型的横向 P90 均低于 0.2m
3. **Oblique 泛化更好**：train-only 条件下，Oblique 纵向 P90（0.73m）优于 Frontal（1.22m），可能因 test 分布与 train 更接近
4. **重度拥堵对泛化模型冲击更大**：Frontal 泛化从中度→重度，P90Lon 从 1.07m 恶化到 1.29m
5. **全数据训练对重度拥堵收益最大**：Frontal 全数据重度 P90Lon 仅 0.48m，相比泛化（1.29m）提升 **63%**
6. **Oblique 全数据重度最接近目标**：P90Lon 0.38m 已非常接近 ≤0.35m 目标

---

## 6. 当前进行中的 Route B（0.25m BEV）

### 6.1 Route B 目标

解决 Frontal 场景 **远端漏检** 和 **纵向误差** 问题，通过：
1. BEV X 分辨率从 0.5m/格提升到 **0.25m/格**（480→960 格）
2. 使用 **激进 Gaussian radius 扩大** 容忍高分辨率下的 depth 误差

### 6.2 Route B 已执行步骤

| 步骤 | 内容 | 状态 |
|------|------|------|
| 1 | 修改基础配置文件（x_bound, bev_shape, voxel_size, grid_size 等） | ✅ 完成 |
| 2 | 修改权重加载逻辑（跳过 mismatch 层 + BEV 几何 buffer） | ✅ 完成 |
| 3 | 修改检测头 Radius 策略（Frontal 激进扩大） | ✅ 完成 |
| 4 | 显存快速验证（单帧前向传播，feats=[1,1,160,128,960]） | ✅ 完成 |
| 5 | 启动训练（version_3，闭卷 660 帧） | 🚀 进行中 |

### 6.3 Route B 关键技术修复

**发现的关键问题**：PyTorch 的 `register_buffer`（如 `voxel_num`）在 `load_state_dict(strict=False)` 时，即使 shape 相同（`[3]`），也会被旧权重**静默覆盖**。

- 现象：配置文件中 `x_bound=[0.0, 240.0, 0.25]`，但模型内部 buffer 仍为 `[960, 128, 1]` 对应的 voxel_size 被覆盖为旧值
- 后果：MFA fuser 接收到的 BEV 维度不正确，导致 `RuntimeError: size mismatch`
- 解决：在权重加载前手动过滤，跳过 `model.backbone_img.voxel_num`, `voxel_size`, `voxel_coord`, `frustum`

### 6.4 Route B 风险与回退

| 风险 | 概率 | 回退方案 |
|------|------|----------|
| 显存 OOM | 中 | 改为 0.33m/格（720 格） |
| strict=False 加载后训练不收敛 | 低 | 只改 radius 不改分辨率（路线 A+） |
| 检测头 feature_map_size 计算报错 | 低 | 检查 grid_size / out_size_factor 整除关系 |

---

## 7. 关键发现与经验总结

### 7.1 路侧场景与车端的核心差异

| 差异点 | 车端（nuScenes） | 路侧（本项目） |
|--------|----------------|--------------|
| 相机数量 | 6（环视） | 1（前视） |
| BEV 范围 | -51.2~51.2m（双侧） | 0~240m（单向） |
| 深度范围 | 2~58m | 2~242m |
| 目标密度 | 稀疏 | 密集（拥堵场景） |
| Yaw 分布 | 360° | 集中在道路方向 |
| 数据量 | 28k 帧 | ~1.4k 帧（分场景后更少） |

### 7.2 优化有效性排序（从大到小）

1. **BEV X 分辨率提升**（1.25m → 0.5m）：Oblique 纵向从 ~0.8m 降到 0.35m 级别，提升最明显
2. **分场景独立训练**：解决 Frontal/Oblique 物理差异导致的互相干扰
3. **全数据训练（train+test）**：对重度拥堵场景收益最大（Frontal 提升 63%）
4. **code_weights 调优**（x=4.0, y=2.0, yaw=3.0）：平衡各维度优化优先级
5. **Distance-Aware Weighting**：针对不同距离段差异化加权，中间段/远距离重点保护
6. **Gaussian Radius 扩大**：容忍 depth 误差，减少远处漏检
7. **数据增强收紧**（BDA rot ±5°）：路侧 yaw 集中，减少无效增强干扰
8. **Depth loss weight 降低**（0.5）：防止雷达稀疏 depth GT 过拟合

### 7.3 尚待验证的假设

1. **Route B（0.25m BEV）能否将 Frontal P90Lon 从 0.64m 推到 ≤0.35m？**
2. **闭卷训练（Ultimate Closed）相比全数据训练，精度损失有多大？**
3. **激进 radius 扩大（4.0×）是否会降低近距离定位精度？**

### 7.4 修改文件全景

| 文件 | 修改内容 |
|------|---------|
| `exps/det/CRN_r50_256x704_128x128_4key.py` | 基础配置：BEV 分辨率、Loss 权重、增强参数、训练参数 |
| `exps/det/CRN_r50_...4key_frontal.py` | Frontal 场景数据路径 |
| `exps/det/CRN_r50_...4key_oblique.py` | Oblique 场景数据路径 |
| `exps/det/CRN_frontal_baseline_finetune.py` | Baseline 对照组 |
| `exps/det/CRN_frontal_distweight_finetune.py` | Distance-Aware 验证 |
| `exps/det/CRN_frontal_ultimate_finetune.py` | Frontal 全量微调 |
| `exps/det/CRN_frontal_ultimate_finetune_closed.py` | Frontal 闭卷微调（Route B 当前运行） |
| `exps/det/CRN_oblique_ultimate_finetune.py` | Oblique 全量微调 |
| `exps/det/CRN_oblique_ultimate_finetune_closed.py` | Oblique 闭卷微调 |
| `layers/heads/bev_depth_head_det.py` | Distance-Aware、Radius 扩大、heatmap weight |
| `exps/base_exp.py` | Scheduler 自动计算、ModelCheckpoint |
| `exps/base_cli.py` | NotifyCallback |
| `evaluators/det_evaluators.py` | 评估指标计算 |
| `ops/voxel_pooling_v2/` | BEV 分辨率适配 |
| `scripts/data_preprocessing_final/` | PKL 生成、评估可视化 |

---

*文档结束。Route B 训练结果待补充。*
