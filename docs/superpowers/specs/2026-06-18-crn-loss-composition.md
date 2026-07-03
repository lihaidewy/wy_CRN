# CRN Loss 构成详解

日期: 2026-06-18

---

## 1. 概述

CRN 模型采用**多任务联合训练**，总 Loss 由**检测任务**和**深度估计任务**两部分组成。

检测任务又细分为**热力图分类**（判断 BEV 网格是否有物体中心）和**框回归**（预测 3D 框的 10 维参数）。

---

## 2. 总 Loss 公式

```python
# exps/det/CRN_r50_256x704_128x128_4key.py:316
return loss_detection + loss_depth
```

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{detection}} + w_{\text{depth}} \cdot \mathcal{L}_{\text{depth}}
$$

当前配置：$w_{\text{depth}} = 0.5$（已从 1.5 降低，缓解过拟合）。

---

## 3. Detection Loss（检测任务）

来自 `self.model.loss(targets, preds)` 的返回值：

```python
loss_detection, loss_heatmap, loss_bbox = self.model.loss(targets, preds)
```

### 3.1 Heatmap Loss（热力图分类）

```python
'loss_cls': dict(type='GaussianFocalLoss', reduction='mean')
```

**作用**：判断 BEV 平面的每个网格上**是否存在物体中心**。

**原理**：
- 将 GT 框的中心点映射到 BEV 网格，生成一个高斯热力图（中心概率为 1，向外衰减）
- 模型预测每个网格的"存在物体中心"的概率
- **GaussianFocalLoss**：对**难例**（靠近中心但不确定的位置）加大惩罚，让模型学会精确定位中心点；对**简单例**（远离中心的位置）降低惩罚，避免正负样本失衡

**对应 TensorBoard 曲线**：`train/heatmap`、`val/heatmap`

---

### 3.2 BBox Loss（框回归）

```python
'loss_bbox': dict(type='SmoothL1Loss', beta=0.05, reduction='mean', loss_weight=0.5)
```

**作用**：对"预测到物体"的网格，回归出框的 **10 维参数**。

**回归目标**：
```
[x, y, z, w, l, h, sin(yaw), cos(yaw), vx, vy]
```

**SmoothL1Loss (beta=0.05)**：
- 对小误差（$< \beta$）使用 L2（平滑求导）
- 对大误差（$\geq \beta$）使用 L1（防止梯度爆炸）
- beta=0.05 很小，意味着**很小的偏差就开始用 L1**，对异常值更鲁棒

**配置中的 `loss_weight=0.5`**：
```python
loss_detection = loss_heatmap + loss_bbox * 0.5
```

即回归 loss 还要再乘 0.5，说明作者认为分类（heatmap）比回归更重要。

**对应 TensorBoard 曲线**：`train/bbox`、`val/bbox`

---

### 3.3 Code Weights（维度平衡权重）

```python
code_weights=[4.0, 2.0, 0.5, 1.5, 1.5, 0.5, 3.0, 3.0, 0.2, 0.2]
```

每个维度在 loss 中的权重：

| 索引 | 维度 | 权重 | 含义 | 调优理由 |
|---|---|---|---|---|
| 0 | x | **4.0** | 纵向位置 | 纵向精度是核心目标，最高权重 |
| 1 | y | **2.0** | 横向位置 | 次于纵向，但不可忽视 |
| 2 | z | 0.5 | 高度 | 路侧场景高度变化小 |
| 3 | w | 1.5 | 宽度 | 尺寸适中 |
| 4 | l | 1.5 | 长度 | 尺寸适中 |
| 5 | h | 0.5 | 高度尺寸 | 变化小 |
| 6 | sin(yaw) | **3.0** | 偏航角正弦 | 朝向准确影响框的形状 |
| 7 | cos(yaw) | **3.0** | 偏航角余弦 | 同上 |
| 8 | vx | 0.2 | 纵向速度 | 毫米波雷达自带速度测量 |
| 9 | vy | 0.2 | 横向速度 | 同上 |

**核心设计**：`x=4.0` 权重最高，确保模型优先优化纵向精度（目标 $\leq 0.35$m）。

---

### 3.4 Detection Loss 汇总

```python
loss_detection = loss_heatmap + loss_bbox * 0.5
```

**对应 TensorBoard 曲线**：`train/detection`、`val/detection`

---

## 4. Depth Loss（深度估计任务）

```python
loss_depth = self.get_depth_loss(depth_labels.cuda(), depth_preds, weight=0.5)
```

**作用**：监督图像分支预测每个像素的深度值。

**输入**：
- `depth_labels`：从雷达点云投影到图像平面的 **sparse depth map**
- `depth_preds`：模型预测的 **dense depth map**

**Loss 类型**：通常是 **L1Loss** 或 **BCE Loss**（具体取决于 `get_depth_loss` 的实现）。

**当前 weight**：**0.5**（已从 1.5 降低）。

**降低理由**：
- 毫米波雷达点云稀疏，大部分像素没有 depth GT
- depth GT 噪声大，高权重会导致模型过拟合训练集上的噪声
- depth 只是辅助任务，检测精度才是最终目标

**对应 TensorBoard 曲线**：`train/depth`、`val/depth`

---

## 5. Loss 层级图

```
总 Loss
│
├─ loss_detection（检测任务）
│  │
│  ├─ loss_heatmap ──► GaussianFocalLoss
│  │                    └── 目标：BEV 中心点定位准不准
│  │
│  └─ loss_bbox × 0.5 ──► SmoothL1Loss(beta=0.05)
│                         └── 目标：框的 10 维参数回归准不准
│                             └── code_weights 控制各维度优先级
│                                 ├── x=4.0（纵向，最高优先级）
│                                 ├── y=2.0（横向）
│                                 ├── yaw=3.0（朝向）
│                                 └── z/w/l/h/vx/vy = 较低权重
│
└─ loss_depth × 0.5（深度估计任务）
   └── 目标：图像深度估计准不准
       └── 但 radar GT 稀疏，已降低权重防止过拟合
```

---

## 6. 曲线健康度分析

从 TensorBoard 观察各 loss 的收敛情况：

| Loss | Train | Val | 判断 | 说明 |
|---|---|---|---|---|
| **heatmap** | 0.6 → 0.25 | 2.1 → **1.3** | ✅ 收敛 | 分类任务学得较好 |
| **bbox** | 0.7 → 0.3 | 1.7 → **0.8** | ✅ 收敛 | 回归任务稳定下降 |
| **detection** | 1.2 → 0.5 | 3.7 → **2.0** | ✅ 收敛 | 检测主任务健康 |
| **depth** | 3.5 → **1.5** | 7.8 → 7.0 → **7.8** | ❌ 过拟合 | 训练集下降，验证集反弹 |

### 关键结论

1. **检测任务（heatmap + bbox）收敛良好**，val 曲线平稳下降，没有过拟合迹象。
2. **depth 任务严重过拟合**：train depth 持续下降，但 val depth 反弹至 7.8。
3. **降低 depth weight 到 0.5 的目标**：释放梯度给检测任务，让 bbox/heatmap 进一步收敛，同时阻止 depth 分支过拟合。

---

## 7. 修改文件

| 文件 | 修改内容 |
|---|---|
| `exps/det/CRN_r50_256x704_128x128_4key.py:310` | `weight=1.5` → `weight=0.5`（training_step） |
| `exps/det/CRN_r50_256x704_128x128_4key.py:357` | `weight=3.` → `weight=0.5`（validation_step，保持监控一致） |
