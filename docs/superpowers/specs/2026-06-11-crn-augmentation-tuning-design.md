# CRN 数据增强 + 训练参数优化 — 方案 2 设计文档

**日期**: 2026-06-11
**目标**: 在不降低 BEV X 分辨率（0.5m/440格）的前提下，通过增强数据多样性和训练参数，修复 frontal 场景纵向误差恶化问题
**策略**: 扩大 IDA/BDA 增强范围 + 提升 depth loss 权重 + 平滑 heatmap 高斯核

---

## 1. 背景与问题

0.5m BEV X 分辨率让 oblique 场景显著改善（纵向 0.348m ✅），但 frontal 反退化：

| 指标 | Frontal | Oblique |
|------|---------|---------|
| 纵向误差 | 0.502 m ❌ | 0.348 m ✅ |
| 误检率 | 16.6% ❌ | 4.5% ✅ |
| MOTA | 69.4% | 78.0% |

**根因**：0.5m 高分辨率下 frontal 密集场景产生更多 heatmap 峰值分裂，而当前数据增强几乎为零（resize 仅 ±1% 变化），模型对密集场景的尺度泛化不足。

## 2. 改动清单

### 2.1 IDA 图像增强（扩大尺度多样性）

文件: `exps/det/CRN_r50_256x704_128x128_4key.py`

```python
# ida_aug_conf
# 修改前:
'resize_lim': (0.36, 0.38),    # 仅 ±1% 缩放 → 几乎无增强
'rot_lim': (0., 0.),            # 无旋转

# 修改后:
'resize_lim': (0.30, 0.45),     # ±7.5% 缩放 → 增加尺度多样性
'rot_lim': (-3., 3.),           # ±3° 轻微旋转
```

### 2.2 BDA BEV 增强（适度恢复旋转）

```python
# bda_aug_conf
# 修改前:
'rot_lim': (-10.0, 10.0),

# 修改后:
'rot_lim': (-15.0, 15.0),
```

### 2.3 Depth Loss 权重提升

```python
# training_step 中
# 修改前:
loss_depth = self.get_depth_loss(depth_labels.cuda(), depth_preds, weight=5.)

# 修改后:
loss_depth = self.get_depth_loss(depth_labels.cuda(), depth_preds, weight=8.)
```

### 2.4 Heatmap 高斯核平滑

```python
# head_conf['train_cfg']
# 修改前:
gaussian_overlap=0.1,
min_radius=2,

# 修改后:
gaussian_overlap=0.2,   # 高斯重叠区增大 → 目标更平滑
min_radius=3,           # 最小高斯半径从 1m → 1.5m
```

## 3. 显存影响

所有改动不改变 feature map 尺寸（BEV 仍是 128×440），显存占用不变。

## 4. 预期效果

| 指标 | 当前 | 预期 |
|------|------|------|
| Frontal 纵向误差 | 0.502 m | ~0.38 m |
| Frontal 误检率 | 16.6% | ~10% |
| Oblique 纵向误差 | 0.348 m | 保持 ≤0.35 m |
| 全局 2D ATE | 0.483 m | ~0.35 m |

## 5. 验证计划

1. 修改配置后从头训练 24 epoch
2. 使用 model_inference_vis2.py 评估
3. 对比 frontal/oblique 纵向误差
