# CRN 纵向定位精度优化 — 方案 C 设计文档

**日期**: 2026-06-10
**目标**: 将纵向（X 轴）定位误差从 ~0.44m 降低到 ~0.3m
**策略**: Loss 权重 + 数据增强微调（零成本、快速验证）

---

## 1. 当前基线

| 指标 | 全局 | Frontal | Oblique |
|------|------|---------|---------|
| 纵向误差 (X) | 0.439 m | 0.475 m | 0.391 m |
| 横向误差 (Y) | 0.133 m | 0.135 m | 0.131 m |
| 2D ATE | 0.483 m | 0.517 m | 0.436 m |
| 漏检率 | 16.2% | 15.2% | 17.5% |

**核心观察**: 纵向误差是横向误差的 3 倍以上，且 frontal 场景比 oblique 更严重。

---

## 2. 改动点

### 2.1 Loss 权重调整

文件: `exps/det/CRN_r50_256x704_128x128_4key.py`

```python
# head_conf['train_cfg']['code_weights']
# 索引: 0=x, 1=y, 2=z, 3=w, 4=l, 5=h, 6=sin_rot, 7=cos_rot, 8=vx, 9=vy
# 修改前: [4.0, 1.0, 0.5, 1.5, 1.5, 0.5, 2.0, 2.0, 0.2, 0.2]
# 修改后: [6.0, 1.0, 0.5, 1.5, 1.5, 0.5, 2.0, 2.0, 0.2, 0.2]
```

**理由**: 提升 x（纵向）的梯度权重，让网络对纵向偏移更敏感。

### 2.2 SmoothL1 Beta 调小

文件: `exps/det/CRN_r50_256x704_128x128_4key.py`

```python
# head_conf['loss_bbox']
# 修改前: dict(type='SmoothL1Loss', beta=0.11, reduction='mean', loss_weight=0.5)
# 修改后: dict(type='SmoothL1Loss', beta=0.05, reduction='mean', loss_weight=0.5)
```

**理由**: beta 越小，大误差受到的惩罚越接近 L1Loss（更严厉），迫使中心点回归更精准。

### 2.3 BEV 旋转增强范围缩小

文件: `exps/det/CRN_r50_256x704_128x128_4key.py`

```python
# bda_aug_conf
# 修改前: 'rot_lim': (-22.5, 22.5)
# 修改后: 'rot_lim': (-10.0, 10.0)
```

**理由**: ±22.5° 的旋转过大会"拉伸"纵向分布，限制在 ±10° 更符合实际道路场景。

---

## 3. 验证计划

1. 修改配置后**从头训练** 24 epoch
2. 使用 `model_inference_vis2.py` 评估 train 集
3. 对比纵向误差变化

**成功标准**:
- 全局纵向误差从 0.439m 降至 ≤ 0.35m（第一步目标）
- 不显著增加漏检率或误检率

---

## 4. 后续阶段（预留）

若方案 C 效果不足，进入方案 A（提升 BEV X 分辨率）:
- 将 `x_bound` 步长从 1.25m 缩小至 0.625m
- 同步调整 voxel_size、output_shape、grid_size
