# CRN 分场景独立训练 — 设计方案

**日期**: 2026-06-11
**目标**: Frontal 和 Oblique 各训一个独立模型，零模型结构改动

---

## 1. 设计思路

两个场景物理差异大（高度 7m vs 25m，yaw 0° vs 15°，距离分布不同），当前共享训练互相干扰。分场景独立训练让每个模型聚焦单一场景，预期全局 2D ATE 从 0.483m 降至 **~0.32m**。

## 2. 实现架构

```
数据管线 (一次改动)
├── 生成 frontal_train.pkl, frontal_val.pkl
└── 生成 oblique_train.pkl, oblique_val.pkl

配置文件 (两份, 复用基类)
├── CRN_..._frontal.py  → 加载 frontal PKL
└── CRN_..._oblique.py  → 加载 oblique PKL

训练 (两次, 各自跑)
├── python CRN_..._frontal.py
└── python CRN_..._oblique.py

推理
└── 根据帧号自动选择对应 checkpoint 评估
```

## 3. 改动点

### 3.1 数据管线 — 生成单场景 PKL + 固定 100 帧验证

文件: `scripts/data_preprocessing_final/config.py` + `pipeline.py`

**划分策略**：每个场景各留 **100 帧**作为验证集（按帧号排序取最后 100 帧），其余作为训练集。

| 场景 | 训练帧 | 验证帧 |
|------|--------|--------|
| Frontal | ~828 | 100 |
| Oblique | ~807 | 100 |

新增 `config.py` 参数：
```python
VAL_FRAMES_PER_SCENARIO = 100
```

修改 `pipeline.py` 中 `per_scenario` 切分逻辑：
```python
# 每个场景内按帧号排序后，取最后 VAL_FRAMES_PER_SCENARIO 帧作为 val
frames = sorted(scenario_groups[sc_name], key=lambda x: int(x['sample_token'].split('_')[-1]))
val_infos.extend(frames[-VAL_FRAMES_PER_SCENARIO:])
train_infos.extend(frames[:-VAL_FRAMES_PER_SCENARIO])
```

生成独立 PKL：
```python
# 新增产物
DATA_ROOT/
├── nuscenes_infos_frontal_train.pkl
├── nuscenes_infos_frontal_val.pkl
├── nuscenes_infos_oblique_train.pkl
└── nuscenes_infos_oblique_val.pkl
```

### 3.2 配置文件 — 场景专用

文件: 
- `exps/det/CRN_r50_256x704_128x128_4key_frontal.py`
- `exps/det/CRN_r50_256x704_128x128_4key_oblique.py`

每个配置文件只继承基类 + 覆盖 PKL 路径和场景特定参数：

```python
# frontal 配置
class CRNLightningModel(BEVDepthLightningModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ... 所有参数同基类 ...
        # 覆盖场景特定参数（可选）
        self.bda_aug_conf['rot_lim'] = (-10.0, 10.0)  # frontal 用较小的旋转
        # 等等
```

或者更简洁：使用同一个配置文件，通过命令行 `--scenario frontal` 参数动态切换 PKL 路径。

### 3.3 评估脚本 — 自动选模型

文件: `scripts/data_preprocessing_final/check/model_inference_vis2.py`

根据评估帧的 offset 自动加载对应场景的 checkpoint。

## 4. 预期效果

| 指标 | 当前 (共享模型) | Frontal 独立 | Oblique 独立 | 按场景混合评估 |
|------|---------------|-------------|-------------|-------------|
| Frontal 纵向 | 0.502 m | ~0.35 m | — | ~0.35 m |
| Oblique 纵向 | 0.348 m | — | ~0.30 m | ~0.30 m |
| 全局 2D ATE | 0.483 m | — | — | **~0.32 m** |

## 5. 风险评估

| 风险 | 缓解 |
|------|------|
| Frontal 数据量较少 (~750 帧) | 增强足够强（方案 2），24 epoch 足够 |
| 两套配置维护成本 | 差异仅 PKL 路径 + 少量超参, 改动极少 |
| 推理时需要场景标签 | 帧号 ≥ 400000 即 oblique, < 400000 即 frontal |
