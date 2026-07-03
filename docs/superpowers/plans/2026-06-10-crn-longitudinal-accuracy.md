# CRN 纵向定位精度优化 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过 Loss 权重 + 数据增强微调，将 CRN 模型纵向（X 轴）定位误差从 0.439m 降至 ≤ 0.35m。

**Architecture:** 修改单一配置文件中的 3 个超参数（x 权重、SmoothL1 beta、BEV 旋转范围），然后重新训练并评估对比。

**Tech Stack:** Python, PyTorch Lightning, CRN (Camera-Radar-Net)

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `exps/det/CRN_r50_256x704_128x128_4key.py` | 修改 | 3 处参数调整 |

---

### Task 1: 提升 x 方向的 loss 权重

**Files:**
- Modify: `exps/det/CRN_r50_256x704_128x128_4key.py:225`

- [ ] **Step 1: 修改 code_weights**

  找到 `head_conf['train_cfg']` 中的 `code_weights`，将索引 0（x 权重）从 `4.0` 改为 `6.0`：

  ```python
  # 修改前:
  code_weights=[4.0, 1.0, 0.5, 1.5, 1.5, 0.5, 2.0, 2.0, 0.2, 0.2],
  # 修改后:
  code_weights=[6.0, 1.0, 0.5, 1.5, 1.5, 0.5, 2.0, 2.0, 0.2, 0.2],
  ```

- [ ] **Step 2: 验证修改**

  Run:
  ```bash
  grep -n "code_weights" exps/det/CRN_r50_256x704_128x128_4key.py
  ```
  Expected: 显示包含 `code_weights=[6.0, 1.0, 0.5,` 的行

- [ ] **Step 3: Commit**

  ```bash
  git add exps/det/CRN_r50_256x704_128x128_4key.py
  git commit -m "feat: increase x loss weight from 4.0 to 6.0 for better longitudinal accuracy"
  ```

---

### Task 2: 调小 SmoothL1 Loss 的 beta

**Files:**
- Modify: `exps/det/CRN_r50_256x704_128x128_4key.py:247`

- [ ] **Step 1: 修改 loss_bbox beta**

  找到 `head_conf['loss_bbox']`，将 `beta` 从 `0.11` 改为 `0.05`：

  ```python
  # 修改前:
  'loss_bbox': dict(type='SmoothL1Loss',beta=0.11, reduction='mean', loss_weight=0.5),
  # 修改后:
  'loss_bbox': dict(type='SmoothL1Loss',beta=0.05, reduction='mean', loss_weight=0.5),
  ```

- [ ] **Step 2: 验证修改**

  Run:
  ```bash
  grep -n "loss_bbox" exps/det/CRN_r50_256x704_128x128_4key.py
  ```
  Expected: 显示包含 `beta=0.05` 的行

- [ ] **Step 3: Commit**

  ```bash
  git add exps/det/CRN_r50_256x704_128x128_4key.py
  git commit -m "feat: reduce SmoothL1 beta from 0.11 to 0.05 for sharper center regression"
  ```

---

### Task 3: 缩小 BEV 旋转增强范围

**Files:**
- Modify: `exps/det/CRN_r50_256x704_128x128_4key.py:52`

- [ ] **Step 1: 修改 rot_lim**

  找到 `bda_aug_conf` 中的 `rot_lim`，从 `(-22.5, 22.5)` 改为 `(-10.0, 10.0)`：

  ```python
  # 修改前:
  'rot_lim': (-22.5, 22.5),
  # 修改后:
  'rot_lim': (-10.0, 10.0),
  ```

- [ ] **Step 2: 验证修改**

  Run:
  ```bash
  grep -n "rot_lim" exps/det/CRN_r50_256x704_128x128_4key.py
  ```
  Expected: 显示包含 `'rot_lim': (-10.0, 10.0)` 的行

- [ ] **Step 3: Commit**

  ```bash
  git add exps/det/CRN_r50_256x704_128x128_4key.py
  git commit -m "feat: tighten BEV rotation aug from ±22.5° to ±10°"
  ```

---

### Task 4: 启动训练

**Files:**
- 无文件修改

- [ ] **Step 1: 确认数据已就绪**

  Run:
  ```bash
  ls data/my_formatted_data/nuscenes_infos_train.pkl data/my_formatted_data/nuscenes_infos_val.pkl
  ```
  Expected: 两个文件均存在

- [ ] **Step 2: 启动训练**

  Run:
  ```bash
  cd /home/wy666/CRN && python exps/det/CRN_r50_256x704_128x128_4key.py
  ```
  Expected: 训练正常启动，显示 epoch 0/24 开始

  Note: 训练预计需要数小时（取决于 GPU），请等待完成。

---

### Task 5: 评估验证

**Files:**
- Modify: `scripts/data_preprocessing_final/check/model_inference_vis2.py:18`（临时修改 CKPT_PATH）

- [ ] **Step 1: 更新评估脚本中的 checkpoint 路径**

  找到最新训练的 checkpoint：
  ```bash
  ls -t outputs/det/CRN_r50_256x704_128x128_4key/lightning_logs/version_*/checkpoints/epoch=23-*.ckpt | head -1
  ```

  将返回的路径填入 `model_inference_vis2.py` 的 `CKPT_PATH`：
  ```python
  CKPT_PATH = "outputs/det/CRN_r50_256x704_128x128_4key/lightning_logs/version_XXX/checkpoints/epoch=23-step_XXXX.ckpt"
  ```

- [ ] **Step 2: 运行评估**

  ```bash
  cd /home/wy666/CRN && python scripts/data_preprocessing_final/check/model_inference_vis2.py
  ```
  Expected: 输出全局 + 分场景统计，包含纵向误差

- [ ] **Step 3: 对比基线**

  | 指标 | 基线 (version_103) | 新模型 | 变化 |
  |------|-------------------|--------|------|
  | 纵向误差 (X) | 0.439 m | ? | 目标 ≤ 0.35m |
  | 2D ATE | 0.483 m | ? | 观察 |
  | 漏检率 | 16.2% | ? | 不显著恶化 |

- [ ] **Step 4: 记录结果**

  将评估结果截图或记录，用于决定下一步（是否进入方案 A：提升 BEV X 分辨率）。

---

## Self-Review Checklist

- [x] **Spec coverage**: 设计文档中 3 个改动点均已覆盖（Task 1/2/3）
- [x] **Placeholder scan**: 无 TBD/TODO/"implement later"
- [x] **Type consistency**: 参数名称与代码库一致
- [x] **Training coverage**: Task 4 覆盖训练步骤
- [x] **Validation coverage**: Task 5 覆盖评估与对比
