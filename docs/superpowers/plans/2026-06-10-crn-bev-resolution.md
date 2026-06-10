# CRN BEV X 分辨率提升 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 BEV X 方向分辨率从 1.25m/voxel 提升到 0.5m/voxel，纵向定位误差从 0.415m 降至 ≤0.35m。

**Architecture:** 修改单一配置文件中的 5 个关联参数（x_bound、grid_size、voxel_size、bev_shape），保持 Y 方向和感知范围不变，然后重新训练并评估。

**Tech Stack:** Python, PyTorch Lightning, CRN (Camera-Radar-Net)

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `exps/det/CRN_r50_256x704_128x128_4key.py` | 修改 | 5 处 BEV X 分辨率关联参数 |

---

### Task 1: 修改 x_bound 步长

**Files:**
- Modify: `exps/det/CRN_r50_256x704_128x128_4key.py:64`

- [ ] **Step 1: 修改 backbone_img_conf 中的 x_bound**

  找到 `backbone_img_conf` 中的 `x_bound`，将步长从 `1.25` 改为 `0.5`：

  ```python
  # 修改前:
  'x_bound': [0.0, 220.0, 1.25],
  # 修改后:
  'x_bound': [0.0, 220.0, 0.5],
  ```

- [ ] **Step 2: 验证修改**

  Run:
  ```bash
  grep -n "x_bound" exps/det/CRN_r50_256x704_128x128_4key.py
  ```
  Expected: 显示包含 `'x_bound': [0.0, 220.0, 0.5]` 的行

- [ ] **Step 3: Commit**

  ```bash
  git add exps/det/CRN_r50_256x704_128x128_4key.py
  git commit -m "feat: reduce BEV X voxel size from 1.25m to 0.5m"
  ```

---

### Task 2: 修改 fuser bev_shape

**Files:**
- Modify: `exps/det/CRN_r50_256x704_128x128_4key.py:158`

- [ ] **Step 1: 修改 fuser_conf 中的 bev_shape**

  找到 `fuser_conf` 中的 `bev_shape`，将 X 维度从 `176` 改为 `440`：

  ```python
  # 修改前:
  'bev_shape': (128, 176),
  # 修改后:
  'bev_shape': (128, 440),
  ```

  注意：bev_shape 格式为 (y, x)，y=128 保持不变，x=176→440。

- [ ] **Step 2: 验证修改**

  Run:
  ```bash
  grep -n "bev_shape" exps/det/CRN_r50_256x704_128x128_4key.py
  ```
  Expected: 显示包含 `'bev_shape': (128, 440)` 的行

- [ ] **Step 3: Commit**

  ```bash
  git add exps/det/CRN_r50_256x704_128x128_4key.py
  git commit -m "feat: update fuser bev_shape to match doubled X resolution"
  ```

---

### Task 3: 修改 head bbox_coder voxel_size

**Files:**
- Modify: `exps/det/CRN_r50_256x704_128x128_4key.py:201`

- [ ] **Step 1: 修改 bbox_coder 中的 voxel_size**

  找到 `head_conf['bbox_coder']` 中的 `voxel_size`，将 X 步长从 `1.25` 改为 `0.5`：

  ```python
  # 修改前:
  voxel_size=[1.25, 0.4, 8],
  # 修改后:
  voxel_size=[0.5, 0.4, 8],
  ```

- [ ] **Step 2: 验证修改**

  Run:
  ```bash
  grep -n "voxel_size" exps/det/CRN_r50_256x704_128x128_4key.py
  ```
  Expected: 显示包含 `voxel_size=[0.5, 0.4, 8]` 的行（bbox_coder 段落）

- [ ] **Step 3: Commit**

  ```bash
  git add exps/det/CRN_r50_256x704_128x128_4key.py
  git commit -m "feat: update head bbox_coder voxel_size for 0.5m X resolution"
  ```

---

### Task 4: 修改 head train_cfg grid_size 和 voxel_size

**Files:**
- Modify: `exps/det/CRN_r50_256x704_128x128_4key.py:213-216`

- [ ] **Step 1: 修改 train_cfg 中的 grid_size**

  找到 `head_conf['train_cfg']` 中的 `grid_size`，将 X 维度从 `176` 改为 `440`：

  ```python
  # 修改前:
  grid_size=[176, 128, 1],
  # 修改后:
  grid_size=[440, 128, 1],
  ```

- [ ] **Step 2: 修改 train_cfg 中的 voxel_size**

  同一段落中，将 `voxel_size` 的 X 步长从 `1.25` 改为 `0.5`：

  ```python
  # 修改前:
  voxel_size=[1.25, 0.4, 8],
  # 修改后:
  voxel_size=[0.5, 0.4, 8],
  ```

- [ ] **Step 3: 验证修改**

  Run:
  ```bash
  grep -n "grid_size" exps/det/CRN_r50_256x704_128x128_4key.py
  ```
  Expected: 显示包含 `grid_size=[440, 128, 1]` 的行

  Run:
  ```bash
  grep -n "voxel_size" exps/det/CRN_r50_256x704_128x128_4key.py
  ```
  Expected: 两处 `voxel_size=[0.5, 0.4, 8]`（bbox_coder 和 train_cfg 段落各一）

- [ ] **Step 4: Commit**

  ```bash
  git add exps/det/CRN_r50_256x704_128x128_4key.py
  git commit -m "feat: update head train_cfg grid_size and voxel_size"
  ```

---

### Task 5: OOM 回退预案（条件执行）

**Files:**
- Modify: `exps/det/CRN_r50_256x704_128x128_4key.py`

- [ ] **Step 1: 若训练启动时 OOM，执行回退**

  如果运行训练命令后出现 CUDA out of memory 错误，将以下参数从 0.5m 回退到 0.8m：

  | 参数 | 0.5m 值 | 0.8m 回退值 |
  |------|-----------|-------------|
  | `x_bound` | [0.0, 220.0, 0.5] | [0.0, 220.0, 0.8] |
  | `bev_shape` | (128, 440) | (128, 275) |
  | `grid_size` | [440, 128, 1] | [275, 128, 1] |
  | `voxel_size` (两处) | [0.5, 0.4, 8] | [0.8, 0.4, 8] |

  然后重新启动训练。

---

### Task 6: 启动训练

**Files:**
- 无文件修改

- [ ] **Step 1: 启动训练并监控显存**

  Run:
  ```bash
  cd /home/wy666/CRN && python exps/det/CRN_r50_256x704_128x128_4key.py -b 1 --amp_backend native
  ```

  观察前几个 iteration 的显存占用。如果正常开始训练且未 OOM，则继续等待完成。

  Note: 训练预计需要比上次多 50~80% 的时间（因为 head 计算量增加），请耐心等待。

---

### Task 7: 评估验证

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

- [ ] **Step 3: 对比基线**

  | 指标 | 方案 C 基线 | 方案 A 新模型 | 变化 |
  |------|------------|--------------|------|
  | 纵向误差 (X) | 0.415 m | ? | 目标 ≤ 0.35m |
  | 2D ATE | 0.452 m | ? | 观察 |
  | 漏检率 | 13.1% | ? | 不显著恶化 |

- [ ] **Step 4: 记录结果**

  将评估结果截图或记录，用于决定是否需要进一步推进（如尝试 0.5m 分辨率）。

---

## Self-Review Checklist

- [x] **Spec coverage**: 设计文档中 5 个改动点均已覆盖（Task 1/2/3/4）
- [x] **Placeholder scan**: 无 TBD/TODO/"implement later"
- [x] **Type consistency**: 参数名称与代码库一致
- [x] **Training coverage**: Task 6 覆盖训练步骤 + OOM 回退（Task 5）
- [x] **Validation coverage**: Task 7 覆盖评估与对比
