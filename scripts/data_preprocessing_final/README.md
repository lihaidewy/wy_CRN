# CRN 数据预处理管线 — 使用说明

## 概述

本管线将原始路侧传感器数据（雷达 CSV、相机图片、JSON 标注）转换为 CRN 模型训练所需的全部中间产物。

**核心变化**：重构前需要依次手动运行 6 个脚本，参数分散在 5 个文件中。现在只需一份配置文件 + 一个入口脚本。

```
原始雷达 CSV ──→ [阶段1]  ──→ samples/RADAR_FRONT/*.bin (18通道)
                                  │
                                  ▼
                           [阶段2]  ──→ radar_bev_filter/*.bin  (BEV 7通道)
                           │         ──→ radar_pv_filter/*.bin   (PV  7通道)
                           │
                           ├─→ [阶段3]  ──→ depth_gt/*.bin (深度真值)
                           │
JSON 标注 ──────────────────┤
                           │
                           └─→ [阶段4]  ──→ nuscenes_infos_train.pkl
                                           ──→ nuscenes_infos_val.pkl
```

## 核心概念

### 场景 (Scenario) vs 批次 (Offset)

| 概念 | 含义 | 影响参数 | 变化频率 |
|------|------|---------|---------|
| **场景** | 物理安装位置 | H(高度), yaw(偏角) | **固定**，安装后不变 |
| **批次** | 数据采集批次 | offset(帧号偏移) | **递增**，每次新增数据 |

一个场景可以有多个批次（比如正视场景有 4 批数据），每次新增数据时只需：
1. 给一个唯一的 `--offset`
2. 在 `config.py` 的 `OFFSET_SCENARIO_MAP` 中注册：`offset → 场景名`

### 已注册的场景

| 场景名 | H (m) | yaw | 说明 |
|--------|-------|-----|------|
| `frontal` | 6.0 | 0° | 正视路侧，相机正对道路 |
| `oblique` | 18.0 | 15° | 斜视路侧，向左偏 15° |

## 快速开始

### 0. 前置条件

```bash
pip install numpy pandas opencv-python mmcv
```

### 1. 查看当前配置

```bash
cd CRN
python scripts/data_preprocessing_final/pipeline.py --show-config
```

### 2. 处理正视场景数据

```bash
# 逐批处理雷达 CSV
python pipeline.py --scenario frontal --offset 0      --csv data/radar_data/S1s_flag.csv --stage radar2bin
python pipeline.py --scenario frontal --offset 100000 --csv data/radar_data/batch2.csv    --stage radar2bin
```

### 3. 处理斜视场景数据

```bash
# 斜视场景（H=18m, yaw=15°，雷达坐标会自动旋转）
python pipeline.py --scenario oblique --offset 400000 --csv data/radar_data/oblique.csv --stage radar2bin
```

### 4. 生成衍生特征 + 深度真值

```bash
python pipeline.py --scenario frontal --stage derivatives
python pipeline.py --scenario frontal --stage depth_gt
python pipeline.py --scenario oblique --stage derivatives
python pipeline.py --scenario oblique --stage depth_gt
```

### 5. 构建统一 PKL（自动合并所有场景）

```bash
python pipeline.py --stage build_infos
```

该阶段自动：
- 根据帧号查找对应的场景
- 从 JSON 的 `extrinsics_matrix_4x4` 自动提取相机外参（如有）
- 生成包含所有场景帧的 train/val PKL

---

## 阶段详解

### 阶段1: `radar2bin` — 雷达 CSV → 18ch 二进制

**输入**：雷达原始 CSV 文件（含 Angle, Range, Speed, SNR, ID 列）
**输出**：`data/my_formatted_data/samples/RADAR_FRONT/radar_XXXXXX.pcd.bin`

**物理计算**：
- 根据龙门架安装高度 `H`，将斜距解算为地面投影坐标 (X前, Y左, Z=0)
- 速度分量同步分解到世界坐标系
- 滤除盲区噪点（斜距投影 < 0.1m² 的点）

**单独运行**：
```bash
# 正视场景
python pipeline.py --scenario frontal --offset 0 --csv data/radar_data/S1s_flag.csv --stage radar2bin

# 斜视场景（自动应用 yaw=15° 旋转）
python pipeline.py --scenario oblique --offset 400000 --csv data/radar_data/oblique.csv --stage radar2bin
```

**CSV 格式要求**：至少包含 `Frame`, `Angle`, `Range`, `Speed`, `SNR`, `ID` 六列。

---

### 阶段1.5: `merge_new_data` — 合并新批次数据

**用途**：当你有一批新采集的数据（图片 + JSON + 雷达 CSV），需要将它们合并到主数据目录。

**输入**：
- `data/temp_data/images/*.png` — 新图片
- `data/temp_data/jsons/*.json` — 新 JSON 标注
- `data/radar_data/S1s_flag_test.csv` — 新雷达 CSV

**输出**：上述文件按 `原帧号 + OFFSET` 重命名后复制到主数据目录

**单独运行**：
```bash
python pipeline.py --stage merge_new_data --offset 300000
```

> 该阶段会自动调用阶段1处理新雷达 CSV。注意：`--all` **不包含**本阶段（防止误操作覆盖已有数据）。

---

### 阶段2: `derivatives` — 生成双流特征

**输入**：阶段1 产出的 `samples/RADAR_FRONT/*.bin`（18 通道）

**输出**：
| 产物 | 路径 | 通道数 | 说明 |
|------|------|--------|------|
| BEV 特征 | `radar_bev_filter/radar_XXXXXX.pcd.bin` | 7 | 鸟瞰视角 (X,Y,Z,RCS,vx,vy,sweep_idx) |
| PV 特征 | `radar_pv_filter/XXXXXX.png.bin` | 7 | 透视视角 (u,v,depth,RCS,vx,vy,sweep_idx) |

**PV 投影逻辑**：世界坐标 (X前, Y左, Z上) → 相机坐标 (X右, Y下, Z前) → 针孔投影 → 像素坐标 (u, v)

**单独运行**：
```bash
python pipeline.py --scenario frontal --stage derivatives
python pipeline.py --scenario oblique --stage derivatives
```

---

### 阶段3: `depth_gt` — 生成深度真值

**输入**：阶段2 产出的 `radar_bev_filter/*.bin`（BEV 7通道）

**输出**：`depth_gt/XXXXXX.png.bin` — 每行 `[u, v, depth]`，只保留 FOV 内的点

**说明**：将雷达 BEV 点云投影到相机平面，生成稀疏深度真值，用于 CRN 训练中的深度监督 Loss。

**调试验证图**：若 `PipelineConfig.DEBUG_VIS_FRAME = True`（默认），会在 `work_dirs/debug_radar_proj/` 下生成一张投影可视化图。

**单独运行**：
```bash
python pipeline.py --scenario frontal --stage depth_gt
python pipeline.py --scenario oblique --stage depth_gt
```

---

### 阶段4: `build_infos` — 构建 nuScenes 索引

**输入**：`jsons/*.json`（每帧的标注文件）

**输出**：
- `data/my_formatted_data/nuscenes_infos_train.pkl`
- `data/my_formatted_data/nuscenes_infos_val.pkl`

**训练/验证集划分模式**（在 `config.py` 的 `PipelineConfig.TRAIN_VAL_SPLIT` 中控制）：

| 模式 | 行为 |
|------|------|
| `"all_train"` (默认) | 训练集=验证集=全部数据 |
| `"sequential"` | 前 80% 训练，后 20% 验证（比例见 SPLIT_RATIO） |
| `"range_based"` | 按帧号范围过滤验证集（见 VAL_FRAME_RANGE） |

**单独运行**：
```bash
python pipeline.py --stage build_infos
```

---

## 配置文件说明

所有参数集中在 `config.py`，按用途分类：

### `ScenarioConfig` — 场景物理参数 🔑
```python
ScenarioConfig.FRONTAL = {'H': 6.0,  'yaw_deg': 0.0 }
ScenarioConfig.OBLIQUE = {'H': 18.0, 'yaw_deg': 15.0}
```
新增场景只需在此添加一个字典。

### `OFFSET_SCENARIO_MAP` — 批次→场景映射 🔑
```python
OFFSET_SCENARIO_MAP = {
    0:       'frontal',     # 正视第1批
    100000:  'frontal',     # 正视第2批
    200000:  'frontal',     # 正视第3批
    300000:  'frontal',     # 正视第4批
    400000:  'oblique',     # 斜视第1批
}
```
每次新增数据在此注册一行 `offset → 场景名`。

### `SensorConfig` — 传感器固定参数
```python
CAMERA_INTRINSIC = np.array([...])   # 相机内参 3x3
IMAGE_WIDTH = 3840                   # 图像宽度
IMAGE_HEIGHT = 2160                  # 图像高度
```

### `PathConfig` — 路径
```python
DATA_ROOT = "./data/my_formatted_data"
# ... 各子目录定义
```

### `PipelineConfig` — 管线行为
```python
TRAIN_VAL_SPLIT = "all_train"       # 训练/验证集划分
DEBUG_VIS_FRAME = True              # 调试投影图
```

### `CATEGORY_MAP` — 类别映射
```python
CATEGORY_MAP = {
    'car': 'vehicle.car',
    'truck': 'vehicle.truck',
    # ...
}
```

---

## 常见操作场景

### 场景 A：全新环境，首次处理数据

```bash
# 1. 确认配置
python pipeline.py --show-config

# 2. 处理雷达（逐批）
python pipeline.py --scenario frontal --offset 0 --csv data/radar_data/S1s_flag.csv --stage radar2bin

# 3. 衍生特征
python pipeline.py --scenario frontal --stage derivatives
python pipeline.py --scenario frontal --stage depth_gt

# 4. 构建索引
python pipeline.py --stage build_infos
```

### 场景 B：新增一个数据批次（如正视第5批，offset=500000）

```bash
# 1. 在 config.py 中注册：OFFSET_SCENARIO_MAP[500000] = 'frontal'

# 2. 处理新雷达（会自动转发 yaw=0°，H=6m）
python pipeline.py --scenario frontal --offset 500000 --csv data/radar_data/batch5.csv --stage radar2bin

# 3. 重建衍生特征和索引
python pipeline.py --scenario frontal --stage derivatives
python pipeline.py --stage depth_gt
python pipeline.py --stage build_infos
```

### 场景 C：新增斜视场景数据（如斜视第2批，offset=500000）

```bash
# 1. 在 config.py 中注册：OFFSET_SCENARIO_MAP[500000] = 'oblique'

# 2. 处理斜视雷达（自动应用 yaw=15° 旋转，H=18m）
python pipeline.py --scenario oblique --offset 500000 --csv data/radar_data/oblique_batch2.csv --stage radar2bin

# 3. 重建
python pipeline.py --scenario oblique --stage derivatives
python pipeline.py --stage depth_gt
python pipeline.py --stage build_infos
```

### 场景 D：只改了标注 JSON，不想重跑全管线

```bash
python pipeline.py --stage build_infos
```

### 场景 E：换了相机/龙门架，物理参数变了

```bash
# 1. 修改 config.py 中对应场景的 ScenarioConfig（H, yaw_deg）

# 2. 重新跑投影相关阶段
python pipeline.py --scenario oblique --stage derivatives
python pipeline.py --scenario oblique --stage depth_gt
python pipeline.py --stage build_infos
```

---

## 最终数据目录结构

运行全管线后，`data/my_formatted_data/` 下的结构：

```
data/my_formatted_data/
├── samples/
│   ├── RADAR_FRONT/              # 阶段1 产出：18ch 雷达点云
│   │   └── radar_XXXXXX.pcd.bin
│   └── CAM_FRONT/                # 手动放置或阶段1.5 迁移：图片
│       └── XXXXXX.png
├── radar_bev_filter/             # 阶段2 产出：BEV 7ch 雷达
│   └── radar_XXXXXX.pcd.bin
├── radar_pv_filter/              # 阶段2 产出：PV 7ch 雷达
│   └── XXXXXX.png.bin
├── depth_gt/                     # 阶段3 产出：深度真值
│   └── XXXXXX.png.bin
├── jsons/                        # 手动放置或阶段1.5 迁移：标注
│   └── XXXXXX.json
├── nuscenes_infos_train.pkl      # 阶段4 产出：训练索引
└── nuscenes_infos_val.pkl        # 阶段4 产出：验证索引
```

此结构符合 CRN 训练所需的 nuScenes 格式，可直接用于 `exps/det/` 下的训练脚本。

---

## 技巧：命令行覆盖 vs 配置文件

| 场景 | 推荐方式 |
|------|---------|
| 永久修改参数 | 改 `config.py` |
| 临时测试不同参数 | 用 `--csv` / `--offset` 命令行参数 |
| 确认当前值 | `python pipeline.py --show-config` |
| 不确定该不该改 | 先 `--show-config`，搞清楚再改 |

命令行参数优先级 > `config.py` 默认值，未指定时使用 `config.py` 中的值。

---

## 对比：重构前后

| | 重构前 | 重构后 |
|------|--------|--------|
| 脚本数量 | 6 个独立脚本 | 1 个入口 + 1 个配置 |
| 参数位置 | 分散在 5+ 个文件 | 集中在 `config.py` |
| 修改 H 高度 | 需改 4 个文件 | 改 1 行 |
| 修改内参 | 需改 2 个文件 | 改 1 处 |
| 运行方式 | 手动依次执行 4-5 个命令 | `--all` 一步到位 |
| 选跑某阶段 | 需要知道对应哪个脚本 | `--stage <名称>` 即可 |
| 新批次处理 | 手动算 offset、复制文件 | `--stage merge_new_data --offset N` |
| 逻辑重复 | `convert_excel_frame_to_bin` 写了两遍 | 唯一实现 |
