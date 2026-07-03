# CRN 评估可视化坐标系统梳理

日期: 2026-06-17

---

## 1. 概述

评估代码 `scripts/data_preprocessing_final/check/model_inference_vis2.py` 的核心任务是：
1. 加载训练好的 CRN 模型，对验证/测试集做推理；
2. 将模型输出的 3D 框与数据集真值进行匹配，计算 TP/FP/FN；
3. 把 3D 框投影到图像平面，生成带色标注的可视化结果。

整个流程涉及 **三个空间** 的转换：
- **Ego 空间**（模型输出 & 真值标注的统一坐标系）
- **Camera 空间**（相机局部坐标系）
- **Image 空间**（像素平面）

---

## 2. Ego 空间 —— 统一本地坐标系

在路侧场景中，Ego 空间指的是**路侧设备（传感器阵列）的本地坐标系**，而非车辆后轴中心。

### 2.1 坐标轴定义

| 轴 | 方向 | 范围 |
|---|---|---|
| **X** | 道路纵向（前方） | `[0, 240] m` |
| **Y** | 道路横向（左右） | `[-25.6, 25.6] m` |
| **Z** | 高度（向上） | 地面为参考 |

### 2.2 与传感器的关系

- **雷达**：数据天然就在 Ego 空间下。雷达 CSV 中的 `x, y` 直接对应 Ego 空间的纵向/横向距离。
- **相机**：通过外参（`calibrated_sensor`）从 Camera 空间转换到 Ego 空间。
- **真值标注**：`ann['translation']` 同样在 Ego 空间下标注。

### 2.3 场景配置示例

```python
# config.py
FRONTAL = {
    'radar_h': 7.0,   # 雷达安装高度 7m
    'cam_h': 6.0,     # 相机安装高度 6m
    'yaw_deg': 0.0,   # 水平朝向对齐
}
```

雷达与相机共享同一水平朝向（yaw），仅高度不同。因此相机外参的旋转矩阵主要是俯仰角（pitch）校正，平移向量主要是高度差。

---

## 3. 预测框 (Pred) —— 模型输出

### 3.1 输出格式

模型 `get_bboxes()` 返回的 `pred_boxes3d` 格式为：
```python
[x, y, z, w, l, h, yaw, vx, vy]
```

| 字段 | 含义 | 备注 |
|---|---|---|
| `x, y` | Ego 空间下的地面中心坐标 | BEV 平面位置 |
| `z` | **底面高度** | CenterPoint 惯例，非几何中心 |
| `w, l, h` | 宽度、长度、高度 | `w` 沿 X 向，`l` 沿 Y 向 |
| `yaw` | 偏航角 | 需校准后用于画框 |

### 3.2 画框时的角点生成

```python
def get_pred_box_corners(box):
    x, y, z, w, l, h, yaw = box[:7]
    x_corners = [-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]
    y_corners = [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2]
    z_corners = [0, 0, 0, 0, h, h, h, h]  # 从底面往上堆叠
    ...
```

注意 `z_corners` 起点为 `0`，确认 `z` 代表**底面高度**。

---

## 4. 真值框 (GT) —— 数据集标注

### 4.1 标注格式

从 `info['ann_infos']` 读取：
```python
gx, gy, gz = ann['translation']      # [x, y, z]
gt_size = ann['size']                # [l, w, h]
gt_rotation = ann['rotation']        # 四元数
```

| 字段 | 含义 | 备注 |
|---|---|---|
| `translation` | Ego 空间下的物体中心坐标 | `z` 为**几何中心高度** |
| `size` | `[length, width, height]` | nuScenes 标准格式 |
| `rotation` | 四元数 | 表示物体朝向 |

### 4.2 画框时的角点生成

```python
def get_gt_box_corners(size, translation, rotation_quat):
    l, w, h = size
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]  # 以中心为基准
    ...
```

注意 `z_corners` 为 `±h/2`，确认 `translation[2]` 代表**几何中心高度**。

---

## 5. 匹配逻辑 —— Ego 空间 2D 距离

匹配在 Ego 空间的 **BEV 平面（x, y）** 上进行，不考虑 z 轴差异。

### 5.1 匹配函数

```python
def match_boxes(gt_list, pred_list, dist_thresh=2.0):
    cost = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i, (gx, gy, gz, gyaw) in enumerate(gt_list):
        for j, (px, py, pz, pyaw, _) in enumerate(pred_list):
            cost[i, j] = np.sqrt((gx - px)**2 + (gy - py)**2)  # 仅 2D 距离
    
    row_ind, col_ind = linear_sum_assignment(cost)
    # 距离 <= dist_thresh 视为 TP
```

### 5.2 匹配判定

- **TP**：Pred 与 GT 的 (x, y) 中心距离 ≤ 2.0m
- **FN**：有 GT 但没有匹配的 Pred
- **FP**：有 Pred 但没有匹配的 GT

使用 `scipy.optimize.linear_sum_assignment` 进行全局最优一对一匹配。

---

## 6. 投影到图像 —— Ego → Camera → Image

### 6.1 坐标变换流程

```
Ego 空间 (3D 角点)
    │
    ▼
ego2cam 外参变换 ──► Camera 空间 (3D 角点)
    │
    ▼
K 内参投影 ──► Image 空间 (2D 像素坐标)
```

### 6.2 关键代码

#### ① 读取相机内外参
```python
K = np.array(cam_info['calibrated_sensor']['camera_intrinsic'])

quat = Quaternion(cam_info['calibrated_sensor']['rotation'])
trans = np.array(cam_info['calibrated_sensor']['translation'])
cam2world = np.eye(4)
cam2world[:3, :3] = quat.rotation_matrix
cam2world[:3, 3] = trans
ego2cam = np.linalg.inv(cam2world)  # Ego → Camera
```

- `cam2world` 实质是 **Camera → Ego** 的外参
- 取逆后得到 `ego2cam`，用于将 Ego 空间点转换到相机坐标系

#### ② Ego → Camera
```python
corners_world = get_pred_box_corners(box)  # Ego 空间 3×8
corners_world_homo = np.vstack([corners_world, np.ones((1, 8))])
corners_cam = ego2cam @ corners_world_homo  # Camera 空间 4×8
```

#### ③ Camera → Image（针孔投影）
```python
corners_2d = (K @ corners_cam[:3, :] / corners_cam[2, :]).T
```

公式：$\mathbf{p}_{2D} = K \cdot \frac{\mathbf{P}_{cam}}{Z_{cam}}$

#### ④ 深度过滤
```python
if np.any(corners_cam[2, :] > 0.1):
    # 至少有一个角点在相机前方，才进行投影和绘制
```

---

## 7. 可视化配色

| 颜色 | 含义 | 来源 |
|---|---|---|
| **红色** | 真值框 (GT) | `get_gt_box_corners` |
| **绿色** | 正确检测 (TP) | Pred 成功匹配到 GT |
| **紫色/洋红** | 误检 (FP) | Pred 未匹配到任何 GT |

---

## 8. 关键结论

1. **Ego 空间是路侧设备的统一本地坐标系**，雷达数据天然在此系下，相机通过外参变换进来。
2. **模型输出和真值标注都在 Ego 空间下**，因此匹配可以直接在 Ego 空间中进行。
3. **匹配仅使用 2D (x, y) 平面距离**，因为 BEV 检测的核心是地面位置对齐。
4. **可视化需要通过 ego2cam 外参和 K 内参两步投影**，将 3D 角点映射到图像平面。
5. **Pred 的 z 是底面高度，GT 的 z 是中心高度**，两者在画框时通过不同的 `z_corners` 定义得到正确处理，不影响可视化效果。
