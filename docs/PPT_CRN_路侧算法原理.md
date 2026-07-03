# CRN 路侧 3D 目标检测 — 算法原理详解（PPT 素材）

> **全部内容均基于本项目实际代码实现，不涉及论文原版 (nuScenes) 的配置。**
> 代码版本：commit a3118bd，主配置文件 `exps/det/CRN_r50_256x704_128x128_4key.py`

---

## 第1部分：系统总览

---

### Slide 1.1：场景定义

```
         路侧感知系统
   ┌─────────────────────────┐
   │  交通杆 / 龙门架          │
   │                         │
   │   ┌─────┐  ┌─────┐      │
   │   │ 📷  │  │ 📡  │      │
   │   │相机 │  │雷达 │      │
   │   └──┬──┘  └──┬──┘      │
   │      │        │         │
   └──────┼────────┼─────────┘
          │        │
          ▼        ▼
   ═══════════════════════ 道路 → X (纵向, 0~240m)
          │
          ├─ Y (横向, ±25.6m)
          │
          ▼
       检测车辆

两场景:
  Frontal (正视):  雷达高 7m,  相机高 6m,  yaw=0°
  Oblique (斜视):  雷达高 25m, 相机高 18m, yaw=15°
```

---

### Slide 1.2：系统技术指标

| 项目 | 数值 |
|------|------|
| 输入图像 | 3840×2160 单帧 → Resize 到 384×1408 |
| 输入雷达 | 单帧毫米波雷达点云 (CSV → 18ch bin → PV 5ch) |
| 检测范围 (X) | 0 ~ 240m (纵向，沿道路方向) |
| 检测范围 (Y) | -25.6 ~ +25.6m (横向) |
| BEV 分辨率 | 0.5m (X) × 0.4m (Y) → 480 × 128 网格 |
| 深度估计范围 | 2 ~ 242m, 120 bins, 2m/bin |
| 检测类别 | car, truck, bus, trailer, barrier, motorcycle, bicycle, pedestrian, cone 等 |
| 当前最优 P90 纵向 | Frontal 0.64m / Oblique 0.49m |

---

### Slide 1.3：CRN 三模块架构

```
输入                                       输出
┌─────────────────┐                 ┌──────────────────┐
│ 单帧图像          │                 │ 3D 检测框          │
│ [1, 3, 384,1408] │                 │ [x,y,z,w,l,h,     │
│                  │                 │  sin,cos,vx,vy]   │
│ 雷达 PV 点云      │                 │                    │
│ [1, 1536, 5]     │                 │ + 热力图分类        │
│  5ch = [u,d,h,   │                 │                    │
│         rcs,speed]│                └──────────────────┘
│                  │                        ▲
│ 标定矩阵          │                        │
│ (内参/外参/增强)   │                        │
└──────┬───────────┘                        │
       │                                    │
       ▼                                    │
┌───────────────────────────────────────────┴──┐
│              CRN 三模块                       │
│                                              │
│  ① RVTLSSFPN  ──→ 图像 BEV [80, 128, 480]   │
│     (Radar View Transform 引导的 LSS)         │
│                                              │
│  ② PtsBackbone ──→ 雷达 BEV [80, 128, 480]  │
│     (PointPillars 编码)                       │
│     + 雷达占用率 [1, 128, 480] → 反馈给 ①     │
│                                              │
│  ③ MFAFuser ──→ 融合 BEV [128, 128, 480]    │
│     (6层可变形交叉注意力)                       │
│                                              │
│  ④ BEVDepthHead ──→ 检测结果                  │
│     (CenterPoint: 热力图 + 10D 回归)           │
└──────────────────────────────────────────────┘
```

**数据流关键点**：雷达分支同时输出两条路径 — `雷达上下文特征` 送入 MFAFuser 做融合，`雷达占用率` 反馈给图像分支的 RVT 做深度引导。

---

## 第2部分：数据预处理管线

---

### Slide 2.1：从原始数据到训练输入

```
原始传感器                              模型输入
┌────────────────────┐                ┌────────────────────────┐
│ 雷达 CSV           │                │ 雷达 PV 点云             │
│ (Angle,Range,Speed,│  ──── 4阶段 ──►│ [B, 1, 1, 1536, 5]     │
│  SNR, ID, Frame)   │    预处理      │ 5ch = [u,d,h,rcs,speed] │
│                    │                │                        │
│ 图像 PNG           │                │ 图像                    │
│ 3840×2160          │                │ [B, 1, 1, 3, 384,1408] │
│                    │                │                        │
│ JSON 标注           │                │ GT 3D 框               │
│ (3D boxes)         │                │ [N_gt, 10]             │
└────────────────────┘                └────────────────────────┘

预处理 4 阶段 (scripts/data_preprocessing_final/pipeline.py):
  stage1: radar2bin    CSV → 18ch 二进制 (极坐标→笛卡尔→世界坐标)
  stage2: derivatives  18ch → BEV 7ch + PV 7ch (世界→相机→像素投影)
  stage3: depth_gt     BEV → 相机平面 → 稀疏深度图
  stage4: build_infos  JSON → nuscenes_infos_*.pkl (训练元数据)
```

---

### Slide 2.2：stage1 — 雷达 CSV → 18ch 二进制

```
原始 CSV 每行: [Angle(°), Range(m), Speed(m/s), SNR(dB), ObjectID, Frame#]

处理流程 (pipeline.py _convert_csv_frame_to_bin):
  
  Step 1: 极坐标 → 雷达本地笛卡尔坐标
    高度过滤: depth_sq = (r·cosθ)² - H²  >  0.1  (H = 雷达安装高度)
    x_radar = √(depth_sq)      ← 雷达前方 (水平距离)
    y_radar = r·sinθ           ← 雷达左侧
    速度分解: vx_radar = v·cosθ, vy_radar = v·sinθ

  Step 2: 雷达本地 → 世界坐标
    x_world = x_radar·cos(yaw) - y_radar·sin(yaw) + radar_x
    y_world = x_radar·sin(yaw) + y_radar·cos(yaw) + radar_y
    vx_world, vy_world 同理旋转

  Step 3: 写入 18ch 二进制 (nuScenes 格式)
    18ch = [x, y, z=0, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, 
            is_quality, ambig, x_rms, y_rms, invalid, pdh0, vx_rms, vy_rms]
    实际填充: ch0=x, ch1=y, ch4=id, ch5=rcs, ch6=vx, ch7=vy, ch8=vx, ch9=vy
```

---

### Slide 2.3：stage2 — 18ch → BEV 7ch + PV 7ch

```
18ch 雷达点 [N, 18]  (世界坐标)
  │
  ├─→ BEV 7ch [N, 7]:
  │     [X, Y, Z, RCS, vx, vy, sweep_idx=0]
  │     → 保存为 radar_bev_filter/*.bin
  │     → 用于 stage3 生成深度 GT
  │
  └─→ PV 7ch [N, 7] (透视投影):
        世界坐标 → 相机坐标 (外参逆变换)
        P_cam = R^T · (P_world - t)
        
        相机坐标 → 像素坐标 (内参投影)
        u = fx · X_cam / Z_cam + cx
        v = fy · Y_cam / Z_cam + cy
        只保留 Z_cam > 0.5m 的点
        
        PV 7ch = [u, v, Z_cam, RCS, vx, vy, sweep_idx=0]
        → 保存为 radar_pv_filter/*.bin
        → 作为模型 PtsBackbone 的输入
```

---

### Slide 2.4：stage4 — JSON → 训练 PKL

```
JSON 标注每帧:
  {
    objects: [
      { category, translation: [x,y,z], size: [l,w,h], rotation: yaw, velocity: [vx,vy] }
    ],
    extrinsics_matrix_4x4 (可选),
    intrinsics (可选)
  }

处理流程:
  1. 相机外参: 优先从 JSON 的 extrinsics_matrix_4x4 提取 (world→camera 的逆)
     若无 JSON 外参 → 回退到场景参数 (cam_h, yaw)
  2. 目标标注: 类别映射 → nuscenes 格式 → 中心点 = translation + [0, 0, h/2]
  3. 训练/测试划分: 按 offset 严格划分 (如 frontal: train=[0,100000,800000], test=[200000,300000])
  4. 输出: nuscenes_infos_train.pkl, nuscenes_infos_test.pkl
           + 分场景: nuscenes_infos_frontal_{train,test}.pkl, oblique 同理
```

---

## 第3部分：模块① — RVTLSSFPN (图像→BEV)

---

### Slide 3.1：图像分支整体流程

```
输入图像 [B, 1, 1, 3, 384, 1408]
  │
  ▼ ResNet50 + SECONDFPN
  多尺度特征: 4 层 [128, 128, 128, 128] @ stride 16
  最终特征图: [B, 512, 24, 88]  (H=384/16, W=1408/16)
  │
  ▼ DepthNet (深度网络)
  ├─ depth:   [B, 120, 24, 88]  ← 每个像素在 120 个深度 bin 上的分布 (Softmax)
  └─ context: [B, 80,  24, 88]  ← 语义上下文特征
  │
  ▼ Frustum 构建 (Lift)
  depth ⊗ context → [B, 80, 120, 24, 88]   (外积: 每个深度假设复制一份特征)
  │
  ▼ RVT 融合 (Radar View Transform)
  雷达占用率 × 图像特征 → [B, 80, 120, 24, 88]
  与深度引导的结果 Concat → ViewAggregation Conv → [B, 80, 120, 24, 88]
  │
  ▼ 坐标变换 (Splat)
  Frustum 坐标 → Ego 坐标 → BEV 网格
  高度压缩 (sum over z) → [B, 80, 120, 88]
  │
  ▼ AverageVoxelPooling
  [B, 80, 128, 480]  ← 图像 BEV 特征
```

---

### Slide 3.2：LSS 原理 — Lift（2D → 3D）

```
核心思想: 单目图像无法确定深度 → 对每个像素预测一个深度概率分布
           在每一个可能的深度上"复制"一份语义特征

Step 1: 创建 Frustum (视锥体网格)
  create_frustum() 生成 [D, H, W, 3] 网格
  每个格点 (u, v, d) 表示:
    像素 (u,v) 在深度为 d 时的归一化相机坐标 (x_norm, y_norm, 1)
  
  [代码] base_lss_fpn.py create_frustum():
    d_coords × H/16 grids × W/16 grids
    深度: [2, 242]m, 120 bins, 2m/bin
    最后组成 (dx, dy, d, 1) 齐次坐标

Step 2: DepthNet 预测深度分布
  DepthNet 输入: [B, 512, H/16, W/16]  FPN特征
  DepthNet 输出: Concat(depth, context)
    depth:   Softmax(Conv(feat)) → [B, 120, H/16, W/16]
             含义: 像素(u,v)的物体的真实深度落在第k个bin的概率
    context: Conv(feat)          → [B, 80,  H/16, W/16]
             含义: 该像素的语义特征

Step 3: 外积 (Outer Product)
  img_feat_with_depth = depth.unsqueeze(1) × context.unsqueeze(2)
  = [B, 120, H/16, W/16] ⊗ [B, 80, H/16, W/16]
  = [B, 80, 120, H/16, W/16]
  
  含义: 对每个深度假设 d_k，把 context 特征复制一份
        如果深度分布预测正确，真实深度位置的 context 信号最强
```

---

### Slide 3.3：RVT — 雷达引导的视图变换

```
这是 CRN 区别于纯视觉方法 (BEVDepth) 的核心机制。

原理: 深度分布 D(d,u,v) 是"猜"出来的，不一定准确。
      雷达虽然稀疏，但它提供的深度是物理测量的，绝对可靠。
      
做法: 在 LSS 的 frustum 特征中，额外加入一个"雷达告诉我的"信号分支。

图像特征 context [B, 80, H/16, W/16]

  ├── 分支1: 深度引导 (传统 LSS)
  │   depth ⊗ context → [B, 80, 120, H/16, W/16]
  │
  └── 分支2: 雷达引导 (CRN 独创) ⬅ 关键!
      ┌─────────────────────────────────────────────┐
      │ 雷达占用率 radar_occupancy [B, 1, 128, 480]  │
      │   ↓ permute + reshape                       │
      │ [B, 128, 1, 480]                            │
      │                                             │
      │ 图像特征 (高度压缩后)                          │
      │   ↓ (image_feature × z_valid).sum(height)   │
      │ [B, 80, 1, 88]  (88=W/16, 1=height collapsed)│
      │                                             │
      │ 外积:                                       │
      │ radar_occupancy.unsqueeze(1)                 │
      │   × image_feature_collapsed.unsqueeze(2)     │
      │ = [B, 80, 128, 88]                          │
      └─────────────────────────────────────────────┘
      
  两分支 Concat → ViewAggregation (3层Conv+BN+ReLU)
  → [B, 80, 120, 88]  融合后的 frustum 特征

直观理解:
  雷达占用率 = "根据雷达测量，BEV 的 (x,y) 位置有物体的概率"
  外积 = "在这些有雷达回波的位置，增强对应的图像特征"
  效果 = 深度估计被雷达物理测量"锚定"，减少了猜测的不确定性
```

**[代码位置]** `rvt_lss_fpn.py` 第 313-319 行

---

### Slide 3.4：Splat — Frustum → BEV (坐标变换 + 体素池化)

```
这步将 frustum 空间的特征"拍"到 BEV 平面上。

Step 1: 反 IDA 变换 (撤销图像增强)
  图像在训练时做了 resize/crop/flip，需要反变换回到原始相机坐标
  points = IDA⁻¹ @ frustum_coords   [代码: rvt_lss_fpn.py:194]

Step 2: 像素 → 相机 → Ego 坐标
  combine = sensor2ego @ intrin⁻¹    [代码: rvt_lss_fpn.py:201]
  points = combine @ [u·d, v·d, d, 1]^T

  解读:
    intrin⁻¹ @ [u·d, v·d, d, 1]^T = 相机坐标 (X_c, Y_c, Z_c)
    sensor2ego @ 相机坐标 = Ego 坐标 (世界坐标)
    sensor2ego = 相机在世界坐标系中的位姿 (外参)

  路侧场景: ego = 世界 (传感器固定不动, ego2global = I)

Step 3: BDA 增强 (BEV 空间旋转/缩放/翻转)
  points = bda_mat @ points     [代码: rvt_lss_fpn.py:204-207]
  当前配置: 旋转 ±5°, 缩放 0.9~1.1, 随机翻转

Step 4: Z 轴过滤 + 压缩
  只保留 z ∈ [-2, 6] 的点 → 高度压缩(求和)
  理由: 雷达不提供可靠的高度信息，且压缩省显存

Step 5: 量化为 BEV 网格
  grid_x = (X - 0) / 0.5     → [0, 480]
  grid_y = (Y - (-25.6)) / 0.4  → [0, 128]
  grid_z 强制 = 0

Step 6: 平均体素池化 (AverageVoxelPooling)
  对落入同一 (grid_x, grid_y) 的所有 frustum 点 → 取平均

  为什么取平均而非求和？
    近处 BEV 格子对应更多 frustum 点 (透视效应)
    求和会导致近处特征远强于远处 → 网络偏向近处检测
    平均让每个 BEV 格子的特征幅度与距离无关 → 远处也能检测
```

---

## 第4部分：模块② — PtsBackbone (雷达→BEV)

---

### Slide 4.1：雷达分支整体流程

```
输入: 雷达 PV 点云 [B, 1, 1, 1536, 5]
  5ch = [u, d, h_placeholder, rcs_norm, speed_norm]
  
  u: 雷达点在图像上的水平像素坐标
  d: 相机坐标系下的深度 (m)
  h: 高度占位符 (恒为1，因雷达不提供可靠高度)
  rcs_norm: 归一化 RCS = (rcs - 4.783) / 7.576
  speed_norm: 归一化速率 = (|v| - 0.677) / 1.976

  [代码] nusc_det_dataset.py transform_radar_pv() 第 295-356 行

  ↓

Step 1: Voxelization (体素化)
  参数: voxel_size = [8, 1.0, 2.0]  (X粗 / Y中 / Z粗)
        point_cloud_range = [0, 2, 0, 1408, 242, 2]
        每个体素最多保留 8 个点
        [代码] pts_backbone.py voxelize() 第 106-132 行

  ↓

Step 2: PillarFeatureNet
  输入: 体素内点集 [M, 8, 5] + 体素中心坐标
  处理: PointNet 风格 — MLP 逐点 → MaxPool 逐体素
  输出: [M, 64] 每个体素一个 64 维特征
  [代码] pts_backbone.py:153

  ↓

Step 3: PointPillarsScatter
  将 M 个体素特征按坐标散布到伪图像
  输出: [B, 64, 128, 480]  (伪图像，尺寸=BEV尺寸)
  [代码] pts_backbone.py:154

  ↓

Step 4: SECOND Backbone (3层稀疏卷积)
  layer1: 64 → 64,  stride=1  [128, 480]
  layer2: 64 → 128, stride=2  [64,  240]
  layer3: 128→ 256, stride=2  [32,  120]
  [代码] pts_backbone.py:155

  ↓

Step 5: SECONDFPN (特征金字塔)
  三层特征上采样 + 融合 → [128, 128, 480]
  [代码] pts_backbone.py:156-157

  ↓

Step 6: 双头输出
  ┌─ pred_context:  Conv → [80, 128, 480]  ← 雷达上下文特征 (送入 MFAFuser)
  └─ pred_occupancy: Conv → Sigmoid → [1, 128, 480] ← 雷达占用率 (反馈给 RVT)
     初始 bias 约 -4.6，使得初始占用率 ≈ 0.01 (稀疏先验，避免初始假阳性)
  [代码] pts_backbone.py:166-169
```

---

### Slide 4.2：雷达 PV 点云的数据增强

```
训练时的雷达增强 (RDA):
  radar_idx = 随机选择 sweep (当前 N_sweeps=1, 只有当前帧)
  drop_ratio = 0.1 → 随机丢弃 10% 的雷达点 (模拟噪声/遮挡)

点云标准化:
  RCS 标准化: (rcs_raw - 4.783) / 7.576   ← 训练集统计的均值/标准差
  速度标准化: (|v| - 0.677) / 1.976

点数固定:
  最多保留 1536 个点 (max_radar_points_pv)
  不足 1536 → 填充 -999 占位
  若 0 个有效点 → 填充一个 dummy 点 (避免空 tensor)

通道重排:
  [u, d, h, rcs, speed] → permute → 最终输入 [u, d, h, rcs, speed]
  注意: d 和 h 位置有交换 [代码: transform_radar_pv 第354行]
```

---

## 第5部分：模块③ — MFAFuser (多模态融合)

---

### Slide 5.1：为什么要融合，而不简单拼接？

```
简单拼接: Concat(img_feat, pts_feat) → Conv
  问题: 图像 BEV 特征语义丰富但空间位置不准
        雷达 BEV 特征空间精准但稀疏、噪声多
        拼接+卷积无法精确处理两模态间的空间错位

MFAFuser 方案: 可变形交叉注意力 (Deformable Cross Attention)
  每个 BEV 位置作为一个 query
  query 可以 "看向" 图像特征和雷达特征的任意位置
  自主决定: 在这个位置，我应该更信任图像还是雷达？

[代码] multimodal_feature_aggregation.py MFAFuser 类
```

---

### Slide 5.2：MFAFuser 结构详解

```
输入:
  feat_img [B, 80,  128, 480]  ← 图像 BEV (来自 RVTLSSFPN)
  feat_pts [B, 80,  128, 480]  ← 雷达 BEV (来自 PtsBackbone)

═══════════════════════════════════════════════════
Step 1: 归一化 + 拼接
  feat_img → LayerNorm → flatten → [B, H·W, 80]
  feat_pts → LayerNorm → flatten → [B, H·W, 80]
  
Step 2: Query 构建
  z_q = Linear( Concat(feat_img, feat_pts) )
  = [B, H·W, 128]    ← 128维 query (embed_dims=128)

Step 3: 位置编码
  LearnedPositionalEncoding → bev_pos [B, H·W, 128]
  这是可学习的参数 (不是固定的正弦编码)
  query = z_q + bev_pos

Step 4: 参考点
  ref_2d = 2D 归一化坐标 [0,1]²
  每个 query 位置 (x/W, y/H)

═══════════════════════════════════════════════════
Step 5: 6 层 MDCA + FFN (循环 6 次)
  
  For each layer:
    
    ① DeformableCrossAttention:
       query (128D) → Linear → 
         ├─ sampling_offsets:  [H=4, M=2, K=4, 2]  ← 采样偏移量
         └─ attention_weights: [H=4, M=2, K=4]     ← 注意力权重 (Softmax)
       
       对每个头 h、每个模态 m、每个采样点 k:
         采样位置 = ref + offset
         在 feat_m 的双线性插值取值
         加权求和 = Σ A_hmk · Value_m(sampling_loc)
       
       最终聚合所有头 → 输出 query [B, H·W, 128]
       
    ② LayerNorm(query)
    ③ FFN: Linear(128→256) → ReLU → Linear(256→128)
    ④ LayerNorm + Residual

  [代码] multimodal_deformable_cross_attention.py DeformableCrossAttention 类

Step 6: 输出
  Reshape → [B, 128, 128, 480]
  ReduceConv → [B, 128, 128, 480]  ← 融合后的 BEV 特征
═══════════════════════════════════════════════════
```

---

### Slide 5.3：可变形注意力的直观理解

```
普通 Attention: query 关注所有位置 → O(N²) 复杂度
可变形 Attention: query 只关注 K=4 个采样点 → O(NK) 复杂度

对于一个 query 位置 (比如 BEV 的 (240, 64) 对应道路上 120m 处):

  ┌──────────────────────────────────┐
  │ 参考点 ref = (120/240, 64/128)   │
  │             = (0.5, 0.5)        │
  │                                  │
  │ 预测的 4 个采样偏移:              │
  │   Δp₁ → 特征图上的采样位置 p₁    │
  │   Δp₂ → 特征图上的采样位置 p₂    │
  │   Δp₃ → 特征图上的采样位置 p₃    │
  │   Δp₄ → 特征图上的采样位置 p₄    │
  │                                  │
  │ 同时预测 4 个注意力权重 A₁..A₄   │
  │                                  │
  │ 分别在图像特征图 和 雷达特征图上   │
  │ 各自采样 → Value_img, Value_pts  │
  │ 加权求和 → 该 query 的新特征      │
  └──────────────────────────────────┘

为什么采样偏移是可学习的？
  网络自己学到: "如果当前 BEV 位置有雷达点，采样偏移应该倾向于雷达特征图"
               "如果雷达点稀疏，采样偏移应该更依赖图像特征"
  
为什么每个模态独立的 Value Proj？
  value_proj_img ≠ value_proj_pts
  两模态的特征分布不同 → 需要各自独立投影
  即使一个传感器故障，另一个仍能正常工作
```

---

### Slide 5.4：MFA 配置参数

| 参数 | 值 | 含义 | 代码位置 |
|------|-----|------|---------|
| img_dims | 80 | 图像 BEV 通道数 | fuser_conf |
| pts_dims | 80 | 雷达 BEV 通道数 | fuser_conf |
| embed_dims | 128 | Query/融合特征维度 | fuser_conf |
| num_layers | 6 | MDCA + FFN 的层数 | fuser_conf |
| num_heads | 4 | 注意力头数 | fuser_conf |
| num_points | 4 | 每头每模态采样点数 | DeformableCrossAttention |
| bev_shape | (128, 480) | BEV 空间尺寸 [Y, X] | fuser_conf |

---

## 第6部分：模块④ — BEVDepthHead (检测头)

---

### Slide 6.1：CenterPoint 检测原理

```
Anchor-Free 检测: 不预设候选框，直接预测目标中心点 + 属性

BEV 特征 [128, 128, 480]  ← 融合后的 BEV
  │
  ▼ BEV Backbone: ResNet18, 3 stage
  │   stage1: 128 → 160, stride=1  [128, 480]
  │   stage2: 160 → 320, stride=2  [64,  240]
  │   stage3: 320 → 640, stride=2  [32,  120]
  │
  ▼ BEV Neck: SECONDFPN
  │   上采样 ×1,×2,×4,×8 → 4层 [64,64,64,64]
  │   Concat → [256, 128, 480]
  │
  ▼ Shared Conv: Conv3×3(256→256)
  │
  ▼ 6 个 Task-Specific Heads (每个任务独立):

  Task 1 — car:                    heatmap[1] + reg[10]
  Task 2 — truck, construction:    heatmap[2] + reg[10]
  Task 3 — bus, trailer:           heatmap[2] + reg[10]
  Task 4 — barrier:                heatmap[1] + reg[10]
  Task 5 — motorcycle, bicycle:    heatmap[2] + reg[10]
  Task 6 — pedestrian, cone:       heatmap[2] + reg[10]

  每个 Task Head 输出:
    heatmap:  Conv1×1(N_classes)  → 每个位置是目标中心的概率
    reg:       Conv1×1(2)  → Δx, Δy (中心位置亚像素偏移)
    height:    Conv1×1(1)  → z (目标底部高度)
    dim:       Conv1×1(3)  → w, l, h (尺寸)
    rot:       Conv1×1(2)  → sin(yaw), cos(yaw) (朝向)
    vel:       Conv1×1(2)  → vx, vy (速度)
                 合计: 10D 回归
```

---

### Slide 6.2：训练目标 — 高斯热力图

```
为什么用高斯热力图而不是 one-hot？
  → 检测框中心附近的点也有一定概率包含目标 (定位模糊性)
  → 高斯分布 → 网络学到平滑的概率分布，更容易优化

目标生成 (gt_box → heatmap target):

Step 1: 中心坐标 → 热力图网格
  coor_x = (x - 0) / 0.5 / 1 = x / 0.5      → [0, 480]
  coor_y = (y - (-25.6)) / 0.4 / 1           → [0, 128]

Step 2: 计算高斯半径
  width_on_grid  = w / 0.5    (目标宽度占几个格子)
  length_on_grid = l / 0.4    (目标长度占几个格子)
  radius = gaussian_radius((length_on_grid, width_on_grid), min_overlap=0.1)
  radius = max(min_radius=2, radius)

Step 3: 在对应类别通道上绘制
  heatmap[class_id, y_center, x_center] += exp(-d²/(2σ²))
  其中 σ 与 radius 成正比，d 是到中心的距离

  多目标重叠时: 逐元素取 max

Step 4: 回归目标 (仅对 heatmap peak 位置)
  reg_target = [Δx, Δy, z, w, l, h, sin(yaw), cos(yaw), vx, vy]

[代码] bev_depth_head_det.py get_targets_single() 第 135-276 行
```

---

### Slide 6.3：损失函数

```
总损失: L = L_heatmap + L_bbox + L_depth

┌──────────────────────────────────────────────────────┐
│ ① L_heatmap: GaussianFocalLoss                       │
│                                                      │
│ FL(p_t) = -(1-p_t)^γ · log(p_t)   (γ=2.0)          │
│                                                      │
│ 作用: 对已经预测得很好的位置 (p_t → 1)，降低 loss     │
│       让网络专注于预测困难的负样本                     │
│       高斯权重 → 离中心越远，正样本权重越小            │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ ② L_bbox: SmoothL1Loss (β=0.05)                      │
│                                                      │
│ 在 |x| < 0.05 时用 L2 loss (平滑区)                  │
│ 在 |x| ≥ 0.05 时用 L1 loss (鲁棒区)                  │
│                                                      │
│ Code Weights (每个回归维度的损失权重):                 │
│                                                      │
│ 参数:  Δx   Δy   z    w    l    h   sin  cos  vx  vy │
│ 权重:  4.0  2.0  0.5  1.5  1.5  0.5  3.0  3.0  0.2 0.2│
│        ↑                  ↑              ↑      ↑     │
│   纵向最高           朝向次之       速度最低          │
│                                                      │
│ 含义:                                                │
│   Δx=4.0 → 纵向定位是路侧核心指标，绝对优先           │
│   Δy=2.0 → 横向次之，比纵向容易                       │
│   sin/cos=3.0 → 方向对路侧场景重要                    │
│   vx/vy=0.2 → 速度最低 (雷达速度有噪声)                │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ ③ L_depth: Binary Cross Entropy, weight=0.5          │
│                                                      │
│ 只对前景像素计算 (有雷达深度标注的位置)                │
│                                                      │
│ 深度 GT 生成:                                         │
│   雷达点 (X,Y,Z) → 投影到相机 → (u, v, depth)         │
│   16×16 像素块 → 取最小深度 → 得到 135×240 稀疏深度图  │
│   → One-hot 编码到 120 个深度 bin                      │
│                                                      │
│ weight=0.5 原因:                                      │
│   毫米波雷达点云稀疏 (远不如 LiDAR)                    │
│   → 深度 GT 有大量空洞 → 权重太高会过拟合噪声          │
└──────────────────────────────────────────────────────┘
```

---

## 第7部分：训练与推理流程

---

### Slide 7.1：完整训练流程

```
训练数据加载 (NuscDatasetRadarDet):

  1. 从 PKL 读取帧元数据
  2. 加载图像 → IDA 增强
     resize: 3840×2160 → 384×1408 (比例 0.365)
     random crop, random flip (50%), rotation=0
  3. 加载雷达 PV → 同样的 IDA 坐标变换 → 标准化 → dropout 10%
  4. 加载深度 GT → IDA 变换 → 稀疏深度图
  5. 读取 GT Boxes → world→ego 坐标变换
  6. BDA 增强: 旋转 ±5°, 缩放 0.9-1.1, 随机翻转 (50%)

训练超参数:

| 参数 | 值 |
|------|-----|
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 |
| Scheduler | MultiStepLR, milestones @ 80% & 95% epochs |
| Epochs | 96 |
| Batch Size | 1 (单 GPU) |
| Precision | FP16 (混合精度) |
| Gradient Clip | 5.0 |

训练循环 (training_step):
  sweep_imgs + mats + pts_pv → cuda
  forward → (preds, depth_preds)
  get_targets(gt_boxes, gt_labels) → (heatmaps, anno_boxes, inds, masks)
  L_total = L_heatmap + L_bbox + 0.5 × L_depth
  backward → optimizer.step
```

---

### Slide 7.2：推理与后处理

```
输入: 单帧图像 + 雷达点云
  ↓
模型前向 (is_train=False):
  PtsBackbone → radar_context + radar_occupancy
  RVTLSSFPN(radar_occupancy) → img_bev + depth_preds
  MFAFuser(img_bev, radar_context) → fused_bev
  BEVDepthHead(fused_bev) → 6 个 task 的 (heatmap + regression)

  ↓  后处理 (CenterPointBBoxCoder.decode):

Step 1: Sigmoid(heatmap) → 找到 top-K 峰值 (每类最多 1000)
Step 2: 提取峰值位置对应的回归值
Step 3: 解码 3D 框
  x_center = grid_x × 0.5 + 0 + Δx      (BEV grid → 米)
  y_center = grid_y × 0.4 + (-25.6) + Δy
  z        = reg_z
  w,l,h    = reg_dim  (或 exp() 如果 norm_bbox)
  yaw      = atan2(sin, cos)
  vx, vy   = reg_vel

Step 4: Circle NMS
  对每个类别: 基于 BEV 中心距离抑制重复检测
  nms_thr = 0.2 → 两个框中心距离 < 0.2m 的视为重复

Step 5: 置信度过滤
  score_threshold = 0.35 → 保留高置信度检测

Step 6: 坐标变换 (ego → global)
  世界坐标 = ego2global @ ego坐标
  (路侧场景 ego2global = I，因为传感器固定在世界坐标系)
```

---

### Slide 7.3：数据增强策略

```
三种增强，分别在三个空间中独立进行:

┌─────────────────────────────────────────────────┐
│ IDA (Image Data Augmentation) — 图像空间         │
│                                                 │
│ 目的: 模拟不同距离、光照下的观测                   │
│ 操作:                                           │
│   resize:   2160→384, 缩放范围 0.36-0.38        │
│             → 模拟车辆距离的微小变化              │
│   crop:     W 随机, H 底对齐 → 384×1408          │
│   flip:     50% 概率水平翻转                      │
│   rotate:   0° (不做旋转!)                       │
│             因为旋转会破坏 RVT 的高度压缩假设      │
│                                                 │
│ 注意: 雷达 PV 点和深度 GT 同时应用相同 IDA 变换   │
│       保证像素级对齐                              │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ BDA (BEV Data Augmentation) — BEV 空间           │
│                                                 │
│ 目的: 模拟不同的车辆朝向和尺度                     │
│ 操作:                                           │
│   rotate:   ±5° (路侧车辆朝向集中，缩小范围)      │
│   scale:    0.9 ~ 1.1                           │
│   flip_x:   50% 概率 (BEV X轴翻转)               │
│   flip_y:   50% 概率 (BEV Y轴翻转)               │
│                                                 │
│ GT Box 同步变换: 坐标、尺寸、朝向、速度全部更新    │
│ bda_mat 记录变换矩阵 → 传给模型用于 frustum 逆变换  │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ RDA (Radar Data Augmentation) — 雷达点云空间      │
│                                                 │
│ 目的: 模拟雷达噪声和部分遮挡                       │
│ 操作:                                           │
│   sweep 选择: N_sweeps=1 (当前仅单帧)            │
│   dropout:  10% 随机丢弃雷达点                    │
│   点数限制: 最多 1536 个点，不足的填充 -999       │
└─────────────────────────────────────────────────┘
```

---

## 第8部分：核心创新点总结

---

### Slide 8.1：三个创新机制

```
创新 1: Radar View Transform (RVT)
══════════════════════════════════════
  问题: 单目深度估计不可靠 (ill-posed)
  解法: 雷达提供物理深度先验，引导深度分布
  实现: 深度引导 ⊗ context + 雷达占用 ⊗ context → ViewAggregation
  效果: 深度估计从"纯猜测"变成"有物理锚定的估计"

创新 2: 双流并行 + 交叉反馈
══════════════════════════════
  图像流: RVTLSSFPN → 密集语义 BEV 特征
  雷达流: PtsBackbone → 精确几何 BEV 特征 + 占用率
  反馈: 雷达占用率 → 引导图像流的 RVT
  融合: 两流在 MFAFuser 中通过注意力自适应融合

创新 3: 可变形交叉注意力融合 (MDCA)
══════════════════════════════════
  每个 BEV 位置自适应选择关注哪个模态、哪个位置
  每模态独立的值投影 → 传感器故障鲁棒
  可学习的采样偏移 → 自动处理跨模态空间错位
  线性复杂度 O(NK) → 可扩展到远距离 (240m)
```

---

### Slide 8.2：路侧场景的关键设计决策

```
1. BEV 长条形设计: 480×128 (而非正方形 128×128)
   道路是狭长的 → 纵向分辨率 0.5m 远高于横向 0.4m
   → 240m 检测范围覆盖高速公路长距离需求

2. 纵向损失高权重: code_weights[0] = 4.0
   路侧场景核心指标是纵向定位误差
   → loss 中纵向权重是横向的 2 倍，是尺寸的 ~3 倍

3. 深度损失降低: weight = 0.5
   雷达 GT 远稀疏于 LiDAR → 避免过拟合稀疏噪声

4. 保守增强策略:
   旋转 ±5° (而非 ±22.5°) → 路侧车辆沿道路行驶，朝向分布集中
   resize 0.36-0.38 (而非 0.35-0.55) → 减少几何扰动

5. 单相机 + Camera-Aware 关闭
   无跨相机区分需求 → 减少参数 → 降低过拟合风险

6. 分场景独立训练
   Frontal (0°) 和 Oblique (15°) 数据分布差异大
   → 分别训练，各自优化
```

---

### Slide 8.3：代码模块索引

| 模块 | 文件 | 核心类/函数 |
|------|------|-----------|
| 主模型 | `models/camera_radar_net_det.py` | `CameraRadarNetDet` |
| 图像分支 | `layers/backbones/rvt_lss_fpn.py` | `RVTLSSFPN`, `DepthNet`, `ViewAggregation` |
| 雷达分支 | `layers/backbones/pts_backbone.py` | `PtsBackbone` |
| 融合模块 | `layers/fuser/multimodal_feature_aggregation.py` | `MFAFuser` |
| 交叉注意力 | `layers/modules/multimodal_deformable_cross_attention.py` | `DeformableCrossAttention` |
| 检测头 | `layers/heads/bev_depth_head_det.py` | `BEVDepthHead` |
| 数据集 | `datasets/nusc_det_dataset.py` | `NuscDatasetRadarDet` |
| 训练框架 | `exps/base_exp.py` | `BEVDepthLightningModel` |
| 实验配置 | `exps/det/CRN_r50_256x704_128x128_4key.py` | `CRNLightningModel` |
| 体素池化 | `ops/average_voxel_pooling_v2/` | `AverageVoxelPooling` (CUDA) |
| 数据预处理 | `scripts/data_preprocessing_final/pipeline.py` | 4 阶段管线 |
| 预处理配置 | `scripts/data_preprocessing_final/config.py` | `ScenarioConfig`, `PathConfig` |

---

> **更新时间**: 2026-06-23
> **全部内容基于项目实际代码**，参数、配置、流程均来自当前运行的实现。
