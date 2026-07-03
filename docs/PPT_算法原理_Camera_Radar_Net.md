# Camera Radar Net (CRN) 算法原理详解 — PPT 素材

> **依据**：ICCV 2023 论文原文 (LaTeX source: `3_Method.tex`, `9_Appendix.tex`) + 项目实际代码
> **标注说明**：所有内容均标注来源：论文原文用 `[论文]`，代码实现用 `[代码]`，路侧适配用 `[路侧适配]`

---

## PPT 第1部分：项目背景与 RVT 核心思想

---

### Slide 1.1：为什么需要相机-雷达融合？

```
相机 (Camera)                      毫米波雷达 (Radar)
┌────────────────────┐             ┌────────────────────┐
│  ✓ 纹理丰富          │             │  ✓ 精确深度/速度     │
│  ✓ 语义信息强        │             │  ✓ 对光照不敏感      │
│  ✓ 高分辨率          │             │  ✓ 全天候工作        │
│  ✗ 无深度信息        │             │  ✗ 点云极度稀疏      │
│  ✗ 光照敏感          │             │  ✗ 无颜色/纹理       │
│  ✗ 天气影响大        │             │  ✗ 噪声多 (多径)     │
└────────────────────┘             └────────────────────┘
         │                                    │
         └────────────┬───────────────────────┘
                      ▼
              CRN 融合框架
         (取长补短，1+1>2)
```

**[论文] 核心理念**（论文 Sec.3.1）：雷达测量具有两个关键特性——**精确的深度/速度**，但**稀疏且有噪声**，且**不提供高度信息**。因此 CRN 的设计原则是：**自适应地利用雷达**（"exploit radar in an adaptive manner to handle its sparsity and ambiguity"），而非简单地把雷达当成 LiDAR 来用。

---

### Slide 1.2：场景差异：nuScenes → 路侧

| 维度 | nuScenes (论文原版) | 路侧场景 (本项目) |
|------|-------------------|------------------|
| 安装平台 | 车顶 | 交通杆/龙门架 |
| 相机数量 | 6 个环视 (N=6) | **1 个前视 (N=1)** |
| 雷达数量 | 5 个 | **1 个** |
| 检测范围 | ±51.2m | **[路侧适配] 0~240m (纵向) × ±25.6m (横向)** |
| BEV 分辨率 | 0.8m → 128×128 | **[路侧适配] 0.5m×0.4m → 480×128** |
| 深度范围 | [2.0, 58.0]m, D=112, bin=0.5m | **[路侧适配] [2.0, 242.0]m, D=120, bin=2.0m** |
| 安装高度 | ~1.7m | **[路侧适配] 6~25m** |
| 运动状态 | 运动 (自车ego运动) | **[路侧适配] 静止 (ego2global = I)** |
| 时序帧 _[论文 Tab.9]_ | T=3 (提交) / T=1 (消融) | **[路侧适配] T=1 (当前单帧)** |
| 目标类别 | 10 类 | **[路侧适配] 6 类 (以车辆为主)** |

---

## PPT 第2部分：CRN 整体架构 [论文 Fig.1]

---

### Slide 2.1：论文架构总览

```
                      输入                                   输出
             ┌──────────────────┐              ┌─────────────────────┐
             │ N 个环视图像       │              │ 3D 检测框            │
             │ + 雷达点云         │              │ [x, y, z, w, l, h,  │
             │ + 标定参数         │              │  yaw, vx, vy]       │
             └───────┬──────────┘              └─────────────────────┘
                     │
        ┌────────────┴─────────────────────────────────┐
        │              CRN 三模块 [论文 Sec.3]           │
        │                                               │
        │  ① RVT (Radar-assisted View Trans.) Sec.3.2  │
        │     图像特征 + 深度分布 ──┐                    │
        │     雷达特征 + 占用率  ──┤→ BEV 特征          │
        │                          │                    │
        │  ② MFA (Multi-modal Feature Agg.) Sec.3.3    │
        │     MDCA 可变形交叉注意力 → 融合 BEV          │
        │                                               │
        │  ③ Task Heads Sec.3.4                        │
        │     CenterPoint → 热力图 + 10D 回归           │
        └───────────────────────────────────────────────┘
```

**[论文 vs 代码] 关键实现差异：**
- 论文的雷达编码在**frustum 空间 (d,u,v)** [论文 Eq.5]；代码的 PtsBackbone 输出 **BEV 空间特征**。这是因为路侧单相机场景下 frustum→BEV 映射更直接。
- 论文 MFA 使用 **8 个注意力头** [论文附录 C]；代码/路侧使用 **4 个头** (`fuser_conf.num_heads=4`)。

---

## PPT 第3部分：模块① — RVT (Radar-assisted View Transformation) [论文 Sec.3.2]

---

### Slide 3.1：论文原文 — 问题形式化

**[论文] LSS 范式**（论文 Eq.1）：

```
F_3D(x,y,z) = M( F_2D(u,v) ⊗ D(u,v) )
              ↑              ↑       ↑
         视图变换        2D特征   深度分布（外积）
```

其中 ⊗ 表示外积（outer product），M 是视图变换模块（如 Voxel Pooling）。

**CRN 的 RVT 改进**：显式利用雷达测量来提升这个变换过程。
论文原文表述："We aim to explicitly improve the transformation process using radar measurement."

---

### Slide 3.2：论文原文 — 图像特征提取与深度分布

**[论文] Eq.2**：

```
C_I^{PV} = Conv(F_I)                         # 图像上下文特征 [N, C, H, W]
D_I(u,v) = Softmax(Conv(F_I)(u,v))           # 深度分布 [N, D, H, W]
```

其中：
- F_I 是 ResNet+FPN 输出的 16× 下采样特征图
- (u,v) 是图像平面坐标
- D 是深度 bin 数量（论文 D=112，范围 [2.0, 58.0]m，bin=0.5m）
- Softmax 保证每个像素的深度分布和为 1

**[代码]** `rvt_lss_fpn.py` 第 288-301 行：DepthNet 输出 concat(depth, context)，depth 部分做 softmax：
```python
depth_feature = self._forward_depth_net(source_features, mats_dict)
image_feature = depth_feature[:, self.depth_channels:(...)]
depth_occupancy = depth_feature[:, :self.depth_channels].softmax(dim=1)
```

---

### Slide 3.3：论文原文 — 雷达特征编码与雷达占用率

**[论文] Eq.5**：

```
C_R^{FV} = Conv(F_R)                         # 雷达上下文特征 [N, C, D, W]
O_R(d,u) = σ(Conv(F_R)(d,u))                # 雷达占用率 [N, 1, D, W]，sigmoid
```

**处理流程** [论文 Sec.3.2]：

```
雷达点云 (x,y,z,rcs,doppler)
  │
  ▼ 投影到每个相机视图（保持深度）
相机像素坐标 (u, v, depth)
  │
  ▼ 体素化到 frustum 空间 (d, u, v)
  │  v=1 (pillar-style，因为雷达不提供可靠的 elevation)
  │
  ▼ PointNet + Sparse Conv (SECOND)
F_R ∈ R^{N×C×D×W}
  │
  ├──→ C_R^{FV}:  雷达上下文特征 (Conv)
  └──→ O_R(d,u):  雷达占用率 (Conv + Sigmoid)
```

**关键区别**：深度分布 D_I 用 Softmax（多分类），雷达占用率 O_R 用 Sigmoid（二分类）。
**[论文原文]**："sigmoid is used instead of softmax since radar occupancy is not necessarily one-hot encoded as a depth distribution."

**[代码差异]** `pts_backbone.py`：PtsBackbone 输出 BEV 空间特征 [B, 80, 128, 480] 而非 frustum 空间。雷达占用率 `pred_occupancy` 也在 BEV 空间 [B, 1, 128, 480]，经 reshape 后在 RVT 中使用。

---

### Slide 3.4：论文原文 — Frustum View Transformation（核心公式）

**[论文] Eq.3**（RVT 的核心）：

```
C_I^{FV} = Conv[ C_I^{PV} ⊗ D_I  ;  C_I^{PV} ⊗ O_R ]
             ↑       ↑       ↑         ↑       ↑
           Conv   图像特征  深度分布   图像特征  雷达占用率
                  └── 深度引导的分支 ──┘  └── 雷达引导的分支 ──┘
                  └────────── Concat ──────────┘
```

**含义**：
- `C_I^{PV} ⊗ D_I`：用估计的深度分布来"散布"图像特征到 frustum（传统 LSS 的做法）
- `C_I^{PV} ⊗ O_R`：用雷达物理测量来"散布"图像特征到 frustum（CRN 独创）
- `[ ; ]`：沿 channel 维度拼接两个分支
- `Conv`：用 ViewAggregation 卷积网络融合两者

**为什么 RVT 有效？** 深度估计本质上是 ill-posed（单目→密度是不可靠的猜测），而雷达提供了真实物理世界中"这里有个障碍物"的信号。即使雷达点稀疏，这个信号也足以在对应的 BEV 位置增强图像特征。

**[代码]** `rvt_lss_fpn.py` 第 313-319 行：
```python
if self.radar_view_transform:
    # 雷达占用率 → frustum 空间
    radar_occupancy = pts_occupancy.permute(0, 2, 1, 3).contiguous()
    # 图像特征沿高度压缩
    image_feature_collapsed = (image_feature * geom_xyz_valid.max(2).values).sum(2).unsqueeze(2)
    # 外积：雷达占用 × 图像特征
    img_feat_with_radar = radar_occupancy.unsqueeze(1) * image_feature_collapsed.unsqueeze(2)
    # 拼接 + ViewAggregation
    img_context = torch.cat([img_feat_with_depth, img_feat_with_radar], dim=1)
    img_context = self._forward_view_aggregation_net(img_context)
```

**[论文原文]**：高度维通过求和（summation）压缩：
"we collapse the image context feature by summation along the height axis"

---

### Slide 3.5：论文原文 — BEV 变换 (Average Voxel Pooling)

**[论文] Eq.4**：

```
F^{BEV} = M({F_i^{FV}}_{i=1}^{N})
```

将 N 个相机 frustum 视图的 context feature（含图像和雷达）变换到统一的 BEV 空间。

**[论文原文]**（关键实现细节）：
"We adopt CUDA-enabled Voxel Pooling and **modify it to aggregate features within each BEV grid using average pooling instead of summation**."

**为什么用平均池化而非求和？** 由于透视投影，近处的 BEV grid 对应更多 frustum 点（密集），远处对应更少（稀疏）。如果求和，近距离 BEV 特征会远强于远距离。平均池化保证每个 BEV grid 的特征强度与距离无关，网络能预测更一致的 BEV 特征图。

**[代码]** `ops/average_voxel_pooling_v2/`：自定义 CUDA kernel，每个 voxel 内对特征取平均（除以计数）。

---

### Slide 3.6：几何变换链 — frustum → BEV 的坐标变换

**[代码]** `rvt_lss_fpn.py` `get_geometry_collapsed()` 第 187-214 行：

```
Step 1: 构造 frustum grid
  frustum = create_frustum() → [D, H, W, 3]
  其中每个点 (x, y, d) 表示"深度为 d 时，像素 (u,v) 在归一化相机坐标下的位置"

Step 2: 反 IDA 变换（撤销图像增强）
  points = IDA⁻¹ @ frustum
  → 恢复原始相机坐标系

Step 3: 像素 → 归一化相机坐标
  combine = sensor2ego @ intrin⁻¹   (外参 @ 内参的逆)
  points = combine @ [u·d, v·d, d, 1]ᵀ
  → 从像素坐标 (u,v)×深度 d 变换到 ego 坐标 (x, y, z)

Step 4: BDA 变换（BEV 增强）
  points = bda_mat @ points
  → 对 ego 坐标施加旋转/缩放/翻转

Step 5: Z 轴压缩
  z_min=-2, z_max=6
  超出此范围的 frustum 点被 mask 掉
  有效点 → VoxelPooling
```

---

## PPT 第4部分：模块② — PtsBackbone (雷达分支) [论文 Sec.3.2]

---

### Slide 4.1：论文 vs 代码 — 雷达编码的两种方式

| | 论文 [Sec.3.2] | 代码/路侧 [pts_backbone.py] |
|---|---|---|
| 输入空间 | Frustum (d,u,v) | BEV (X,Y) |
| 体素化坐标 | 深度(d) × 像素宽(u) × 像素高(v=1) | X(纵向) × Y(横向) × Z(高度) |
| 输出空间 | Frustum: C×D×W | BEV: C×128×480 |
| 卷积方向 | top-view (d,u) 即"深度×宽度" | BEV (Y×X) |
| 占用率含义 | 每个 (d,u) 是否有雷达点 | 每个 BEV 格子是否有雷达点 |

**[论文原文]**：雷达点只在 frustum 空间编码。雷达占用 O_R(d,u) 与图像深度分布 D_I(d,u,v) 在同一空间做外积 → 公式 C_I^{PV} ⊗ O_R 可直接进行。

**[路侧适配]**：由于单相机固定视角，frustum 和 BEV 的映射是固定的，因此代码将雷达编码为 BEV 空间特征，再在 RVT 中 reshape 回 frustum 维度用于外积。

---

### Slide 4.2：代码 — PtsBackbone 完整流程

**[代码]** `pts_backbone.py` 第 134-176 行：

```
输入: 雷达 PV 点云 [B, N_sweeps, N_cams, 1536 points, 5ch]
  5ch = [u, d, h, rcs_norm, speed_norm]

Step 1: Voxelization [pts_backbone.py:116-132]
  参数（路侧当前配置）:
    voxel_size = [8, 1.0, 2.0]      ← X=8m粗, Y=1m中, Z=2m粗
    point_cloud_range = [0, 2, 0, 1408, 242, 2]
    max_voxels = (768, 1024)
    max_num_points_per_voxel = 8

Step 2: PillarFeatureNet（体素特征编码）[pts_backbone.py:153]
  输入: [M_voxels, 8, 5] + voxel_center
  → PointNet: 5ch + 3ch(center_offset) → MLP → max_pool
  输出: [M_voxels, 64]

Step 3: PointPillarsScatter（散射到伪图像）[pts_backbone.py:154]
  输出: 伪图像 [B, 64, 128, 480]
  （对应 BEV 的 Y=128格, X=480格 → 0.5m×0.4m 分辨率）

Step 4: SECOND Backbone + SECONDFPN [pts_backbone.py:155-157]
  SECOND (3层稀疏卷积):
    layer1: 64 → 64,  stride=1  [128×480]
    layer2: 64 → 128, stride=2  [64×240]
    layer3: 128→ 256, stride=2  [32×120]
  SECONDFPN: 三层上采样融合 → 统一 [128, 128, 480]

Step 5: 双头预测 [pts_backbone.py:166-169]
  ├─ pred_context:  [80, 128, 480]  ← 雷达上下文特征（送入 MFA）
  └─ pred_occupancy: [1, 128, 480]  ← 雷达占用率（sigmoid，送入 RVT）
     bias 初始化为 log(0.01/(1-0.01)) ≈ -4.6
     → 初始占用率 ≈ 0.01（稀疏先验）
```

---

## PPT 第5部分：模块③ — MFA (Multi-modal Feature Aggregation) [论文 Sec.3.3]

---

### Slide 5.1：论文 — MFA 的设计动机

**[论文原文]**（Sec.3.3）：

"Image feature has rich semantic cues, but their spatial position is inherently inaccurate; on the other hand, radar feature is spatially accurate, but contextual information is insufficient and noisy."

"Naive approaches are channel-wise concatenation or summation, but these cannot handle neither spatial misalignment nor ambiguity between two modalities."

**设计目标**：用注意力机制自适应地融合，让网络自己决定在每个 BEV 位置应该更信任图像还是雷达。

---

### Slide 5.2：论文 — MDCA (Multi-modal Deformable Cross Attention) 公式

**[论文] Eq.6**（核心公式）：

```
MDCA(z_q, p_q, x_m) =

  H              M    K
  Σ  W_h [ Σ    Σ    A_hmqk · W'_hm · x_m( φ_m(p_q + Δp_hmqk) ) ]
  h            m    k

其中:
  z_q      : query 特征（图像+雷达 BEV concat 后投影）
  p_q      : 参考点（2D BEV 归一化坐标，[0,1]²）
  x_m      : 多模态特征图 {C_I^{BEV}, C_R^{BEV}}
  H        : 注意力头数（论文=8, 道路侧代码=4）
  M        : 模态数（=2: 图像 + 雷达）
  K        : 采样点数（论文=4, 代码=4）
  A_hmqk   : 注意力权重（由 z_q 线性投影得到，对 m,k 做 Softmax 归一化）
  Δp_hmqk : 采样偏移量（由 z_q 线性投影得到）
  W'_hm    : 每模态独立的值投影矩阵（modality-specific）
  W_h      : 输出投影矩阵
  φ_m(p)   : 缩放函数（两模态 shape 不同时使用）
```

**关键设计细节** [论文附录 Sec.B]：
- `W'_hm` 对每个模态独立（"separated input value projection matrices for each modality"）
  → 这使得即使一个传感器故障，另一个仍能正常工作
- 注意力权重 A_hmqk 在模态和采样点间做 Softmax 归一化：`Σ_m Σ_k A_hmqk = 1`

---

### Slide 5.3：论文 — 复杂度分析与 Sparse Aggregation

**[论文] 复杂度分析**：

```
普通 Cross Attention: O(N²)，其中 N = X×Y（BEV 网格数）
  → 若感知范围 R = X/2 = Y/2，则复杂度为 O(16R⁴) — 不可扩展

MDCA (Deformable): O(2N + NK)，K = H×M×K_sampling
  → 线性复杂度，可扩展至远距离感知
```

**[论文] Sparse Aggregation**（论文 Sec.3.3 结尾）：

进一步减少 query 数量：只对 top-K 置信度的 BEV 位置做融合

```
选择概率 = max(D_I, O_P)   ← 取深度分布和雷达占用率的较大值
选 top-N_k 个 query 送入 MDCA

论文消融结果（附录 Tab.3）: N_k=4096 时 FPS 从 11.5→14.0，AP 从 56.9→54.0
```

**[路侧适配]**: 当前代码未使用 Sparse Aggregation（`fuser_conf` 中无相关参数）。

---

### Slide 5.4：代码 — MFAFuser 实现细节

**[代码]** `multimodal_feature_aggregation.py` 第 145-216 行：

```
输入:
  feat_img: [B, 80, 128, 480]  ← 图像 BEV 特征（来自 RVTLSSFPN）
  feat_pts: [B, 80, 128, 480]  ← 雷达 BEV 特征（来自 PtsBackbone）

Step 1: 预处理
  feat_img → LayerNorm → [B, 80, 128, 480]
  feat_pts → LayerNorm → [B, 80, 128, 480]

Step 2: 构建 query
  z_q = Linear(Concat(feat_img, feat_pts))  → [B, 128, 128, 480]
  论文公式: z_q = W_z[LN(C_I); LN(C_P)], W_z ∈ R^{C×2C}

Step 3: 位置编码
  LearnedPositionalEncoding(bev_mask) → [B, 128*480, 128]
  是可学习的 embedding，不是固定的正弦编码

Step 4: 6 层 MDCA + FFN
  for each layer:
    z_q = MDCA(z_q, p_q, {feat_img, feat_pts})  ← 可变形交叉注意力
    z_q = LayerNorm(z_q)
    z_q = FFN(z_q)  ← 两层 MLP: 128→256→128
    z_q = LayerNorm(z_q)

Step 5: 输出
  Reshape → [B, 128, 128, 480]
  ReduceConv (128→128) → [B, 128, 128, 480]  ← 融合 BEV
```

**[论文 vs 代码] MFA 配置差异：**

| 参数 | 论文 [附录 C] | 代码 [CRN_r50_...4key.py] |
|------|-------------|--------------------------|
| 注意力头数 H | 8 | **4** |
| MFA 层数 | 6 | 6 |
| 采样点数 K | 4 | 4 |
| Embedding 维度 | 256 | **128** |
| 输入维度 (img/pts) | 80/80 | 80/80 |

---

## PPT 第6部分：模块③ — 检测头与训练目标 [论文 Sec.3.4]

---

### Slide 6.1：论文 — CenterPoint 检测范式

**[论文] Sec.3.4**：

"we follow CenterPoint to predict the center heatmap with anchor-free and multi-group head."

基于 CenterPoint 的 anchor-free 检测：
- 不预设 anchor，直接预测目标中心点
- 分类：热力图（heatmap），每个类别一个通道
- 回归：10 维向量（中心偏移×2 + 高度×1 + 尺寸×3 + 朝向×2 + 速度×2）

---

### Slide 6.2：代码 — BEVDepthHead 网络结构

**[代码]** `bev_depth_head_det.py` 第 33-133 行：

```
融合 BEV [128, 128, 480]
  │
  ▼ BEV Backbone: ResNet18, 3 stages
  │   stage1: 128→160, stride=1  [160, 128, 480]
  │   stage2: 160→320, stride=2  [320,  64, 240]
  │   stage3: 320→640, stride=2  [640,  32, 120]
  │
  ▼ BEV Neck: SECONDFPN
  │   上采样 ×1, ×2, ×4, ×8 → 4 层特征 [64, 64, 64, 64]
  │   Concat → [256, 128, 480]
  │
  ▼ Shared Conv: Conv3×3(256→256)
  │
  ├→ Task 1 (car):                    heatmap[1] + reg[10]
  ├→ Task 2 (truck, construction):    heatmap[2] + reg[10]
  ├→ Task 3 (bus, trailer):           heatmap[2] + reg[10]
  ├→ Task 4 (barrier):                heatmap[1] + reg[10]
  ├→ Task 5 (motorcycle, bicycle):    heatmap[2] + reg[10]
  └→ Task 6 (pedestrian, traffic_cone): heatmap[2] + reg[10]

每个 Task Head = SeparateHead (from CenterPoint):
  heatmap_head: Conv1×1(N_class)
  reg_head:     Conv1×1(2)    中心偏移 (Δx, Δy)
  height_head:  Conv1×1(1)    高度 z
  dim_head:     Conv1×1(3)    尺寸 (w, l, h)  ← 注意: 代码用 (w,l,h) 顺序
  rot_head:     Conv1×1(2)    朝向 (sin, cos)
  vel_head:     Conv1×1(2)    速度 (vx, vy)
```

---

### Slide 6.3：高斯热力图目标生成

**[代码]** `bev_depth_head_det.py` `get_targets_single()` 第 135-276 行：

```
Step 1: 将 GT Box 中心转换到热力图网格坐标
  coor_x = (x - pc_range[0]) / voxel_size[0] / out_size_factor
  coor_y = (y - pc_range[1]) / voxel_size[1] / out_size_factor

Step 2: 自适应高斯半径
  radius = gaussian_radius((length/grid_scale, width/grid_scale),
                           min_overlap=0.1)    ← [论文] gaussian_overlap=0.1
  radius = max(min_radius=2, radius)

Step 3: 在对应类别通道上绘制高斯圆
  draw_heatmap_gaussian(heatmap[class_id], center_int, radius)
  如果多个同类目标重叠 → 逐元素取 max

Step 4: 回归目标
  reg_target = [
      center - grid_center,  # Δx, Δy (亚像素偏移)
      z,                      # 目标底部高度
      w, l, h,                # 尺寸 (width, length, height)
      sin(yaw), cos(yaw),     # 朝向角
      vx, vy                  # 速度
  ]
```

---

### Slide 6.4：损失函数（代码实际配置）

**[论文]** Sec.3.4：使用 LiDAR 点投影到图像得到深度图来训练深度分布网络（同 BEVDepth）。

**[代码]** `CRN_r50_256x704_128x128_4key.py` `training_step()` 第 283-317 行：

```
总损失: L_total = L_heatmap + L_bbox + L_depth

① Heatmap Loss: GaussianFocalLoss
  FL(p_t) = -α_t · (1-p_t)^γ · log(p_t)   (γ=2.0)
  对正样本的 loss 按高斯权重衰减，聚焦于困难负样本

② BBox Regression Loss: SmoothL1Loss (β=0.05)
  |x| < β 时用 L2（平滑），|x| ≥ β 时用 L1
  加权矩阵 code_weights:

  参数:  Δx   Δy    z    w    l    h   sin  cos   vx   vy
  权重:  4.0  2.0  0.5  1.5  1.5  0.5  3.0  3.0  0.2  0.2
         ↑                            ↑              ↑
     Δx(纵向)最高               朝向次之         速度最低
     → 优先保证纵向定位精度     → 方向对路侧重要   → 雷达速度有噪声

③ Depth Loss: Binary Cross Entropy
  论文: weight=3.0（LiDAR 深度 GT）
  路侧: weight=0.5（毫米波雷达深度 GT 更稀疏、噪声更大，降低权重避免过拟合）
  只计算前景像素（有雷达深度标注的位置）
  下采样: 16×16 block → 1 值（取最小深度 → 保留最近障碍物信号）

④ 深度 GT 下采样策略 [代码] base_exp.py:320-354
  将 2160×3840 原图 → 经 16× downsample → 135×240 深度图
  每个 16×16 block 取非零深度的最小值
  → One-hot 编码（soft label on depth bins）
```

---

## PPT 第7部分：时序融合与推理 [论文附录]

---

### Slide 7.1：论文 — 时序 BEV 特征拼接

**[论文附录 Sec.C]**：

"fused BEV feature maps from the previous T timestamps are aligned to the current timestamp and concatenated."

- T=3 用于正式提交，T=1 用于消融实验
- 历史帧的 BEV 特征使用 `torch.no_grad()` 计算（节省显存）
- 时序帧以 1 秒间隔取得比 0.5 秒间隔更好的性能
- 论文消融 [附录 Tab.2]：1→2→3→4 帧持续提升（4 帧达到饱和）

```
T=1: NDS=50.3, mAP=42.9
T=2: NDS=54.5, mAP=46.0  (+4.2/+3.1)
T=3: NDS=55.7, mAP=47.3  (+1.2/+1.3)
T=4: NDS=56.0, mAP=48.1  (+0.3/+0.8) ← 趋于饱和
```

**[路侧适配]**: 当前使用 T=1（单帧）。时序融合是论文已验证的有效提升手段，也是你评估报告中提到的后续优化方向。

---

### Slide 7.2：推理后处理流程

```
模型输出 → [B, N_class, 128, 480] heatmap + [B, 10, 128, 480] regression

Step 1: Sigmoid(heatmap) → 找到 top-K 峰值
Step 2: 提取峰值位置的回归值
Step 3: 解码为 3D 框
  x_center = grid_x * voxel_size[0] + pc_range[0] + Δx
  y_center = grid_y * voxel_size[1] + pc_range[1] + Δy
  yaw = atan2(sin_yaw, cos_yaw)
  (w, l, h) = dim  (或 exp(dim) 如果 norm_bbox=True)

Step 4: Circle NMS
  基于 BEV 中心距离抑制重复检测
  nms_thr=0.2

Step 5: 置信度过滤
  score_threshold=0.01 → 保留低分框给后续评估
  实际使用时 score_threshold=0.35

Step 6: 坐标变换 (ego → global)
  通过 ego2global 变换矩阵转到世界坐标系
```

---

## PPT 第8部分：路侧场景关键适配总结

---

### Slide 8.1：论文实现 vs 路侧代码 — 差异对照表

| 配置项 | 论文 | 路侧代码 | 原因 |
|--------|------|---------|------|
| 相机数 N | 6 | 1 | 单杆安装 |
| BEV 范围 X | ±51.2m | 0~240m | 路侧单向观察 |
| BEV 范围 Y | ±51.2m | ±25.6m | 道路宽度受限 |
| BEV 分辨率 | 0.8m×0.8m | 0.5m×0.4m | 纵向精度优先 |
| BEV 网格 | 128×128 | 480×128 | 长条形适应道路 |
| 深度范围 | [2.0, 58.0]m | [2.0, 242.0]m | 覆盖更远 |
| 深度 bin | 0.5m, D=112 | 2.0m, D=120 | 适配远距离 |
| MFA 头数 H | 8 | 4 | 节省显存 |
| MFA embed_dim | 256 | 128 | 节省显存 |
| Camera-Aware | True | False | 单相机无需求 |
| 时序帧 T | 3 (提交) | 1 | 简化实现 |
| Rot augmentation | False (论文明确丢弃) | False | 论文一致 |
| BDA rot_lim | ±22.5° | ±5.0° | 路侧车辆朝向集中 |
| IDA resize_lim | — | (0.36, 0.38) | 减少几何扰动 |
| Depth loss weight | 3.0 | 0.5 | 雷达GT噪声大 |
| code_weights[0] (Δx) | 1.0 | 4.0 | 纵向定位是核心指标 |

---

### Slide 8.2：论文 — 数据增强策略

**[论文附录 C]**：

- **IDA** (Perspective View): Resize, crop, horizontal flip。**明确丢弃 rotation** 因为"rotation can have an adverse effect when collapsing the height dimension in RVT"。
- **BDA** (BEV View): Random flip X/Y, global rotation ±22.5°, global scale [0.95, 1.05]。**不使用 GT-AUG**（Ground Truth Sampling）。
- **RDA** (Radar): N_sweeps 中随机选择，随机 dropout。

**[路侧适配]**：BDA rot 缩小到 ±5.0°，保持路侧数据几何真实性。

---

### Slide 8.3：两场景：Frontal vs Oblique

```
Frontal (正视)                         Oblique (斜视)
┌──────────────────────┐             ┌──────────────────────┐
│ 安装高度:              │             │ 安装高度:              │
│   雷达 7m, 相机 6m     │             │   雷达 25m, 相机 18m   │
│ Yaw偏角: 0°           │             │ Yaw偏角: 15°          │
│                        │             │                        │
│   ┌──┐                 │             │          ┌──┐          │
│   │📷│  ← 正对道路       │             │          │📷│ ← 俯视+侧视│
│   │📡│                 │             │          │📡│          │
│   └──┘                 │             │          └──┘          │
│    │                   │             │         /             │
│    ▼                   │             │        / 15°          │
│ ═══════════ 道路 →      │             │ ═══════════ 道路 →     │
│                        │             │                        │
│ 优势: 正视无畸变        │             │ 优势: 高视角少遮挡      │
│ 劣势: 远距离目标小       │             │ 劣势: 透视变形更大      │
└──────────────────────┘             └──────────────────────┘
```

---

## PPT 第9部分：CRN 创新点总结

---

### Slide 9.1：论文四大创新

```
创新 1: Radar-assisted View Transformation (RVT)
  ─────────────────────────────────────────────
  论文公式: C_I^{FV} = Conv[C_I^{PV} ⊗ D_I; C_I^{PV} ⊗ O_R]
  用雷达物理测量增强深度估计不确定的像素
  核心洞察: "sigmoid instead of softmax" — 雷达占用不是互斥的 one-hot

创新 2: 双流并行 + frustum 空间雷达编码
  ────────────────────────────────
  图像流: 密集但有深度歧义 → C_I^{FV} (frustum)
  雷达流: 稀疏但深度精确   → C_R^{FV} (frustum) + O_R (占用)
  两个流在同一 frustum 空间 — 保证外积操作对齐

创新 3: Multi-modal Deformable Cross Attention (MDCA)
  ────────────────────────────────────────────
  每模态独立的值投影 W'_hm → 传感器故障鲁棒
  可学习采样偏移 Δp → 跨模态空间对齐
  线性复杂度 O(NK) → 可扩展到远距离

创新 4: 平均体素池化 + 高度压缩 + 无旋转增强
  ──────────────────────────────
  Average Pooling (非 Sum) → 消除近远距离特征强度不均
  Height Summation → 省显存 + 雷达无 elevation 合理假设
  No Rotation Aug → RVT 中高度压缩时不产生畸变
```

---

### Slide 9.2：路侧适配的核心决策

```
1. BEV 空间重新设计: 正方形→长条形，分辨率提升
   论文: 128×128 @ 0.8m   →   路侧: 480×128 @ 0.5m×0.4m

2. 损失权重精细化: code_weights[0]=4.0 (纵向)
   论文: 均匀权重          →   路侧: 纵向权重 4× 放大

3. 深度损失降低: weight=0.5 (vs 论文 3.0)
   原因: 毫米波雷达GT 远比 LiDAR GT 稀疏

4. Camera-Aware 关闭
   原因: 单相机固定内参，无跨相机区分需求

5. 时序帧暂不启用 (T=1)
   论文已验证 T=4 最优，路侧未来可扩展
```

---

## PPT 附录 A：关键公式速查卡

| 公式 | 表达式 | 出处 |
|------|--------|------|
| LSS 范式 | F_3D = M(F_2D ⊗ D) | [论文 Eq.1] |
| 深度分布 | D_I(u,v) = Softmax(Conv(F_I)(u,v)) | [论文 Eq.2] |
| 雷达占用率 | O_R(d,u) = σ(Conv(F_R)(d,u)) | [论文 Eq.5] |
| RVT 核心 | C_I^{FV} = Conv[C_I^{PV}⊗D_I ; C_I^{PV}⊗O_R] | [论文 Eq.3] |
| BEV 变换 | F^{BEV} = M({F_i^{FV}}_{i=1}^N) | [论文 Eq.4] |
| MDCA | Σ_h W_h[Σ_m Σ_k A_hmqk·W'_hm·x_m(φ_m(p_q+Δp_hmqk))] | [论文 Eq.6] |
| 总损失 | L = L_focal + λ_bbox·L_smoothL1 + λ_depth·L_BCE | [代码] |

---

## PPT 附录 B：代码-论文术语对照表

| 论文符号 | 含义 | 代码变量 | 代码位置 |
|---------|------|---------|---------|
| F_I | 图像特征 | `img_feats` | `rvt_lss_fpn.py:278` |
| C_I^{PV} | 图像上下文特征 | `image_feature` | `rvt_lss_fpn.py:297` |
| D_I | 深度分布 | `depth_occupancy` | `rvt_lss_fpn.py:299` |
| F_R | 雷达特征 | `x` (pts backbone 中) | `pts_backbone.py:155` |
| C_R^{FV} | 雷达上下文特征 | `x_context` | `pts_backbone.py:167` |
| O_R | 雷达占用率 | `x_occupancy` | `pts_backbone.py:169` |
| C_I^{FV} | 图像 frustum 特征 | `img_context` | `rvt_lss_fpn.py:327` |
| M | 视图变换模块 | `average_voxel_pooling` | `rvt_lss_fpn.py:341` |
| F^{BEV} | 融合 BEV 特征 | `fused` | `camera_radar_net_det.py:95` |
| z_q | MFA query | `bev_queries` | `multimodal_feature_aggregation.py:174` |
| p_q | 参考点 | `ref_2d` | `multimodal_feature_aggregation.py:44` |
| x_m | 多模态特征 | `feat_img, feat_pts` | `multimodal_feature_aggregation.py:145` |

---

> **文档用途**: PPT 逐页参考素材，每节对应 1-3 张幻灯片
> **数据来源**: ICCV 2023 论文 LaTeX 源文件 + 项目实际代码 (commit a3118bd)
> **更新时间**: 2026-06-23
