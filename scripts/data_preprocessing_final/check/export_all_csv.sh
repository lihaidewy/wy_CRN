#!/bin/bash
# CRN 跟踪 CSV 一键导出脚本
# ===========================
# 同时导出场景一和场景二在训练集/测试集上的全部 CSV。
# 用法:
#   bash scripts/data_preprocessing_final/check/export_all_csv.sh
#
# 输出目录结构:
#   work_dirs/export_frontal_v1_480/泛化/     (Frontal 测试集)
#   work_dirs/export_frontal_v1_480/非泛化/   (Frontal 训练集)
#   work_dirs/export_oblique_v0_480/泛化/     (Oblique 测试集)
#   work_dirs/export_oblique_v0_480/非泛化/   (Oblique 训练集)

cd /home/wy666/CRN
export PYTHONPATH=.

echo "=========================================="
echo "  CRN 跟踪 CSV 一键导出"
echo "=========================================="
echo ""

# ------------------------------------------------------------------------------
# 1. 场景一 Frontal 480 version_1 - 测试集（泛化）
# ------------------------------------------------------------------------------
echo ">>> [1/4] Frontal 480 - 测试集（泛化）"
PYTHONPATH=. python scripts/data_preprocessing_final/check/export_tracking_csv.py \
    --ckpt outputs/det/CRN_frontal_ultimate_finetune_closed/lightning_logs/version_1/checkpoints/last.ckpt \
    --out_dir work_dirs/export_frontal_v1_480 \
    --scene frontal --resolution 480 --eval_split val
echo ""

# ------------------------------------------------------------------------------
# 2. 场景一 Frontal 480 version_1 - 训练集（非泛化）
# ------------------------------------------------------------------------------
echo ">>> [2/4] Frontal 480 - 训练集（非泛化）"
PYTHONPATH=. python scripts/data_preprocessing_final/check/export_tracking_csv.py \
    --ckpt outputs/det/CRN_frontal_ultimate_finetune_closed/lightning_logs/version_1/checkpoints/last.ckpt \
    --out_dir work_dirs/export_frontal_v1_480 \
    --scene frontal --resolution 480 --eval_split train
echo ""

# ------------------------------------------------------------------------------
# 3. 场景二 Oblique 480 version_0 - 测试集（泛化）
# ------------------------------------------------------------------------------
echo ">>> [3/4] Oblique 480 - 测试集（泛化）"
PYTHONPATH=. python scripts/data_preprocessing_final/check/export_tracking_csv.py \
    --ckpt outputs/det/CRN_oblique_ultimate_finetune_closed/lightning_logs/version_0/checkpoints/last.ckpt \
    --out_dir work_dirs/export_oblique_v0_480 \
    --scene oblique --resolution 480 --eval_split val
echo ""

# ------------------------------------------------------------------------------
# 4. 场景二 Oblique 480 version_0 - 训练集（非泛化）
# ------------------------------------------------------------------------------
echo ">>> [4/4] Oblique 480 - 训练集（非泛化）"
PYTHONPATH=. python scripts/data_preprocessing_final/check/export_tracking_csv.py \
    --ckpt outputs/det/CRN_oblique_ultimate_finetune_closed/lightning_logs/version_0/checkpoints/last.ckpt \
    --out_dir work_dirs/export_oblique_v0_480 \
    --scene oblique --resolution 480 --eval_split train
echo ""

# ------------------------------------------------------------------------------
# 完成
# ------------------------------------------------------------------------------
echo "=========================================="
echo "  全部导出完成！"
echo "=========================================="
echo ""
echo "输出目录："
echo "  work_dirs/export_frontal_v1_480/泛化/"
echo "  work_dirs/export_frontal_v1_480/非泛化/"
echo "  work_dirs/export_oblique_v0_480/泛化/"
echo "  work_dirs/export_oblique_v0_480/非泛化/"
echo ""
