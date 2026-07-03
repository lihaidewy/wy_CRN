"""
将 BEV 可视化图片序列合成为 GIF 动图
=======================================
"""
import os
import glob
from PIL import Image


def make_gif_from_images(img_dir, out_path, fps=5, step=2):
    """
    img_dir: 图片目录 (如 vis_bev/)
    out_path: 输出 gif 路径
    fps: 每秒帧数
    step: 每隔多少帧取一张 (降采样，减小体积)
    """
    pattern = os.path.join(img_dir, "bev_*.jpg")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[!] 找不到图片: {pattern}")
        return

    files = files[::step]
    print(f">>> 生成 GIF: {out_path}")
    print(f"    共 {len(files)} 张图片 (step={step}, fps={fps})")

    images = [Image.open(f) for f in files]
    duration_ms = int(1000 / fps)

    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0
    )
    print(f"    完成: {out_path}")


if __name__ == '__main__':
    BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'work_dirs')

    # Frontal
    make_gif_from_images(
        os.path.join(BASE, 'model_predict_results_frontal_40epoch', 'vis_bev'),
        os.path.join(BASE, 'model_predict_results_frontal_40epoch', 'bev_frontal.gif'),
        fps=5, step=2
    )

    # Oblique
    make_gif_from_images(
        os.path.join(BASE, 'model_predict_results_oblique_240', 'vis_bev'),
        os.path.join(BASE, 'model_predict_results_oblique_240', 'bev_oblique.gif'),
        fps=5, step=2
    )
