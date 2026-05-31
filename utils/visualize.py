"""
可视化模块

功能:
    - 拼接对比图 (Input | DCP | AOD-Net | Ours | GT)
    - 保存单张结果图
    - 生成训练曲线
"""

import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt


def make_comparison_image(
    hazy: np.ndarray,
    dcp_output: np.ndarray | None,
    aod_output: np.ndarray | None,
    ours_output: np.ndarray | None,
    gt: np.ndarray | None,
    save_path: str,
    titles: list[str] | None = None,
) -> None:
    """
    生成对比拼接图

    格式: Input | DCP | AOD-Net | Ours | GT

    Args:
        hazy:        有雾输入图像 (H, W, 3)，值域 [0, 1] 或 [0, 255]
        dcp_output:  DCP 输出 (H, W, 3)，可为 None
        aod_output:  AOD-Net 输出 (H, W, 3)，可为 None
        ours_output: Ours 输出 (H, W, 3)，可为 None
        gt:          真值图像 (H, W, 3)，可为 None
        save_path:   保存路径
        titles:      各子图标题
    """
    # 收集所有有效图像
    images = []
    all_titles = []

    if titles is None:
        titles = ["Input", "DCP", "AOD-Net", "Ours", "GT"]

    candidates = [
        (hazy, titles[0]),
        (dcp_output, titles[1]),
        (aod_output, titles[2]),
        (ours_output, titles[3]),
        (gt, titles[4]),
    ]

    for img, title in candidates:
        if img is not None:
            images.append(img)
            all_titles.append(title)

    n = len(images)
    if n == 0:
        return

    # 创建画布
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, all_titles):
        # 统一值域处理
        if img.max() <= 1.0:
            display_img = np.clip(img, 0, 1)
        else:
            display_img = np.clip(img, 0, 255) / 255.0

        ax.imshow(display_img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"对比图已保存: {save_path}")


def save_image(
    img: np.ndarray, save_path: str, convert_bgr: bool = True
) -> None:
    """
    保存单张图像

    Args:
        img:       图像 (H, W, 3)，值域 [0, 1] 或 [0, 255]
        save_path: 保存路径
        convert_bgr: 是否转换为 BGR（OpenCV 保存需要）
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    if convert_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def plot_training_curves(
    losses: list[float],
    save_path: str,
    title: str = "Training Loss",
) -> None:
    """
    绘制训练损失曲线

    Args:
        losses:    每个 epoch 的损失值
        save_path: 保存路径
        title:     图表标题
    """
    plt.figure(figsize=(10, 5))

    epochs = range(1, len(losses) + 1)

    plt.plot(epochs, losses, "b-", linewidth=1.5, label="L1 Loss")
    plt.fill_between(epochs, 0, losses, alpha=0.1, color="blue")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 标注最低点
    min_idx = np.argmin(losses)
    plt.annotate(
        f"Min: {losses[min_idx]:.4f}",
        xy=(min_idx + 1, losses[min_idx]),
        xytext=(min_idx + 1 + 2, losses[min_idx] * 1.1),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def plot_metrics_comparison(
    metrics: dict[str, dict[str, float]],
    save_path: str,
    title: str = "Metrics Comparison",
) -> None:
    """
    绘制指标对比柱状图

    Args:
        metrics:   {method: {metric_name: value}}
        save_path: 保存路径
        title:     图表标题
    """
    methods = list(metrics.keys())
    metric_names = list(metrics[methods[0]].keys())

    x = np.arange(len(methods))
    width = 0.8 / len(metric_names)

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 4))

    if len(metric_names) == 1:
        axes = [axes]

    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

    for i, metric_name in enumerate(metric_names):
        values = [metrics[m][metric_name] for m in methods]
        bars = axes[i].bar(
            x,
            values,
            width * len(metric_names),
            color=colors[: len(methods)],
            edgecolor="white",
        )

        # 数值标注
        for bar, val in zip(bars, values):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        axes[i].set_title(metric_name, fontsize=12)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(methods, fontsize=10)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"指标对比图已保存: {save_path}")


if __name__ == "__main__":
    # 测试
    hazy = np.random.rand(256, 256, 3)
    gt = np.random.rand(256, 256, 3)
    dcp = np.random.rand(256, 256, 3)

    make_comparison_image(
        hazy=hazy,
        dcp_output=dcp,
        aod_output=None,
        ours_output=None,
        gt=gt,
        save_path="test_comparison.png",
    )
