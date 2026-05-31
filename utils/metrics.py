"""
图像质量评估指标

实现:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index Measure)

可选:
    - LPIPS (Learned Perceptual Image Patch Similarity)
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算 PSNR

    Args:
        img1: 图像1 (H, W, 3)，值域 [0, 255] 或 [0, 1]
        img2: 图像2 (H, W, 3)，值域 [0, 255] 或 [0, 1]

    Returns:
        PSNR 值 (dB)，越高越好
    """
    # 自动检测值域并转换
    if img1.max() <= 1.0:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.max() <= 1.0:
        img2 = (img2 * 255).astype(np.uint8)

    return peak_signal_noise_ratio(img1, img2, data_range=255)


def calculate_ssim(
    img1: np.ndarray, img2: np.ndarray, multichannel: bool = True
) -> float:
    """
    计算 SSIM

    Args:
        img1: 图像1 (H, W, 3)，值域 [0, 255] 或 [0, 1]
        img2: 图像2 (H, W, 3)，值域 [0, 255] 或 [0, 1]
        multichannel: 是否多通道（RGB）

    Returns:
        SSIM 值 [0, 1]，越高越好
    """
    # 自动检测值域并转换
    if img1.max() <= 1.0:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.max() <= 1.0:
        img2 = (img2 * 255).astype(np.uint8)

    if multichannel:
        channel_axis = 2
    else:
        channel_axis = None

    return structural_similarity(
        img1,
        img2,
        data_range=255,
        channel_axis=channel_axis,
        win_size=11,  # 默认窗口大小
    )


def calculate_psnr_torch(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    使用 PyTorch 计算 PSNR（支持 GPU）

    Args:
        img1: (C, H, W) 或 (B, C, H, W)，值域 [0, 1]
        img2: (C, H, W) 或 (B, C, H, W)，值域 [0, 1]

    Returns:
        PSNR 值 (dB)
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    return float(10 * torch.log10(1.0 / mse))


def calculate_ssim_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
) -> float:
    """
    使用 PyTorch 计算 SSIM（简化版，支持 GPU）

    Args:
        img1: (1, C, H, W)，值域 [0, 1]
        img2: (1, C, H, W)，值域 [0, 1]
        window_size: 高斯窗口大小

    Returns:
        SSIM 值 [0, 1]
    """
    # 高斯窗口
    sigma = 1.5
    gauss = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(gauss**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    _1d_window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)  # (window_size, window_size)
    _2d_window = _1d_window.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)
    window = _2d_window.expand(img1.size(1), 1, window_size, window_size).to(img1.device)

    # 常数
    C1 = 0.01**2
    C2 = 0.03**2

    # 均值
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # 方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2

    # SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return float(ssim_map.mean())


def calculate_lpips(
    img1: torch.Tensor,
    img2: torch.Tensor,
    model=None,
    device: str = "cuda",
) -> float:
    """
    计算 LPIPS（可选）

    Args:
        img1: (1, 3, H, W)，值域 [0, 1] 或 [-1, 1]
        img2: (1, 3, H, W)
        model: LPIPS 模型实例，如果为 None 则返回 -1
        device: 计算设备

    Returns:
        LPIPS 值，越低越好
    """
    if model is None:
        return -1.0

    img1 = img1.to(device)
    img2 = img2.to(device)

    # LPIPS 需要 [-1, 1] 范围
    if img1.min() >= 0:
        img1 = img1 * 2 - 1
    if img2.min() >= 0:
        img2 = img2 * 2 - 1

    with torch.no_grad():
        dist = model(img1, img2)

    return float(dist.item())


class AverageMeter:
    """运行平均值统计器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # 测试
    img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)

    print(f"随机图像 — PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

    # 完全相同的图像
    psnr_same = calculate_psnr(img1, img1)
    ssim_same = calculate_ssim(img1, img1)
    print(f"相同图像 — PSNR: {psnr_same:.2f} dB, SSIM: {ssim_same:.4f}")
