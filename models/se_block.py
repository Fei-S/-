"""
Squeeze-and-Excitation (SE) Attention Block
论文: Squeeze-and-Excitation Networks (CVPR 2018)

用于增强通道特征表达，提升雾区域感知能力。
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    SE Attention Block

    结构:
        Feature → Global Average Pooling → FC → ReLU → FC → Sigmoid → Channel Reweight
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: 输入特征通道数
            reduction: 压缩比例，默认16
        """
        super(SEBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction

        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling → (B, C, 1, 1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图 (B, C, H, W)

        Returns:
            加权后的特征图 (B, C, H, W)
        """
        b, c, _, _ = x.size()

        # Squeeze: Global Average Pooling
        y = self.gap(x).view(b, c)  # (B, C)

        # Excitation: FC → ReLU → FC → Sigmoid
        y = self.fc(y).view(b, c, 1, 1)  # (B, C, 1, 1)

        # Channel Reweight
        return x * y
