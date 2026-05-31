"""
AOD-Net: All-in-One Dehazing Network
Ref: AOD-Net: All-in-One Dehazing Network (ICCV 2017)

Core formula: J(x) = K(x) * I(x) - K(x) + 1

Improved version:
    - Increased mid_channels (32) for better capacity
    - BatchNorm for training stability
    - No output clamping (preserves gradients)
"""

import torch
import torch.nn as nn


class AODNet(nn.Module):
    """AOD-Net with improved capacity.

    Architecture:
        Input (3 ch)
          -> Conv(3, mid) + BN + ReLU
          -> Conv(mid, mid) + BN + ReLU
          -> Conv(mid, mid) + BN + ReLU
          -> Conv(mid, mid) + BN + ReLU
          -> Conv(mid, mid) + BN + ReLU
          -> Conv(mid, mid) + BN + ReLU
          -> Conv(mid, 3)
          -> K(x)
          -> J = K*I - K + 1
    """

    def __init__(self, in_channels: int = 3, mid_channels: int = 32):
        super(AODNet, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        # Encoder — K(x) estimation
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(mid_channels)

        self.conv4 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(mid_channels)

        self.conv5 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(mid_channels)

        self.conv6 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(mid_channels)

        # Output layer
        self.conv_out = nn.Conv2d(mid_channels, in_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.relu(self.bn1(self.conv1(x)))
        k = torch.relu(self.bn2(self.conv2(k)))
        k = torch.relu(self.bn3(self.conv3(k)))
        k = torch.relu(self.bn4(self.conv4(k)))
        k = torch.relu(self.bn5(self.conv5(k)))
        k = torch.relu(self.bn6(self.conv6(k)))
        k = self.conv_out(k)  # (B, 3, H, W)

        # J = K*I - K + 1  (no clamp — keeps gradients flowing)
        out = k * x - k + 1.0
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AODNet(mid_channels=32)
    print(f"AOD-Net params: {count_parameters(model):,} (~{count_parameters(model)/1e6:.3f}M)")
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape} -> Output: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
