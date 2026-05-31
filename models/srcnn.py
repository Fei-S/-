"""
SRCNN — Super-Resolution CNN adapted for dehazing.
Ref: Image Super-Resolution Using Deep Convolutional Networks (ECCV 2014)

Adapted as a lightweight image-to-image baseline for dehazing.
Direct hazy→clean mapping (no physics model).
"""

import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """SRCNN for dehazing: 3 conv layers.

    Architecture:
        Conv(3, 64, 9) + ReLU       ← feature extraction (large kernel)
        Conv(64, 32, 1) + ReLU      ← non-linear mapping (1x1)
        Conv(32, 3, 5)              ← reconstruction

    Direct mapping: hazy → clean (no intermediate K estimation).
    """

    def __init__(self, in_channels: int = 3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, in_channels, kernel_size=5, padding=2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x


if __name__ == "__main__":
    model = SRCNN()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SRCNN params: {n:,} (~{n/1e6:.3f}M)")
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
