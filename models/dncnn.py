"""
DnCNN — Denoising CNN adapted for dehazing.
Ref: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for
     Image Denoising (TIP 2017)

Uses residual learning: output = input - noise_residual
For dehazing: haze = clean + haze_residual → clean = hazy - residual
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """DnCNN for dehazing with residual learning.

    Architecture:
        Conv(3, 64) + ReLU                    ← input layer
        6 × [Conv(64,64)+BN+ReLU]             ← hidden blocks
        Conv(64, 3)                           ← output layer
        output = input - residual              ← residual learning

    The network predicts the haze residual, then subtracts it from input.
    """

    def __init__(self, in_channels: int = 3, depth: int = 8, mid_channels: int = 64):
        """
        Args:
            in_channels:  input channels (3 for RGB)
            depth:        total conv layers (default 8)
            mid_channels: feature channels (default 64)
        """
        super(DnCNN, self).__init__()
        layers = []
        # First layer: no BN
        layers.append(nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers: Conv + BN + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(mid_channels))
            layers.append(nn.ReLU(inplace=True))

        # Output layer
        layers.append(nn.Conv2d(mid_channels, in_channels, 3, padding=1, bias=False))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.net(x)
        out = x - residual  # residual learning: clean = hazy - haze_residual
        return out


if __name__ == "__main__":
    model = DnCNN(depth=8, mid_channels=64)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DnCNN params: {n:,} (~{n/1e6:.3f}M)")
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
