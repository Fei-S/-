"""
Ours: AOD-Net + SE Attention
SE Block inserted after conv3, then channel-reweighed features flow through conv4-6.

Same improved backbone as AOD-Net (6 conv + BN + mid_channels=32).
"""

import torch
import torch.nn as nn

from .se_block import SEBlock


class AODNetWithSE(nn.Module):
    """AOD-Net + SE Attention (improved backbone).

    Architecture:
        Input (3 ch)
          -> Conv1 + BN + ReLU
          -> Conv2 + BN + ReLU
          -> Conv3 + BN + ReLU
          -> SE Block   <-- channel attention
          -> Conv4 + BN + ReLU
          -> Conv5 + BN + ReLU
          -> Conv6 + BN + ReLU
          -> Conv_out
          -> J = K*I - K + 1
    """

    def __init__(self, in_channels: int = 3, mid_channels: int = 32, se_reduction: int = 16):
        super(AODNetWithSE, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(mid_channels)

        # SE Attention after conv3
        self.se = SEBlock(channels=mid_channels, reduction=se_reduction)

        self.conv4 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(mid_channels)

        self.conv5 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(mid_channels)

        self.conv6 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(mid_channels)

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

        k = self.se(k)  # channel attention

        k = torch.relu(self.bn4(self.conv4(k)))
        k = torch.relu(self.bn5(self.conv5(k)))
        k = torch.relu(self.bn6(self.conv6(k)))
        k = self.conv_out(k)

        out = k * x - k + 1.0
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from .aodnet import AODNet, count_parameters as cp

    aod = AODNet(mid_channels=32)
    ours = AODNetWithSE(mid_channels=32)
    print(f"AOD-Net params:  {cp(aod):,} (~{cp(aod)/1e6:.3f}M)")
    print(f"Ours params:     {cp(ours):,} (~{cp(ours)/1e6:.3f}M)")
    print(f"SE overhead:     {cp(ours) - cp(aod)} params")

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        ya = aod(x)
        yo = ours(x)
    print(f"AOD output range:  [{ya.min():.3f}, {ya.max():.3f}]")
    print(f"Ours output range: [{yo.min():.3f}, {yo.max():.3f}]")
