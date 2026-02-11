"""
Boundary-Aware Attention modules for MABAAN.

Contains Channel Attention, Spatial Attention, and the combined
BoundaryAwareAttentionBlock used in the MABAAN decoder.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention using average and max pooling."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.conv(torch.cat([avg_out, max_out], dim=1))


class BoundaryAwareAttentionBlock(nn.Module):
    """Combined channel + spatial + boundary-aware attention.

    Uses a learnable gamma parameter to control boundary attention strength.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ca = self.channel_att(x)
        sa = self.spatial_att(ca)
        boundary_mask = self.boundary_conv(x)
        out = sa + self.gamma * (sa * boundary_mask)
        return out
