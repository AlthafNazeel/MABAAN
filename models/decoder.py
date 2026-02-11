"""
MABAAN Decoder with Boundary-Aware Attention.

Custom U-Net decoder that applies BoundaryAwareAttentionBlock
at each decoder stage for enhanced boundary segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import BoundaryAwareAttentionBlock


class MABAANDecoder(nn.Module):
    """U-Net decoder with attention blocks at each stage."""

    def __init__(self, encoder_channels, reduction=16):
        super().__init__()
        dec_channels = [256, 128, 64, 32]
        self.blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        for i, dc in enumerate(dec_channels):
            in_ch = encoder_channels[-(i + 1)] + (
                encoder_channels[-(i + 2)] if i < len(dec_channels) - 1 else encoder_channels[0]
            )
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, dc, 3, padding=1, bias=False),
                nn.BatchNorm2d(dc),
                nn.ReLU(inplace=True),
                nn.Conv2d(dc, dc, 3, padding=1, bias=False),
                nn.BatchNorm2d(dc),
                nn.ReLU(inplace=True),
            ))
            self.attention_blocks.append(BoundaryAwareAttentionBlock(dc, reduction))

    def forward(self, features):
        x = features[-1]
        for i, (block, att) in enumerate(zip(self.blocks, self.attention_blocks)):
            skip = features[-(i + 2)] if i < len(self.blocks) - 1 else features[0]
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
            x = att(x)
        return x
