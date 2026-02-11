"""
MABAANUNet — Morphology-Adaptive Boundary-Aware Attention Network.

Full U-Net architecture with ResNet encoder, MABAAN decoder with attention,
and dual prediction heads (mask + boundary).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from .decoder import MABAANDecoder


class MABAANUNet(nn.Module):
    """MABAAN U-Net with dual heads for mask and boundary prediction."""

    def __init__(self, encoder_name="resnet34", in_channels=4, classes=1,
                 encoder_weights="imagenet", reduction=16):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights
        )
        enc_channels = self.encoder.out_channels
        self.decoder = MABAANDecoder(enc_channels, reduction)
        self.mask_head = nn.Conv2d(32, classes, 1)
        self.boundary_head = nn.Conv2d(32, classes, 1)

    def forward(self, x):
        features = self.encoder(x)
        dec_out = self.decoder(features)
        dec_up = F.interpolate(dec_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        mask = torch.sigmoid(self.mask_head(dec_up))
        boundary = torch.sigmoid(self.boundary_head(dec_up))
        return {'mask': mask, 'boundary': boundary, 'logits': self.mask_head(dec_up)}
