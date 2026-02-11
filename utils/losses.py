"""
Loss functions for MABAAN.

Includes DiceLoss, per-sample DiceLoss, and the MorphologyAwareLoss
with per-sample complexity weighting.
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Standard Dice loss (batch-level)."""

    def forward(self, pred, target, smooth=1e-6):
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        inter = (pred_flat * target_flat).sum()
        return 1 - (2 * inter + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


class DiceLossPerSample(nn.Module):
    """Dice loss computed per-sample for individual weighting."""

    def forward(self, pred, target, smooth=1e-6):
        B = pred.shape[0]
        losses = []
        for i in range(B):
            p = pred[i].reshape(-1)
            t = target[i].reshape(-1)
            inter = (p * t).sum()
            losses.append(1 - (2 * inter + smooth) / (p.sum() + t.sum() + smooth))
        return torch.stack(losses)


class MorphologyAwareLoss(nn.Module):
    """Combined BCE + Dice + Boundary loss with per-sample morphology weighting.

    Args:
        bce_w: Weight for mask BCE loss.
        dice_w: Weight for mask Dice loss.
        boundary_w: Weight for boundary BCE loss.
        complexity_scale: Scale factor for complexity weighting (0 = no weighting).
    """

    def __init__(self, bce_w=0.4, dice_w=0.3, boundary_w=0.3, complexity_scale=0.5):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.dice_ps = DiceLossPerSample()
        self.bce_w, self.dice_w, self.boundary_w = bce_w, dice_w, boundary_w
        self.complexity_scale = complexity_scale

    def forward(self, preds, targets, complexity=None):
        B = preds['mask'].shape[0]
        # Per-sample BCE for mask
        bce_mask = self.bce(preds['mask'], targets['mask']).view(B, -1).mean(dim=1)
        # Per-sample Dice
        dice = self.dice_ps(preds['mask'], targets['mask'])
        # Per-sample boundary BCE
        bce_bnd = self.bce(preds['boundary'], targets['boundary']).view(B, -1).mean(dim=1)
        # Combined per-sample loss
        total_ps = self.bce_w * bce_mask + self.dice_w * dice + self.boundary_w * bce_bnd
        # Per-sample complexity weighting
        if complexity is not None and self.complexity_scale > 0:
            sample_weights = 1.0 + self.complexity_scale * complexity.to(total_ps.device)
            total_ps = total_ps * sample_weights
        total = total_ps.mean()
        return {
            'total': total,
            'bce': bce_mask.mean(),
            'dice': dice.mean(),
            'boundary': bce_bnd.mean(),
        }
