"""
Training utilities for MABAAN.

Includes MetricTracker for logging, train/val epoch functions,
and the main training loop with early stopping and complexity logging.
"""

import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .metrics import compute_dice, compute_iou


class MetricTracker:
    """Tracks and plots training metrics across epochs."""

    def __init__(self):
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': [],
            'avg_complexity': [],
        }

    def update(self, tl, vl, td, vd, ti, vi, avg_c=0.0):
        self.history['train_loss'].append(tl)
        self.history['val_loss'].append(vl)
        self.history['train_dice'].append(td)
        self.history['val_dice'].append(vd)
        self.history['train_iou'].append(ti)
        self.history['val_iou'].append(vi)
        self.history['avg_complexity'].append(avg_c)

    def plot(self):
        """Plot training curves: Loss, Dice, IoU, and Complexity."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        e = range(1, len(self.history['train_loss']) + 1)
        for ax, key, title in [
            (axes[0, 0], 'loss', 'Loss'),
            (axes[0, 1], 'dice', 'Dice'),
            (axes[1, 0], 'iou', 'IoU'),
            (axes[1, 1], 'avg_complexity', 'Avg Complexity'),
        ]:
            if key == 'avg_complexity':
                ax.plot(e, self.history['avg_complexity'], 'g-o', markersize=3)
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Complexity')
            else:
                ax.plot(e, self.history[f'train_{key}'], 'b-o', markersize=3, label='Train')
                ax.plot(e, self.history[f'val_{key}'], 'r-o', markersize=3, label='Val')
                ax.legend()
                ax.set_title(title)
                ax.set_xlabel('Epoch')
        plt.tight_layout()
        plt.show()


def train_epoch(model, dl, criterion, optimizer, device):
    """Run one training epoch, returning loss, dice, iou, and avg complexity."""
    model.train()
    total_loss, total_dice, total_iou, total_complexity, n = 0, 0, 0, 0, 0
    for batch in tqdm(dl, desc="Train", leave=False):
        inp = batch['input'].to(device)
        masks = {'mask': batch['mask'].to(device), 'boundary': batch['boundary'].to(device)}
        complexity = batch['complexity'].to(device)
        out = model(inp)
        losses = criterion(out, masks, complexity)
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        total_loss += losses['total'].item() * inp.size(0)
        total_dice += compute_dice(out['mask'].detach(), masks['mask']) * inp.size(0)
        total_iou += compute_iou(out['mask'].detach(), masks['mask']) * inp.size(0)
        total_complexity += complexity.mean().item() * inp.size(0)
        n += inp.size(0)
    return total_loss / n, total_dice / n, total_iou / n, total_complexity / n


def val_epoch(model, dl, criterion, device):
    """Run one validation epoch, returning loss, dice, iou, and avg complexity."""
    model.eval()
    total_loss, total_dice, total_iou, total_complexity, n = 0, 0, 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="Val", leave=False):
            inp = batch['input'].to(device)
            masks = {'mask': batch['mask'].to(device), 'boundary': batch['boundary'].to(device)}
            complexity = batch['complexity'].to(device)
            out = model(inp)
            losses = criterion(out, masks, complexity)
            total_loss += losses['total'].item() * inp.size(0)
            total_dice += compute_dice(out['mask'], masks['mask']) * inp.size(0)
            total_iou += compute_iou(out['mask'], masks['mask']) * inp.size(0)
            total_complexity += complexity.mean().item() * inp.size(0)
            n += inp.size(0)
    return total_loss / n, total_dice / n, total_iou / n, total_complexity / n


def train_model(model, dataloaders, criterion, optimizer, scheduler, device,
                num_epochs=30, patience=10, save_path='best_model.pth'):
    """Full training loop with early stopping and DataParallel-aware saving."""
    tracker = MetricTracker()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        tl, td, ti, tc = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        vl, vd, vi, vc = val_epoch(model, dataloaders['val'], criterion, device)
        scheduler.step()
        tracker.update(tl, vl, td, vd, ti, vi, tc)
        print(f"  Train - Loss:{tl:.4f} Dice:{td:.4f} IoU:{ti:.4f} Complexity:{tc:.3f}")
        print(f"  Val   - Loss:{vl:.4f} Dice:{vd:.4f} IoU:{vi:.4f}")
        if vl < best_val_loss:
            best_val_loss = vl
            epochs_no_improve = 0
            # Save without DataParallel wrapper
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"  *** Best model saved (val_loss={vl:.4f}) ***")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
    return tracker
