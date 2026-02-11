"""
Evaluation functions for MABAAN.

Includes inference runner, model evaluation with all metrics,
and morphology-stratified evaluation.
"""

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .metrics import compute_all_metrics


@torch.no_grad()
def run_inference(model, dl, device):
    """Run inference on a dataloader, returning predictions, targets, and complexities."""
    model.eval()
    preds, tgts, complexities = [], [], []
    for batch in tqdm(dl, desc="Inference"):
        out = model(batch['input'].to(device))
        preds.extend(out['mask'].cpu().numpy())
        tgts.extend(batch['mask'].numpy())
        complexities.extend(batch['complexity'].numpy())
    return preds, tgts, complexities


def evaluate_model(preds, tgts, model_name="Model"):
    """Evaluate model predictions with all metrics, printing summary statistics."""
    all_metrics = []
    for p, t in tqdm(zip(preds, tgts), total=len(preds), desc=f"Evaluating {model_name}"):
        all_metrics.append(compute_all_metrics(p[0], t[0]))
    df = pd.DataFrame(all_metrics)
    hd_valid = df['hausdorff'].replace([np.inf, -np.inf], np.nan).dropna()
    hd95_valid = df['hausdorff_95'].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\n{'=' * 60}")
    print(f"{model_name} EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Dice:         {df['dice'].mean():.4f} +/- {df['dice'].std():.4f}")
    print(f"  IoU:          {df['iou'].mean():.4f} +/- {df['iou'].std():.4f}")
    print(f"  Boundary F1:  {df['boundary_f1'].mean():.4f} +/- {df['boundary_f1'].std():.4f}")
    if len(hd_valid) > 0:
        print(f"  HD:           {hd_valid.mean():.2f} +/- {hd_valid.std():.2f}")
    if len(hd95_valid) > 0:
        print(f"  HD95:         {hd95_valid.mean():.2f} +/- {hd95_valid.std():.2f}")
    print(f"{'=' * 60}")
    return df


def morphology_stratified_evaluation(preds, tgts, complexities, model_name="Model"):
    """Evaluate model predictions stratified by morphology complexity bins.

    Bins:
        - Circular (c < 0.2)
        - Moderate (0.2 <= c <= 0.5)
        - Irregular (c > 0.5)
    """
    bins = {'Circular (c<0.2)': [], 'Moderate (0.2<=c<=0.5)': [], 'Irregular (c>0.5)': []}
    bin_keys = list(bins.keys())
    for p, t, c in tqdm(zip(preds, tgts, complexities), total=len(preds), desc="Stratified eval"):
        metrics = compute_all_metrics(p[0], t[0])
        if c < 0.2:
            bins[bin_keys[0]].append(metrics)
        elif c <= 0.5:
            bins[bin_keys[1]].append(metrics)
        else:
            bins[bin_keys[2]].append(metrics)
    print(f"\n{'=' * 70}")
    print(f"{model_name} -- Morphology-Stratified Evaluation")
    print(f"{'=' * 70}")
    print(f"{'Bin':<30} {'N':>5} {'Dice':>14} {'IoU':>14} {'BF1':>14} {'HD95':>14}")
    print("-" * 70)
    results = {}
    for bin_name, metrics_list in bins.items():
        if len(metrics_list) == 0:
            print(f"{bin_name:<30} {'0':>5} {'N/A':>14} {'N/A':>14} {'N/A':>14} {'N/A':>14}")
            continue
        df = pd.DataFrame(metrics_list)
        hd95 = df['hausdorff_95'].replace([np.inf, -np.inf], np.nan).dropna()
        hd95_str = f"{hd95.mean():.2f}+/-{hd95.std():.2f}" if len(hd95) > 0 else "N/A"
        print(
            f"{bin_name:<30} {len(df):>5} "
            f"{df['dice'].mean():.4f}+/-{df['dice'].std():.4f} "
            f"{df['iou'].mean():.4f}+/-{df['iou'].std():.4f} "
            f"{df['boundary_f1'].mean():.4f}+/-{df['boundary_f1'].std():.4f} "
            f"{hd95_str:>14}"
        )
        results[bin_name] = df
    print(f"{'=' * 70}")
    return results
