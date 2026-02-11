"""
PyTorch Dataset for LIVECell with edge maps and shape complexity.

Integrates preprocessing, edge detection, boundary extraction, and
morphology complexity scoring into each sample.
"""

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from skimage.segmentation import find_boundaries

from .preprocessing import preprocess_image
from .edge_detection import compute_shape_descriptors, MorphologyAdaptiveEdgeDetector


class LiveCellDataset(Dataset):
    """PyTorch Dataset that returns 4-channel input (3 image + 1 edge),
    mask, boundary, and complexity score."""

    def __init__(self, loader, img_ids, img_size=256, edge_detector=None, max_samples=None):
        self.loader = loader
        self.img_ids = img_ids[:max_samples] if max_samples else img_ids
        self.img_size = img_size
        self.edge_detector = edge_detector or MorphologyAdaptiveEdgeDetector()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image = self.loader.load_image(img_id)
        if image is None:
            return {
                'input': torch.zeros(4, self.img_size, self.img_size),
                'mask': torch.zeros(1, self.img_size, self.img_size),
                'boundary': torch.zeros(1, self.img_size, self.img_size),
                'complexity': torch.tensor(0.0),
            }
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = self.loader.generate_mask(img_id)
        preprocessed = preprocess_image(image)
        edge_map = self.edge_detector.detect(preprocessed)
        binary_mask = (mask > 0).astype(np.float32)
        boundary = find_boundaries(binary_mask > 0, mode='thick').astype(np.float32)
        shape_info = compute_shape_descriptors(mask)
        complexity = shape_info['complexity']
        # Resize all to target size
        img_r = cv2.resize(preprocessed, (self.img_size, self.img_size))
        edge_r = cv2.resize(edge_map, (self.img_size, self.img_size))
        mask_r = cv2.resize(binary_mask, (self.img_size, self.img_size))
        bnd_r = cv2.resize(boundary, (self.img_size, self.img_size))
        # Stack: 3-ch image + 1-ch edge
        combined = np.concatenate(
            [np.stack([img_r] * 3, axis=0), edge_r[np.newaxis, ...]], axis=0
        ).astype(np.float32)
        return {
            'input': torch.from_numpy(combined),
            'mask': torch.from_numpy(mask_r[np.newaxis, ...].astype(np.float32)),
            'boundary': torch.from_numpy(bnd_r[np.newaxis, ...].astype(np.float32)),
            'complexity': torch.tensor(complexity, dtype=torch.float32),
        }
