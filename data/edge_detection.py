"""
Morphology-adaptive edge detection for MABAAN.

Computes shape descriptors (circularity, solidity, complexity) and
multi-scale morphological edge fusion.
"""

import numpy as np
import cv2
from skimage.measure import regionprops, label


def compute_shape_descriptors(mask):
    """Compute shape descriptors for cell morphology analysis.

    Returns dict with circularity, solidity, and complexity (0=circular, 1=irregular).
    """
    if mask.sum() == 0:
        return {'circularity': 1.0, 'solidity': 1.0, 'complexity': 0.0}
    labeled = label(mask > 0)
    props = regionprops(labeled)
    if len(props) == 0:
        return {'circularity': 1.0, 'solidity': 1.0, 'complexity': 0.0}
    circularities, solidities = [], []
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            circularities.append(min(circularity, 1.0))
        if prop.convex_area > 0:
            solidities.append(area / prop.convex_area)
    avg_circ = np.mean(circularities) if circularities else 1.0
    avg_sol = np.mean(solidities) if solidities else 1.0
    complexity = 1.0 - (avg_circ * avg_sol)
    return {'circularity': avg_circ, 'solidity': avg_sol, 'complexity': complexity}


def compute_morphological_gradient(image, kernel_size=3):
    """Compute morphological gradient (dilation - erosion) for edge detection."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if image.dtype in [np.float32, np.float64]:
        img = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    else:
        img = image.astype(np.uint8)
    gradient = cv2.dilate(img, kernel).astype(float) - cv2.erode(img, kernel).astype(float)
    return gradient / (gradient.max() + 1e-8)


def multi_scale_edge_fusion(image, scales=(3, 5, 7)):
    """Fuse edges detected at multiple scales with weighted combination."""
    edges = [compute_morphological_gradient(image, s) for s in scales]
    weights = [0.2, 0.35, 0.45]
    fused = sum(w * e for w, e in zip(weights, edges))
    return fused / (fused.max() + 1e-8)


class MorphologyAdaptiveEdgeDetector:
    """Multi-scale morphological edge detector."""

    def __init__(self, scales=(3, 5, 7)):
        self.scales = scales

    def detect(self, image):
        """Detect edges using multi-scale morphological gradient fusion."""
        return multi_scale_edge_fusion(image, self.scales)
