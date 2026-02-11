"""
Image preprocessing functions for MABAAN.

Provides normalization, denoising (bilateral filter), contrast enhancement (CLAHE),
and a combined preprocessing pipeline.
"""

import numpy as np
import cv2


def normalize_image(image):
    """Normalize image to zero mean and unit variance."""
    image = image.astype(np.float32)
    return (image - image.mean()) / (image.std() + 1e-8)


def reduce_noise(image, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter for edge-preserving noise reduction."""
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def enhance_contrast(image, clip_limit=2.0, tile_size=8):
    """Apply CLAHE for adaptive contrast enhancement."""
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)


def preprocess_image(image, apply_clahe=True):
    """Full preprocessing pipeline: grayscale -> denoise -> CLAHE -> normalize."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    denoised = reduce_noise(image)
    enhanced = enhance_contrast(denoised) if apply_clahe else denoised
    return normalize_image(enhanced)
