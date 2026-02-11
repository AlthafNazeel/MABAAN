"""
MABAAN — Morphology-Adaptive Boundary-Aware Attention Network.

A deep learning framework for cell segmentation that adapts to cell morphology
using boundary-aware attention mechanisms and morphology-weighted loss.
"""

__version__ = "0.1.0"

from .config import Config
from .models import MABAANUNet
from .utils import MorphologyAwareLoss
