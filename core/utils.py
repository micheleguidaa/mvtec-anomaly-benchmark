"""
General utility functions for mvtec-anomaly-benchmark.

Provides formatting, statistics, and image processing helpers.
"""

import numpy as np
from PIL import Image


# =============================================================================
# FORMATTING
# =============================================================================

def format_metric(value, decimals: int = 4) -> str:
    """
    Formats a metric value for printing.
    
    Args:
        value: Numeric value or None
        decimals: Number of decimal places
    
    Returns:
        Formatted string
    """
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value) if value is not None else "N/A"


# =============================================================================
# STATISTICS
# =============================================================================

def safe_mean(values: list) -> float | None:
    """
    Calculates mean ignoring None values.
    
    Args:
        values: List of numeric values (may contain None)
    
    Returns:
        Mean value or None if no valid values
    """
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def resize_to_match(array: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resizes an array to match the target shape.
    
    Args:
        array: Input numpy array (2D, values 0-1)
        target_shape: Target (height, width) tuple
    
    Returns:
        Resized array
    """
    if array.shape == target_shape:
        return array
    
    scaled = (array * 255).astype(np.uint8)
    pil_img = Image.fromarray(scaled)
    pil_img = pil_img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.array(pil_img) / 255.0


def scale_efficientad_score(score: float) -> float:
    """
    Scales EfficientAD anomaly score to be more interpretable.
    
    Args:
        score: Raw anomaly score
    
    Returns:
        Scaled score (0-1 range, pushed towards extremes)
    """
    if score < 0.5:
        # Good: use power function to push low (e.g. 0.4998 -> ~0.25)
        return (score * 2) ** 2 / 4
    else:
        # Anomaly: steep sigmoid to push high (e.g. 0.5063 -> ~0.96)
        k = 500
        return 1 / (1 + np.exp(-k * (score - 0.5)))
