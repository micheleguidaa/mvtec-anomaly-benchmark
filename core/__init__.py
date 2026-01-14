"""
Core package for mvtec-anomaly-benchmark.

Provides shared utilities, configuration, and model loading functions
used across training, inference, and UI components.
"""

from core.config import (
    MVTEC_CATEGORIES,
    SCRIPT_DIR,
    DIR_DATASET,
    DIR_RESULTS,
    DIR_CONFIGS,
    DIR_OUTPUT,
    get_available_models,
    load_model_config,
)

from core.models import (
    get_class_from_path,
    load_model,
    get_checkpoint_path,
    get_model_size_mb,
)

from core.utils import (
    format_metric,
    safe_mean,
    resize_to_match,
    scale_efficientad_score,
)

__all__ = [
    # Config
    "MVTEC_CATEGORIES",
    "SCRIPT_DIR",
    "DIR_DATASET",
    "DIR_RESULTS",
    "DIR_CONFIGS",
    "DIR_OUTPUT",
    "get_available_models",
    "load_model_config",
    # Models
    "get_class_from_path",
    "load_model",
    "get_checkpoint_path",
    "get_model_size_mb",
    # Utils
    "format_metric",
    "safe_mean",
    "resize_to_match",
    "scale_efficientad_score",
]
