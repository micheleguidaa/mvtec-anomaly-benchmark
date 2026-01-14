"""
Model loading and utility functions for mvtec-anomaly-benchmark.

Provides functions for dynamic model import, checkpoint handling,
and model size calculation.
"""

import importlib
from pathlib import Path

from core.config import DIR_RESULTS, load_model_config


# =============================================================================
# MODEL LOADING
# =============================================================================

def get_class_from_path(class_path: str):
    """
    Imports a class from a module path string.
    
    Args:
        class_path: Full path like 'anomalib.models.Patchcore'
    
    Returns:
        The imported class
    """
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_model(model_name: str):
    """
    Loads the model with the configuration from YAML.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Instantiated model
    """
    config = load_model_config(model_name)
    model_class = get_class_from_path(config["class_path"])
    model_params = config["init_args"]
    return model_class(**model_params)


def get_checkpoint_path(category: str, model_name: str) -> Path:
    """
    Returns the checkpoint path for a category and model.
    
    Args:
        category: MVTec category name
        model_name: Name of the model
    
    Returns:
        Path to the checkpoint file
    """
    config = load_model_config(model_name)
    result_dirname = config["result_dirname"]
    return DIR_RESULTS / result_dirname / "MVTecAD" / category / "latest" / "weights" / "lightning" / "model.ckpt"


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def get_model_size_mb(model) -> float:
    """
    Calculates model size in MB.
    
    Args:
        model: PyTorch model
    
    Returns:
        Size in megabytes
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024
