"""
Configuration and constants for mvtec-anomaly-benchmark.

Centralizes all paths, categories, and configuration loading.
"""

import os
from pathlib import Path
import yaml


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.parent.absolute()
DIR_DATASET = SCRIPT_DIR / "data" / "MVTecAD"
DIR_RESULTS = SCRIPT_DIR / "results"
DIR_CONFIGS = SCRIPT_DIR / "configs"
DIR_OUTPUT = SCRIPT_DIR / "output"


# =============================================================================
# CATEGORIES
# =============================================================================

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def get_available_models() -> list[str]:
    """
    Returns list of available model names from configs directory.
    
    Returns:
        Sorted list of model names (without .yaml extension)
    """
    models = []
    if DIR_CONFIGS.exists():
        for f in DIR_CONFIGS.iterdir():
            if f.suffix == '.yaml':
                models.append(f.stem)
    return sorted(models)


def load_model_config(model_name: str) -> dict:
    """
    Loads model configuration from YAML.
    
    Args:
        model_name: Name of the model (without .yaml extension)
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = DIR_CONFIGS / f"{model_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
