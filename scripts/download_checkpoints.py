#!/usr/bin/env python3
"""
Download checkpoints from HuggingFace Hub.

Downloads pre-trained model checkpoints for all categories and models.
Checkpoints are stored on HuggingFace Hub to keep the Git repository lightweight.

Usage:
    python scripts/download_checkpoints.py              # Download all
    python scripts/download_checkpoints.py --model patchcore  # Specific model
    python scripts/download_checkpoints.py --category bottle  # Specific category
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import hf_hub_download, list_repo_files, HfApi
from tqdm import tqdm

from core.config import MVTEC_CATEGORIES, DIR_RESULTS, get_available_models


# =============================================================================
# CONFIGURATION
# =============================================================================

# TODO: Replace with your HuggingFace Hub repository
HF_REPO_ID = "YOUR_USERNAME/mvtec-anomaly-checkpoints"

# Mapping from model name to result directory name
MODEL_TO_DIRNAME = {
    "patchcore": "Patchcore",
    "efficientad": "EfficientAd",
    "fastflow": "Fastflow",
    "stfpm": "Stfpm",
    "padim": "Padim",
}


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def get_checkpoint_hf_path(model_name: str, category: str) -> str:
    """
    Returns the path of the checkpoint file in the HF repository.
    
    Args:
        model_name: Name of the model
        category: MVTec category
        
    Returns:
        Path string relative to HF repo root
    """
    dirname = MODEL_TO_DIRNAME.get(model_name, model_name.capitalize())
    return f"{dirname}/MVTecAD/{category}/latest/weights/lightning/model.ckpt"


def get_local_checkpoint_path(model_name: str, category: str) -> Path:
    """
    Returns the local path where the checkpoint should be stored.
    
    Args:
        model_name: Name of the model
        category: MVTec category
        
    Returns:
        Path object for local checkpoint
    """
    dirname = MODEL_TO_DIRNAME.get(model_name, model_name.capitalize())
    return DIR_RESULTS / dirname / "MVTecAD" / category / "latest" / "weights" / "lightning" / "model.ckpt"


def download_checkpoint(model_name: str, category: str, force: bool = False) -> bool:
    """
    Downloads a single checkpoint from HuggingFace Hub.
    
    Args:
        model_name: Name of the model
        category: MVTec category
        force: If True, re-download even if exists
        
    Returns:
        True if downloaded/exists, False if failed
    """
    local_path = get_local_checkpoint_path(model_name, category)
    
    # Skip if already exists
    if local_path.exists() and not force:
        return True
    
    hf_path = get_checkpoint_hf_path(model_name, category)
    
    try:
        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download from HF Hub
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_path,
            local_dir=DIR_RESULTS,
            local_dir_use_symlinks=False,
        )
        
        return True
        
    except Exception as e:
        print(f"  ⚠ Failed to download {model_name}/{category}: {e}")
        return False


def download_all_checkpoints(
    models: list[str] = None,
    categories: list[str] = None,
    force: bool = False
) -> dict:
    """
    Downloads checkpoints for specified models and categories.
    
    Args:
        models: List of model names (None = all available)
        categories: List of categories (None = all MVTec categories)
        force: If True, re-download even if exists
        
    Returns:
        Dict with download statistics
    """
    if models is None:
        models = get_available_models()
    if categories is None:
        categories = MVTEC_CATEGORIES
    
    stats = {"downloaded": 0, "existed": 0, "failed": 0}
    
    total = len(models) * len(categories)
    
    print(f"📦 Downloading checkpoints from: {HF_REPO_ID}")
    print(f"   Models: {', '.join(models)}")
    print(f"   Categories: {len(categories)} total")
    print()
    
    with tqdm(total=total, desc="Downloading") as pbar:
        for model in models:
            for category in categories:
                local_path = get_local_checkpoint_path(model, category)
                
                if local_path.exists() and not force:
                    stats["existed"] += 1
                elif download_checkpoint(model, category, force):
                    stats["downloaded"] += 1
                else:
                    stats["failed"] += 1
                    
                pbar.update(1)
    
    return stats


def check_checkpoint_exists(model_name: str, category: str) -> bool:
    """
    Checks if a checkpoint exists locally.
    
    Args:
        model_name: Name of the model
        category: MVTec category
        
    Returns:
        True if checkpoint exists locally
    """
    return get_local_checkpoint_path(model_name, category).exists()


def ensure_checkpoint(model_name: str, category: str) -> Path:
    """
    Ensures a checkpoint exists, downloading if necessary.
    
    This is the main function to call from inference/app code.
    
    Args:
        model_name: Name of the model
        category: MVTec category
        
    Returns:
        Path to the checkpoint
        
    Raises:
        FileNotFoundError: If checkpoint cannot be found or downloaded
    """
    local_path = get_local_checkpoint_path(model_name, category)
    
    if local_path.exists():
        return local_path
    
    print(f"⬇ Checkpoint not found locally. Downloading {model_name}/{category}...")
    
    if download_checkpoint(model_name, category):
        if local_path.exists():
            print(f"✓ Downloaded successfully")
            return local_path
    
    raise FileNotFoundError(
        f"Checkpoint not found: {local_path}\n"
        f"Please train the model first with: python train.py --model {model_name} --category {category}\n"
        f"Or download from HuggingFace Hub: {HF_REPO_ID}"
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download checkpoints from HuggingFace Hub"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model to download (default: all)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="Category to download (default: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available checkpoints on HF Hub"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine models and categories
    models = get_available_models() if args.model == "all" else [args.model]
    categories = MVTEC_CATEGORIES if args.category == "all" else [args.category]
    
    # Download
    stats = download_all_checkpoints(models, categories, args.force)
    
    # Report
    print()
    print("=" * 50)
    print(f"✓ Downloaded: {stats['downloaded']}")
    print(f"○ Already existed: {stats['existed']}")
    if stats['failed'] > 0:
        print(f"✗ Failed: {stats['failed']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
