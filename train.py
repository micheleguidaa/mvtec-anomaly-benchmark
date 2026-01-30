#!/usr/bin/env python3
"""
Training script for MVTec AD using Anomalib models.

Examples:
    python train.py                     # Train Patchcore on bottle (default)
    python train.py --model all         # Train all models on bottle
    python train.py --category all      # Train Patchcore on all categories
"""

import os
import time
import json
import logging
import argparse
from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.data.utils import download_and_extract

from core import (
    MVTEC_CATEGORIES,
    DIR_RESULTS,
    DIR_DATASET,
    get_available_models,
    load_model_config,
    get_class_from_path,
    get_model_size_mb,
    format_metric,
    safe_mean,
)

logger = logging.getLogger(__name__)

EFFICIENTAD_RESOURCES_DIR = Path(__file__).parent / "efficientad_resources"

def _patched_prepare_pretrained_model(self) -> None:
    """Patched version that uses efficientad_resources/pre_trained/ directory."""
    from anomalib.models.image.efficient_ad.lightning_model import WEIGHTS_DOWNLOAD_INFO
    from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize
    
    pretrained_models_dir = EFFICIENTAD_RESOURCES_DIR / "pre_trained"
    pretrained_models_dir.mkdir(parents=True, exist_ok=True)
    
    weights_dir = pretrained_models_dir / "efficientad_pretrained_weights"
    if not weights_dir.is_dir():
        download_and_extract(pretrained_models_dir, WEIGHTS_DOWNLOAD_INFO)
    
    model_size_str = self.model_size.value if isinstance(self.model_size, EfficientAdModelSize) else self.model_size
    teacher_path = weights_dir / f"pretrained_teacher_{model_size_str}.pth"
    logger.info(f"Load pretrained teacher model from {teacher_path}")
    self.model.teacher.load_state_dict(
        torch.load(teacher_path, map_location=torch.device(self.device), weights_only=True),
    )


def patch_efficientad():
    """Apply monkey-patch to EfficientAd to use custom pretrained weights directory."""
    from anomalib.models import EfficientAd
    EfficientAd.prepare_pretrained_model = _patched_prepare_pretrained_model
    print(f"   [INFO] EfficientAd: Pretrained weights directory: {EFFICIENTAD_RESOURCES_DIR / 'pre_trained'}")


def save_metrics(category_metrics, category, model_name):
    """Saves metrics in the Anomalib directory structure."""
    config = load_model_config(model_name)
    result_dirname = config["result_dirname"]
    category_base_dir = DIR_RESULTS / result_dirname / "MVTecAD" / category
    
    if not category_base_dir.exists():
        return
    
    # Find current version (v0, v1, v2, ...)
    versions = [d.name for d in category_base_dir.iterdir() 
                if d.is_dir() and d.name.startswith('v') and d.name[1:].isdigit()]
    if not versions:
        return
        
    latest_version = sorted(versions, key=lambda x: int(x[1:]))[-1]
    
    # Save in v_n
    version_dir = category_base_dir / latest_version
    version_json_path = version_dir / "metrics.json"
    with open(version_json_path, 'w', encoding='utf-8') as f:
        json.dump(category_metrics, f, indent=2, ensure_ascii=False)
    print(f"   Saved: {version_json_path}")
    
    # Save in latest (only if it exists)
    latest_dir = category_base_dir / "latest"
    if latest_dir.exists():
        latest_json_path = latest_dir / "metrics.json"
        with open(latest_json_path, 'w', encoding='utf-8') as f:
            json.dump(category_metrics, f, indent=2, ensure_ascii=False)


def print_category_metrics(metrics):
    """Prints metrics for a category."""
    print(f"\n[METRICS]")
    print(f"   EFFICACY:   AUROC img={format_metric(metrics['image_auroc'])} | "
          f"AUROC pix={format_metric(metrics['pixel_auroc'])} | "
          f"F1={format_metric(metrics['image_f1'])}")
    print(f"   EFFICIENCY: Train={format_metric(metrics['train_time_s'], 1)}s | "
          f"Inf={format_metric(metrics['inference_time_ms'], 1)}ms | "
          f"FPS={format_metric(metrics['fps'], 1)} | "
          f"Size={format_metric(metrics['model_size_mb'], 1)}MB")


def print_final_report(all_metrics, model_name):
    """Prints final report with all metrics."""
    if not all_metrics:
        return
        
    print(f"\n{'='*100}")
    print(f"FINAL REPORT - {model_name.upper()} PERFORMANCE METRICS")
    print(f"{'='*100}\n")
    
    # Header
    header = f"{'Category':<12} | {'Img AUROC':<10} | {'Pix AUROC':<10} | {'Img F1':<10} | {'Train(s)':<10} | {'Inf(ms)':<10} | {'FPS':<8} | {'Size(MB)':<10}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for m in all_metrics:
        print(f"{m['category']:<12} | "
              f"{format_metric(m['image_auroc']):<10} | "
              f"{format_metric(m['pixel_auroc']):<10} | "
              f"{format_metric(m['image_f1']):<10} | "
              f"{format_metric(m['train_time_s'], 2):<10} | "
              f"{format_metric(m['inference_time_ms'], 2):<10} | "
              f"{format_metric(m['fps'], 1):<8} | "
              f"{format_metric(m['model_size_mb'], 2):<10}")
    
    # Average (only if more than one category)
    if len(all_metrics) > 1:
        print("-" * len(header))
        print(f"{'AVERAGE':<12} | "
              f"{format_metric(safe_mean([m['image_auroc'] for m in all_metrics])):<10} | "
              f"{format_metric(safe_mean([m['pixel_auroc'] for m in all_metrics])):<10} | "
              f"{format_metric(safe_mean([m['image_f1'] for m in all_metrics])):<10} | "
              f"{format_metric(safe_mean([m['train_time_s'] for m in all_metrics]), 2):<10} | "
              f"{format_metric(safe_mean([m['inference_time_ms'] for m in all_metrics]), 2):<10} | "
              f"{format_metric(safe_mean([m['fps'] for m in all_metrics]), 1):<8} | "
              f"{format_metric(safe_mean([m['model_size_mb'] for m in all_metrics]), 2):<10}")
    
    print(f"\n{'='*100}")


def train_category(category, model_name):
    """Runs training, test, and calculates metrics for a category."""
    print(f"\n{'='*60}")
    print(f"Training: {category} ({model_name})")
    print(f"{'='*60}")
    
    # Load config
    config = load_model_config(model_name)
    
    # Initialize data with train_batch_size if specified (required for EfficientAD)
    train_batch_size = config.get("train_batch_size", 32)
    datamodule = MVTecAD(root=str(DIR_DATASET), 
                         category=category, train_batch_size=train_batch_size)
    
    # Initialize model
    model_class = get_class_from_path(config["class_path"])
    model_params = config["init_args"]
    
    # EfficientAd-specific setup
    if model_name == "efficientad":
        patch_efficientad()
        model_params["imagenet_dir"] = str(EFFICIENTAD_RESOURCES_DIR / "imagenette")
        print(f"   [INFO] EfficientAd: ImageNet directory: {EFFICIENTAD_RESOURCES_DIR / 'imagenette'}")
        print("   [INFO] EfficientAd: Image visualization disabled")
        model_params["visualizer"] = False
        
    model = model_class(**model_params)
    
    # Training
    train_start = time.time()
    max_epochs = config.get("max_epochs", 100)
    engine = Engine(default_root_dir=str(DIR_RESULTS), max_epochs=max_epochs)
    engine.fit(model=model, datamodule=datamodule)
    train_time = time.time() - train_start
    
    # Test
    test_results = engine.test(model=model, datamodule=datamodule)
    metrics = test_results[0] if test_results else {}
    
    # Inference for FPS measurement
    inference_start = time.time()
    predictions = engine.predict(model=model, datamodule=datamodule)
    inference_time = time.time() - inference_start
    num_images = len(predictions) if predictions else 1
    
    # Collect metrics
    category_metrics = {
        "category": category,
        "image_auroc": metrics.get('image_AUROC'),
        "pixel_auroc": metrics.get('pixel_AUROC'),
        "image_f1": metrics.get('image_F1Score'),
        "train_time_s": train_time,
        "inference_time_ms": (inference_time / num_images) * 1000,
        "fps": num_images / inference_time if inference_time > 0 else 0,
        "model_size_mb": get_model_size_mb(model),
    }
    
    # Output and save
    print_category_metrics(category_metrics)
    save_metrics(category_metrics, category, model_name)
    print(f"\nCompleted: {category}\n")
    
    return category_metrics


def parse_args():
    """Parse command line arguments."""
    available_models = get_available_models()
    
    parser = argparse.ArgumentParser(
        description="Training script for MVTec AD using Anomalib",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                     # Training on bottle (default model: patchcore)
  python train.py --model all         # Train all models on bottle
  python train.py --category all      # Train Patchcore on all categories
  python train.py --model all --category all  # Train all models on all categories
        """
    )
    parser.add_argument(
        "--category", type=str, default="bottle",
        choices=MVTEC_CATEGORIES + ["all"],
        help="Category to train on, or 'all' (default: bottle)"
    )
    parser.add_argument(
        "--model", type=str, default="patchcore",
        choices=available_models + ["all"],
        help="Model to use, or 'all' (default: patchcore)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.category == "all":
        categories = MVTEC_CATEGORIES
        print(f"Training on ALL {len(categories)} categories")
    else:
        categories = [args.category]
        print(f"Training on: {args.category}")
    
    if args.model == "all":
        models = get_available_models()
        print(f"Models: ALL ({', '.join(models)})")
    else:
        models = [args.model]
        print(f"Model: {args.model}")

    DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    for model_name in models:
        if len(models) > 1:
            print(f"\n{'='*60}")
            print(f"MODEL: {model_name.upper()}")
            print(f"{'='*60}")
        
        model_metrics = [train_category(cat, model_name) for cat in categories]
        all_metrics.extend(model_metrics)
        
        print_final_report(model_metrics, model_name)


if __name__ == "__main__":
    main()
