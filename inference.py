#!/usr/bin/env python3
"""
Inference script for MVTec AD using Anomalib models.

Visualizes heatmap and masks of detected anomalies.

Examples:
    python inference.py --image_path path/to/image.png
    python inference.py --image_path image.png --model padim
    python inference.py --image_path image.png --category cable
"""

import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from anomalib.data import PredictDataset
from anomalib.engine import Engine

from core import (
    MVTEC_CATEGORIES,
    DIR_RESULTS,
    DIR_OUTPUT,
    get_available_models,
    load_model_config,
    load_model,
    get_checkpoint_path,
    resize_to_match,
    scale_efficientad_score,
)


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(image_path: str, category: str = "bottle", model_name: str = "patchcore", checkpoint_path: str = None) -> dict:
    """
    Runs inference on a single image.
    
    Args:
        image_path: Path to the image
        category: Category for the model
        model_name: Name of the model to use
        checkpoint_path: Path to checkpoint (optional, uses default if not provided)
    
    Returns:
        dict with: anomaly_score, anomaly_map, pred_mask, image_path
    """
    ckpt = checkpoint_path or str(get_checkpoint_path(category, model_name))
    
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    
    # Load model and dataset
    model = load_model(model_name)
    dataset = PredictDataset(path=image_path)
    
    # Run prediction (disable automatic image saving)
    engine = Engine(
        default_root_dir="/tmp/anomalib_inference",
        callbacks=[],
    )
    predictions = engine.predict(model=model, dataset=dataset, ckpt_path=ckpt)
    
    # Extract results
    results = {"image_path": image_path, "category": category, "model": model_name}
    
    for batch in predictions:
        results["anomaly_score"] = float(batch.pred_score[0].cpu().numpy()) if batch.pred_score is not None else None
        results["anomaly_map"] = batch.anomaly_map[0].cpu().numpy() if batch.anomaly_map is not None else None
        results["pred_mask"] = batch.pred_mask[0].cpu().numpy() if batch.pred_mask is not None else None
        break
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_results(results: dict, output_dir: Path = None) -> str:
    """
    Visualizes heatmap and mask overlayed on original image.
    
    Args:
        results: Results from run_inference()
        output_dir: Output directory (optional)
    
    Returns:
        Path of saved image
    """
    # Get model name for subfolder
    model_name = results.get("model", "unknown")
    
    # Create output/model_name subfolder
    base_output_dir = output_dir or DIR_OUTPUT
    output_dir = base_output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    image_path = results["image_path"]
    anomaly_score = results["anomaly_score"]
    anomaly_map = results["anomaly_map"]
    pred_mask = results.get("pred_mask")
    model_name = results.get("model", "unknown")
    
    original = np.array(Image.open(image_path).convert("RGB"))
    
    # Prepare anomaly map
    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map.squeeze(0)
    
    # Adaptive normalization: use image-level normalization if values are compressed
    is_efficientad = model_name.lower() == "efficientad"
    
    # Adaptive normalization: use image-level normalization if values are compressed
    # STRICTLY RESTRICTED TO EFFICIENTAD
    amap_min, amap_max = anomaly_map.min(), anomaly_map.max()
    amap_range = amap_max - amap_min
    
    if is_efficientad and amap_range < 0.1:  # Compressed values (e.g., EfficientAD output)
        # Apply image-level min-max normalization
        if amap_range > 1e-8:
            anomaly_map = (anomaly_map - amap_min) / amap_range
        else:
            anomaly_map = np.zeros_like(anomaly_map)
        print(f"ℹ️  Applied image-level normalization (original range: {amap_min:.4f}-{amap_max:.4f})")
    else:
        # Standard clipping for all other models
        anomaly_map = np.clip(anomaly_map, 0, 1)
    
    anomaly_map = resize_to_match(anomaly_map, original.shape[:2])
    
    # EfficientAD-specific: for "good" images, force blue heatmap
    # is_efficientad already defined above
    is_good = anomaly_score is not None and anomaly_score < 0.5
    show_mask_contours = True  # Whether to draw contours on mask panel
    
    if is_efficientad and is_good:
        # Scale heatmap to low values but keep texture variation (0-0.3 range)
        anomaly_map = anomaly_map * 0.3  # Keep variation but in low range
        show_mask_contours = False  # Don't draw contours for good images
        print(f"ℹ️  EfficientAD: Good image detected - scaling heatmap to low values")
    
    # Prepare pred mask
    if pred_mask is not None:
        if pred_mask.ndim == 3:
            pred_mask = pred_mask.squeeze(0)
        pred_mask = resize_to_match(pred_mask, original.shape[:2])
    
    # Create figure - always 4 columns if pred_mask exists or EfficientAD good image
    show_fourth_panel = pred_mask is not None or (is_efficientad and is_good)
    num_cols = 4 if show_fourth_panel else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5), facecolor='white')
    
    # Set white background for all axes to avoid colored borders
    for ax in axes:
        ax.set_facecolor('white')
    
    # 1. Original Image
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    # 2. Heatmap - use aspect='auto' to fill subplot without borders
    # 2. Heatmap - use aspect='auto' to fill subplot without borders
    
    if is_efficientad:
        # Mask 0 values (padding) to remove blue border
        anomaly_map_masked = np.ma.masked_where(anomaly_map == 0, anomaly_map)
        cmap = plt.cm.jet
        cmap.set_bad(color='none')  # Transparent for masked values (padding)
    else:
        # Standard visualization for other models (0 is Blue, not transparent)
        anomaly_map_masked = anomaly_map
        cmap = plt.cm.jet
    
    im = axes[1].imshow(anomaly_map_masked, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    axes[1].set_title(f"Anomaly Heatmap ({model_name})")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    
    # 3. Overlay - use aspect='auto' to match original image
    axes[2].imshow(original, aspect='auto')
    axes[2].imshow(anomaly_map_masked, cmap=cmap, alpha=0.5, vmin=0, vmax=1, aspect='auto')
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    # 4. Mask panel
    if show_fourth_panel:
        axes[3].imshow(original)
        if show_mask_contours and pred_mask is not None:
            axes[3].contour(pred_mask, levels=[0.5], colors="red", linewidths=2)
        axes[3].set_title("Predicted Mask")
        axes[3].axis("off")
    
    # Title and Save - scale score to push towards 0 or 1 for EfficientAD
    if anomaly_score is not None:
        if is_efficientad:
            # Scale EfficientAD scores (typically ~0.5) to be more extreme
            # Good (<0.5) -> ~0.2, Anomalous (>0.5) -> high values
            scaled_score = scale_efficientad_score(anomaly_score)
            score_str = f"{scaled_score:.4f}"
        else:
            score_str = f"{anomaly_score:.4f}"
    else:
        score_str = "N/A"
    plt.suptitle(f"Model: {model_name} | Anomaly Score: {score_str}", fontsize=14)
    plt.tight_layout()
    
    image_name = Path(image_path).stem
    output_path = output_dir / f"inference_{image_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    
    # Always close the figure
    plt.close(fig)
    
    return str(output_path)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference script for MVTec AD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --image_path test.png
  python inference.py --image_path test.png --model padim
  python inference.py --image_path test.png --category cable
        """
    )
    parser.add_argument(
        "--image_path", type=str, required=True,
        help="Path to image to analyze"
    )
    parser.add_argument(
        "--category", type=str, default="bottle", choices=MVTEC_CATEGORIES,
        help="Category of the model (default: bottle)"
    )
    parser.add_argument(
        "--model", type=str, default="patchcore", choices=get_available_models(),
        help="Model to use (default: patchcore)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint path (optional, uses default for category/model)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (optional)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    
    print(f"🔍 Inference on: {args.image_path}")
    print(f"📦 Model: {args.model} | Category: {args.category}")
    
    # Run inference
    results = run_inference(
        image_path=args.image_path,
        category=args.category,
        model_name=args.model,
        checkpoint_path=args.checkpoint
    )
    
    # Print result
    score = results["anomaly_score"]
    print(f"\n{'='*50}")
    print(f"ANOMALY SCORE: {score:.4f}" if score else "ANOMALY SCORE: N/A")
    print(f"{'='*50}\n")
    
    # Visualize and save
    output_dir = Path(args.output_dir) if args.output_dir else None
    output_path = visualize_results(results, output_dir)
    
    print(f"Saved: {output_path}")
    
    return results


if __name__ == "__main__":
    main()