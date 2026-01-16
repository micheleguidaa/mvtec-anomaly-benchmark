"""
Visualization utilities for anomaly detection results.

Contains functions to create matplotlib visualizations for single model results
and side-by-side model comparisons.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

from inference import resize_to_match
from core import scale_efficientad_score


def create_visualization(
    original: np.ndarray, 
    anomaly_map: np.ndarray, 
    pred_mask: np.ndarray, 
    score: float, 
    model_name: str
) -> np.ndarray:
    """
    Creates a visualization figure and returns it as an image array.
    
    Args:
        original: Original image as numpy array
        anomaly_map: Anomaly map array
        pred_mask: Prediction mask (optional)
        score: Anomaly score
        model_name: Name of the model used
    
    Returns:
        Visualization as numpy array
    """
    # Prepare anomaly map
    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map.squeeze(0)
    
    # Adaptive normalization for compressed outputs (like EfficientAD)
    is_efficientad = "efficientad" in model_name.lower()
    
    amap_min = anomaly_map.min()
    amap_max = anomaly_map.max()
    amap_range = amap_max - amap_min
    
    original_map = anomaly_map.copy() # Keep original for checks if needed
    
    if is_efficientad and amap_range < 0.1:
        # Range is too small (e.g. EfficientAD 0.49-0.51) -> normalize to 0-1
        anomaly_map = (anomaly_map - amap_min) / amap_range
    else:
        # Standard range -> just clip
        anomaly_map = np.clip(anomaly_map, 0, 1)

    anomaly_map = resize_to_match(anomaly_map, original.shape[:2])
    
    # Prepare pred mask
    if pred_mask is not None:
        if pred_mask.ndim == 3:
            pred_mask = pred_mask.squeeze(0)
        pred_mask = resize_to_match(pred_mask, original.shape[:2])
    
    # EfficientAD-specific logic
    # is_efficientad already defined above
    is_good = score < 0.5
    
    score_str = f"{score:.4f}"
    status_text = "ANOMALY" if score > 0.5 else "NORMAL"
    
    if is_efficientad:
        # Scale EfficientAD scores
        scaled_score = scale_efficientad_score(score)
        score_str = f"{scaled_score:.4f}"
        
        if is_good:
            # Scale heatmap to low values but keep texture variation (0-0.3 range)
            anomaly_map = anomaly_map * 0.3
    
    # Create figure
    # Always show mask panel if it exists
    show_mask_panel = pred_mask is not None or (is_efficientad and is_good)
    num_cols = 4 if show_mask_panel else 3
    
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5), facecolor='white')
    for ax in axes:
        ax.set_facecolor('white')
    
    # 1. Original Image
    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")
    
    # 2. Heatmap
    # Mask 0 values (padding) to remove blue border
    # Mask 0 values (padding) to remove blue border only for EfficientAD
    if is_efficientad:
        anomaly_map_masked = np.ma.masked_where(anomaly_map == 0, anomaly_map)
        cmap = plt.cm.jet
        cmap.set_bad(color='none')  # Transparent for masked values
    else:
        anomaly_map_masked = anomaly_map
        cmap = plt.cm.jet
    
    im = axes[1].imshow(anomaly_map_masked, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    axes[1].set_title(f"Anomaly Heatmap", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Overlay
    axes[2].imshow(original, aspect='auto')
    axes[2].imshow(anomaly_map_masked, cmap=cmap, alpha=0.5, vmin=0, vmax=1, aspect='auto')
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis("off")
    
    # 4. Mask
    if show_mask_panel:
        axes[3].imshow(original)
        # Only draw contours if NOT (EfficientAD AND Good)
        if pred_mask is not None and not (is_efficientad and is_good):
            axes[3].contour(pred_mask, levels=[0.5], colors="red", linewidths=2)
        axes[3].set_title("Predicted Mask", fontsize=12)
        axes[3].axis("off")
    
    # Title
    plt.suptitle(f"Model: {model_name} | Score: {score_str} | {status_text}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert figure to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    result_img = np.array(Image.open(buf))
    plt.close(fig)
    
    return result_img


def create_comparison_visualization(original: np.ndarray, results_list: list) -> np.ndarray:
    """
    Creates a side-by-side comparison visualization of multiple models.
    
    Args:
        original: Original image as numpy array
        results_list: List of dicts with 'model_name', 'anomaly_map', 'score', 'pred_mask', 'error'
    
    Returns:
        Comparison visualization as numpy array
    """
    valid_results = [r for r in results_list if r.get('error') is None]
    
    if not valid_results:
        # Return error image
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No valid model results', ha='center', va='center', fontsize=16)
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        result_img = np.array(Image.open(buf))
        plt.close(fig)
        return result_img
    
    num_models = len(valid_results)
    
    # TRANSPOSED: rows = models, columns = visualization types
    # 4 columns: Original, Heatmap, Overlay, Pred Mask
    fig, axes = plt.subplots(num_models, 4, figsize=(16, 4 * num_models), facecolor='white')
    for ax in axes.flat:
        ax.set_facecolor('white')
    
    # Handle single model case (axes needs to be 2D)
    if num_models == 1:
        axes = axes.reshape(1, -1)
    
    # Column headers
    col_titles = ["Original", "Heatmap", "Overlay", "Pred Mask"]
    
    # For each model (each row)
    for i, result in enumerate(valid_results):
        row = i
        model_name = result['model_name']
        anomaly_map = result['anomaly_map']
        score = result['score']
        pred_mask = result.get('pred_mask')
        
        # Prepare anomaly map
        if anomaly_map.ndim == 3:
            anomaly_map = anomaly_map.squeeze(0)
            
        # Adaptive normalization
        is_efficientad = "efficientad" in model_name.lower()
        
        amap_min = anomaly_map.min()
        amap_max = anomaly_map.max()
        amap_range = amap_max - amap_min
        
        if is_efficientad and amap_range < 0.1:
            anomaly_map = (anomaly_map - amap_min) / amap_range
        else:
            anomaly_map = np.clip(anomaly_map, 0, 1)

        anomaly_map = resize_to_match(anomaly_map, original.shape[:2])
        
        # Prepare pred mask
        if pred_mask is not None:
            if pred_mask.ndim == 3:
                pred_mask = pred_mask.squeeze(0)
            pred_mask = resize_to_match(pred_mask, original.shape[:2])
        
        # EfficientAD-specific logic
        # is_efficientad already defined above
        is_good = score < 0.5
        
        display_score = score
        if is_efficientad:
            # Scale EfficientAD scores
            display_score = scale_efficientad_score(score)
            
            if is_good:
                anomaly_map = anomaly_map * 0.3
            
        # Status indicator
        status = "ANOMALY" if display_score > 0.5 else "NORMAL"
        
        # Mask 0 values (padding)
        if is_efficientad:
            anomaly_map_masked = np.ma.masked_where(anomaly_map == 0, anomaly_map)
            cmap = plt.cm.jet
            cmap.set_bad(color='none')
        else:
            anomaly_map_masked = anomaly_map
            cmap = plt.cm.jet
        
        # Col 0: Original
        axes[row, 0].imshow(original)
        # annotations handle the labels
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title(col_titles[0], fontsize=12, fontweight='bold')
        
        # Col 1: Heatmap
        im = axes[row, 1].imshow(anomaly_map_masked, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        axes[row, 1].axis('off')
        if row == 0:
            axes[row, 1].set_title(col_titles[1], fontsize=12, fontweight='bold')
        
        # Col 2: Overlay
        axes[row, 2].imshow(original, aspect='auto')
        axes[row, 2].imshow(anomaly_map_masked, cmap=cmap, alpha=0.5, vmin=0, vmax=1, aspect='auto')
        axes[row, 2].axis('off')
        if row == 0:
            axes[row, 2].set_title(col_titles[2], fontsize=12, fontweight='bold')
        
        # Col 3: Prediction Mask
        axes[row, 3].imshow(original)
        # Only draw contours if NOT (EfficientAD AND Good)
        if pred_mask is not None and not (is_efficientad and is_good):
            axes[row, 3].contour(pred_mask, levels=[0.5], colors='red', linewidths=2)
            axes[row, 3].contourf(pred_mask, levels=[0.5, 1.0], colors=['red'], alpha=0.3)
        axes[row, 3].axis('off')
        if row == 0:
            axes[row, 3].set_title(col_titles[3], fontsize=12, fontweight='bold')
        
        # Add model name as row label on the left using annotation
        # Show model name, score and status
        status_color = 'red' if display_score > 0.5 else 'green'
        axes[row, 0].annotate(f"{model_name}\nScore: {display_score:.2f}\n{status}", 
                              xy=(-0.1, 0.5), xycoords='axes fraction',
                              fontsize=11, fontweight='bold', ha='right', va='center',
                              color=status_color)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, label='Anomaly Score')
    
    plt.suptitle("Model Comparison", fontsize=14, fontweight='bold', y=0.98)
    # Adjust rect to ensure colorbar (right) and labels (left) fit
    plt.tight_layout(rect=[0.1, 0, 0.9, 0.95])
    
    # Convert figure to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    result_img = np.array(Image.open(buf))
    plt.close(fig)
    
    return result_img
