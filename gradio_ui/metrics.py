"""
Metrics loading and display utilities.

Contains functions to load, filter, and format model performance metrics
from the results directory.
"""

import json

from inference import DIR_RESULTS


def load_all_metrics() -> list:
    """
    Load metrics from all trained models and categories.
    
    Returns:
        List of dicts with model, category, and metrics.
    """
    all_metrics = []
    
    if not DIR_RESULTS.exists():
        return all_metrics
    
    # Iterate through model directories
    for model_dir in DIR_RESULTS.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        mvtec_dir = model_dir / "MVTecAD"
        
        if not mvtec_dir.exists():
            continue
        
        # Iterate through category directories
        for cat_dir in mvtec_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            
            category = cat_dir.name
            
            # Try to find metrics.json (check latest first, then v0, etc.)
            metrics_file = None
            for version_dir in ['latest', 'v0', 'v1', 'v2']:
                potential_file = cat_dir / version_dir / 'metrics.json'
                if potential_file.exists():
                    metrics_file = potential_file
                    break
            
            if metrics_file and metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    all_metrics.append({
                        'model': model_name,
                        'category': category,
                        'image_auroc': metrics.get('image_auroc', 0),
                        'pixel_auroc': metrics.get('pixel_auroc', 0),
                        'image_f1': metrics.get('image_f1', 0),
                        'train_time_s': metrics.get('train_time_s', 0),
                        'inference_time_ms': metrics.get('inference_time_ms', 0),
                        'fps': metrics.get('fps', 0),
                        'model_size_mb': metrics.get('model_size_mb', 0),
                    })
                except Exception:
                    pass
    
    return all_metrics


def generate_metrics_table(model_filter: str = "All", category_filter: str = "All") -> str:
    """
    Generate a markdown table with metrics, optionally filtered.
    
    Args:
        model_filter: Filter by model name, or "All" for no filter
        category_filter: Filter by category, or "All" for no filter
    
    Returns:
        Markdown formatted table string
    """
    all_metrics = load_all_metrics()
    
    if not all_metrics:
        return "⚠️ No metrics found. Train some models first!"
    
    # Apply filters
    if model_filter != "All":
        all_metrics = [m for m in all_metrics if m['model'].lower() == model_filter.lower()]
    
    if category_filter != "All":
        all_metrics = [m for m in all_metrics if m['category'] == category_filter]
    
    if not all_metrics:
        return "⚠️ No metrics match the selected filters."
    
    # Sort by model then category
    all_metrics.sort(key=lambda x: (x['model'], x['category']))
    
    # Build markdown table
    lines = [
        "| Model | Category | Image AUROC | Pixel AUROC | F1 Score | Train Time | Inference | FPS | Size (MB) |",
        "|-------|----------|-------------|-------------|----------|------------|-----------|-----|-----------|" 
    ]
    
    for m in all_metrics:
        train_time = f"{m['train_time_s']:.1f}s" if m['train_time_s'] < 60 else f"{m['train_time_s']/60:.1f}m"
        inference = f"{m['inference_time_ms']:.0f}ms"
        
        lines.append(
            f"| {m['model']} | {m['category']} | "
            f"{m['image_auroc']:.4f} | {m['pixel_auroc']:.4f} | {m['image_f1']:.4f} | "
            f"{train_time} | {inference} | {m['fps']:.2f} | {m['model_size_mb']:.1f} |"
        )
    
    return "\n".join(lines)


def get_metrics_summary() -> str:
    """
    Generate a summary of best performing models.
    
    Returns:
        Markdown formatted summary string
    """
    all_metrics = load_all_metrics()
    
    if not all_metrics:
        return ""
    
    # Find best models for each metric
    best_image_auroc = max(all_metrics, key=lambda x: x['image_auroc'])
    best_pixel_auroc = max(all_metrics, key=lambda x: x['pixel_auroc'])
    best_f1 = max(all_metrics, key=lambda x: x['image_f1'])
    fastest = max(all_metrics, key=lambda x: x['fps'])
    smallest = min(all_metrics, key=lambda x: x['model_size_mb'])
    
    summary = f"""
### 🏆 Best Performers

| Metric | Best Model | Category | Value |
|--------|------------|----------|-------|
| 🎯 Image AUROC | {best_image_auroc['model']} | {best_image_auroc['category']} | {best_image_auroc['image_auroc']:.4f} |
| 🔬 Pixel AUROC | {best_pixel_auroc['model']} | {best_pixel_auroc['category']} | {best_pixel_auroc['pixel_auroc']:.4f} |
| ✅ F1 Score | {best_f1['model']} | {best_f1['category']} | {best_f1['image_f1']:.4f} |
| ⚡ Fastest | {fastest['model']} | {fastest['category']} | {fastest['fps']:.2f} FPS |
| 📦 Smallest | {smallest['model']} | {smallest['category']} | {smallest['model_size_mb']:.1f} MB |
"""
    return summary
