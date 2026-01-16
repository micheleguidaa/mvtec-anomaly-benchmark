"""
Prediction and event handler functions for Gradio UI.

Contains the main prediction logic, category helpers, and callback handlers
for the Gradio interface components.
"""

import numpy as np
from PIL import Image
import gradio as gr

from inference import run_inference
from core import (
    MVTEC_CATEGORIES,
    DIR_RESULTS,
    load_model_config,
    scale_efficientad_score,
)
from gradio_ui.visualization import (
    create_visualization,
    create_comparison_visualization,
)


def get_trained_categories(model_name: str) -> list:
    """
    Returns categories that have trained checkpoints for a given model.
    
    Args:
        model_name: Name of the model to check
    
    Returns:
        List of trained category names
    """
    try:
        config = load_model_config(model_name)
        result_dirname = config["result_dirname"]
        model_results_dir = DIR_RESULTS / result_dirname / "MVTecAD"
        
        if not model_results_dir.exists():
            return []
        
        trained = []
        for cat in MVTEC_CATEGORIES:
            ckpt = model_results_dir / cat / "latest" / "weights" / "lightning" / "model.ckpt"
            if ckpt.exists():
                trained.append(cat)
        return trained
    except Exception:
        return []


def predict(image, model_name: str, category: str):
    """
    Main prediction function for Gradio interface.
    
    Args:
        image: Input image (numpy array or PIL Image)
        model_name: Selected model name
        category: Selected category
    
    Returns:
        Tuple of (visualization_image, anomaly_score_text, status_text)
    """
    if image is None:
        return None, "⚠️ Please upload an image", ""
    
    try:
        # Save temp image if needed
        if isinstance(image, np.ndarray):
            temp_path = "/tmp/gradio_input.png"
            Image.fromarray(image).save(temp_path)
            image_path = temp_path
        else:
            image_path = image
        
        # Run inference
        results = run_inference(
            image_path=image_path,
            category=category,
            model_name=model_name
        )
        
        # Get results
        score = results["anomaly_score"]
        anomaly_map = results["anomaly_map"]
        pred_mask = results.get("pred_mask")
        
        # Load original image
        original = np.array(Image.open(image_path).convert("RGB"))
        
        # Create visualization
        viz_image = create_visualization(original, anomaly_map, pred_mask, score, model_name)
        
        # Format score output
        if score is not None:
            # Scale score for EfficientAD if needed
            display_score = score
            if "efficientad" in model_name.lower():
                display_score = scale_efficientad_score(score)
                
            if display_score > 0.5:
                combined_text = '<h3 style="color: red; margin-top: 10px;">🔴 ANOMALY DETECTED</h3>'
            else:
                combined_text = '<h3 style="color: green; margin-top: 10px;">🟢 NORMAL</h3>'
        else:
            combined_text = "⚠️ Could not determine status"
        
        return viz_image, combined_text
        
    except FileNotFoundError as e:
        return None, f"❌ Error: {str(e)}\n\nCheck if model is trained for this category"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def predict_sketch(editor_output, model_name: str, category: str):
    """
    Prediction function for sketch interface.
    Handles the ImageEditor output format.
    
    Args:
        editor_output: Output from ImageEditor (dict with 'composite' key)
        model_name: Selected model name
        category: Selected category
    
    Returns:
        Tuple of (visualization_image, combined_text)
    """
    if editor_output is None:
        return None, "⚠️ Please upload an image and draw defects"
    
    # Extract the composite image from editor output
    # ImageEditor returns a dict with 'background', 'layers', and 'composite' keys
    if isinstance(editor_output, dict):
        image = editor_output.get("composite")
        if image is None:
            image = editor_output.get("background")
    else:
        image = editor_output
    
    if image is None:
        return None, "⚠️ No image found"
    
    # Use the standard predict function
    return predict(image, model_name, category)


def predict_compare(image, selected_models: list, category: str):
    """
    Run inference on multiple models and create comparison visualization.
    
    Args:
        image: Input image
        selected_models: List of model names to compare
        category: Selected category
    
    Returns:
        Tuple of (comparison_image, summary_text)
    """
    if image is None:
        return None, "⚠️ Please upload an image"
    
    if not selected_models or len(selected_models) < 2:
        return None, "⚠️ Please select at least 2 models to compare"
    
    if len(selected_models) > 5:
        return None, "⚠️ Please select at most 5 models"
    
    try:
        # Save temp image if needed
        if isinstance(image, np.ndarray):
            temp_path = "/tmp/gradio_compare_input.png"
            Image.fromarray(image).save(temp_path)
            image_path = temp_path
        else:
            image_path = image
        
        # Load original image
        original = np.array(Image.open(image_path).convert("RGB"))
        
        # Run inference on each model
        results_list = []
        for model_name in selected_models:
            try:
                results = run_inference(
                    image_path=image_path,
                    category=category,
                    model_name=model_name
                )
                results_list.append({
                    'model_name': model_name,
                    'anomaly_map': results['anomaly_map'],
                    'score': results['anomaly_score'],
                    'pred_mask': results.get('pred_mask'),
                    'error': None
                })
            except Exception as e:
                results_list.append({
                    'model_name': model_name,
                    'error': str(e)
                })
        
        # Create comparison visualization
        viz_image = create_comparison_visualization(original, results_list)
        
        return viz_image, ""
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def get_common_categories(model_names: list) -> list:
    """
    Get categories that are trained for all selected models.
    
    Args:
        model_names: List of model names
    
    Returns:
        List of commonly trained categories
    """
    if not model_names:
        return MVTEC_CATEGORIES
    
    common = None
    for model_name in model_names:
        trained = set(get_trained_categories(model_name))
        if common is None:
            common = trained
        else:
            common = common.intersection(trained)
    
    return list(common) if common else MVTEC_CATEGORIES


def update_categories(model_name: str, current_category: str = None):
    """
    Updates available categories based on selected model.
    Keeps the current category if it's available for the new model.
    
    Args:
        model_name: Selected model name
        current_category: Currently selected category to preserve if possible
    
    Returns:
        Updated Gradio Dropdown component
    """
    trained = get_trained_categories(model_name)
    if trained:
        # Keep current category if it's available in the new model
        if current_category and current_category in trained:
            return gr.Dropdown(choices=trained, value=current_category)
        return gr.Dropdown(choices=trained, value=trained[0])
    # Keep current category if using all categories
    if current_category and current_category in MVTEC_CATEGORIES:
        return gr.Dropdown(choices=MVTEC_CATEGORIES, value=current_category)
    return gr.Dropdown(choices=MVTEC_CATEGORIES, value="bottle")


def update_compare_categories(selected_models: list, current_category: str = None):
    """
    Update categories dropdown based on selected models.
    Keeps the current category if it's available for all selected models.
    
    Args:
        selected_models: List of selected model names
        current_category: Currently selected category to preserve if possible
    
    Returns:
        Updated Gradio Dropdown component
    """
    common = get_common_categories(selected_models)
    if common:
        # Keep current category if it's in common categories
        if current_category and current_category in common:
            return gr.Dropdown(choices=common, value=current_category)
        return gr.Dropdown(choices=common, value=common[0])
    # Keep current category if using all categories
    if current_category and current_category in MVTEC_CATEGORIES:
        return gr.Dropdown(choices=MVTEC_CATEGORIES, value=current_category)
    return gr.Dropdown(choices=MVTEC_CATEGORIES, value="bottle")


def get_sample_images() -> list:
    """
    Returns a list of sample images from the MVTecAD test set.

    First tries DIR_DATASET (full dataset), then falls back to sample_images folder.

    Returns:
        List of tuples (image_path, label) for gallery display
    """
    from pathlib import Path
    from core import DIR_DATASET

    samples = []

    # Try full dataset first
    for category in MVTEC_CATEGORIES:
        test_dir = DIR_DATASET / category / "test"
        if not test_dir.exists():
            continue

        for defect_type in test_dir.iterdir():
            if not defect_type.is_dir():
                continue

            images = list(defect_type.glob("*.png"))
            if images:
                img_path = str(images[0])
                label = f"{category}/{defect_type.name}"
                samples.append((img_path, label))

    # Fallback to sample_images folder if no samples found
    if not samples:
        sample_dir = Path(__file__).parent.parent / "sample_images"
        for category in MVTEC_CATEGORIES:
            category_dir = sample_dir / category
            if not category_dir.exists():
                continue

            for defect_type in category_dir.iterdir():
                if not defect_type.is_dir():
                    continue

                images = list(defect_type.glob("*.png"))
                if images:
                    img_path = str(images[0])
                    label = f"{category}/{defect_type.name}"
                    samples.append((img_path, label))

    return samples


def get_category_from_sample(image_path_or_label: str) -> str:
    """
    Extracts category from a sample image path or gallery label.
    
    Args:
        image_path_or_label: Path to the sample image or gallery label (format: category/defect_type)
    
    Returns:
        Category name extracted from the path or label
    """
    if image_path_or_label is None:
        return "bottle"
    
    path_str = str(image_path_or_label)
    
    # First try: check if it's a label format "category/defect_type"
    for category in MVTEC_CATEGORIES:
        if path_str.startswith(f"{category}/") or path_str == category:
            return category
    
    # Second try: check if category is in the path
    for category in MVTEC_CATEGORIES:
        if f"/{category}/" in path_str or f"\\{category}\\" in path_str:
            return category
    
    return "bottle"


def on_sample_select(evt: gr.SelectData, current_model: str = None):
    """
    Handler for when a sample image is selected from the gallery.
    Returns the image and updates the category.
    
    Args:
        evt: Gradio SelectData event with image info
        current_model: Currently selected model (for category validation)
    
    Returns:
        Tuple of (image, category_dropdown_update)
    """
    if evt is None or evt.value is None:
        return None, gr.Dropdown()
    
    # Get the image path from the event
    image_data = evt.value
    if isinstance(image_data, dict):
        image_path = image_data.get("image", {}).get("path", "")
    else:
        image_path = str(image_data)
    
    # Extract category from path
    category = get_category_from_sample(image_path)
    
    # Load the image
    try:
        img = np.array(Image.open(image_path).convert("RGB"))
    except Exception:
        return None, gr.Dropdown()
    
    # Get available categories for current model
    if current_model:
        trained = get_trained_categories(current_model)
        if trained and category in trained:
            return img, gr.Dropdown(choices=trained, value=category)
        elif trained:
            return img, gr.Dropdown(choices=trained, value=trained[0])
    
    return img, gr.Dropdown(choices=MVTEC_CATEGORIES, value=category)
