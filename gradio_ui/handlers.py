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
    
    if len(selected_models) > 4:
        return None, "⚠️ Please select at most 4 models"
    
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
        
        # Create summary text
        summary_lines = ["### 📊 Comparison Results\n"]
        summary_lines.append("| Model | Score | Status |")
        summary_lines.append("|-------|-------|--------|")
        
        for result in results_list:
            if result.get('error'):
                summary_lines.append(f"| {result['model_name']} | ❌ Error | {result['error'][:30]}... |")
            else:
                score = result['score']
                model_name = result['model_name']
                
                # Scale if EfficientAD
                display_score = score
                if "efficientad" in model_name.lower():
                    display_score = scale_efficientad_score(score)
                
                status = "🔴 Anomaly" if display_score > 0.5 else "🟢 Normal"
                summary_lines.append(f"| {model_name} | {display_score:.4f} | {status} |")
        
        summary = "\n".join(summary_lines)
        
        return viz_image, summary
        
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


def update_categories(model_name: str):
    """
    Updates available categories based on selected model.
    
    Args:
        model_name: Selected model name
    
    Returns:
        Updated Gradio Dropdown component
    """
    trained = get_trained_categories(model_name)
    if trained:
        return gr.Dropdown(choices=trained, value=trained[0])
    return gr.Dropdown(choices=MVTEC_CATEGORIES, value="bottle")


def update_compare_categories(selected_models: list):
    """
    Update categories dropdown based on selected models.
    
    Args:
        selected_models: List of selected model names
    
    Returns:
        Updated Gradio Dropdown component
    """
    common = get_common_categories(selected_models)
    if common:
        return gr.Dropdown(choices=common, value=common[0])
    return gr.Dropdown(choices=MVTEC_CATEGORIES, value="bottle")
