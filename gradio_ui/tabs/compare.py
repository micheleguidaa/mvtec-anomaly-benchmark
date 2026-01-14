"""
Compare Models Tab component.

Provides the side-by-side model comparison interface.
"""

import gradio as gr

from gradio_ui.content import COMPARE_MODELS_INSTRUCTIONS


def create_compare_tab(available_models: list, initial_categories: list):
    """
    Creates the Compare Models tab.
    
    Args:
        available_models: List of available model names
        initial_categories: Initial list of categories
    
    Returns:
        Dict with tab components for wiring events
    """
    with gr.TabItem("🔄 Compare Models"):
        gr.Markdown(COMPARE_MODELS_INSTRUCTIONS)
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                
                compare_image_input = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=300
                )
                
                gr.Markdown("### 🧠 Select Models (2-4)")
                
                compare_model_checkboxes = gr.CheckboxGroup(
                    choices=available_models,
                    value=available_models[:2] if len(available_models) >= 2 else available_models,
                    label="Models to Compare",
                    interactive=True
                )
                
                compare_category_dropdown = gr.Dropdown(
                    choices=initial_categories,
                    value=initial_categories[0] if initial_categories else "bottle",
                    label="📦 Category",
                    interactive=True
                )
                
                compare_btn = gr.Button(
                    "🔄 Compare Models",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Results
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Comparison Results")
                
                compare_result_image = gr.Image(
                    label="Side-by-Side Comparison",
                    type="numpy",
                    height=600
                )
                
                compare_summary = gr.Markdown("*Select models and upload an image to compare*")
    
    return {
        "image_input": compare_image_input,
        "model_checkboxes": compare_model_checkboxes,
        "category_dropdown": compare_category_dropdown,
        "compare_btn": compare_btn,
        "result_image": compare_result_image,
        "summary": compare_summary,
    }
