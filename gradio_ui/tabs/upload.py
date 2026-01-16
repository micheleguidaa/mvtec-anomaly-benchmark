"""
Upload Image Tab component.

Provides the standard image upload and analysis interface.
"""

import gradio as gr
from gradio_ui.handlers import get_sample_images


def create_upload_tab(available_models: list, default_model: str, initial_categories: list):
    """
    Creates the Upload Image tab.
    
    Args:
        available_models: List of available model names
        default_model: Default model to select
        initial_categories: Initial list of categories
    
    Returns:
        Dict with tab components for wiring events
    """
    with gr.TabItem("📤 Upload Image"):
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                
                image_input = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    height=300
                )
                
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=available_models,
                        value=default_model,
                        label="🧠 Model",
                        interactive=True
                    )
                    category_dropdown = gr.Dropdown(
                        choices=initial_categories,
                        value=initial_categories[0] if initial_categories else "bottle",
                        label="📦 Category",
                        interactive=True
                    )
                
                analyze_btn = gr.Button(
                    "🔬 Analyze Image",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Output
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                
                result_image = gr.Image(
                    label="Visualization",
                    type="numpy",
                    height=300
                )

                result_text = gr.Markdown(value="", label="Results")
        
        # Sample Images Gallery
        gr.Markdown("### 🖼️ Sample Images (click to select)")
        sample_gallery = gr.Gallery(
            value=get_sample_images(),
            label="Sample Images",
            show_label=False,
            columns=8,
            rows=3,
            height=350,
            object_fit="cover",
            allow_preview=False,
        )
        
    
    return {
        "image_input": image_input,
        "model_dropdown": model_dropdown,
        "category_dropdown": category_dropdown,
        "analyze_btn": analyze_btn,
        "result_image": result_image,
        "result_image": result_image,
        "result_text": result_text,
        "sample_gallery": sample_gallery,
    }
