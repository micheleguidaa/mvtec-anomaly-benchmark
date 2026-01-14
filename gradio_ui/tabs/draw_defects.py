"""
Draw Defects Tab component.

Provides the sketch interface for drawing artificial defects on images.
"""

import gradio as gr

from gradio_ui.content import (
    DRAW_DEFECTS_INSTRUCTIONS,
    BRUSH_COLORS_INFO,
    HEATMAP_INTERPRETATION,
)


def create_draw_defects_tab(available_models: list, default_model: str, initial_categories: list):
    """
    Creates the Draw Defects tab.
    
    Args:
        available_models: List of available model names
        default_model: Default model to select
        initial_categories: Initial list of categories
    
    Returns:
        Dict with tab components for wiring events
    """
    with gr.TabItem("✏️ Draw Defects"):
        gr.Markdown(DRAW_DEFECTS_INSTRUCTIONS)
        
        with gr.Row():
            # Left column - Editor
            with gr.Column(scale=1):
                gr.Markdown("### 🖌️ Image Editor")
                
                image_editor = gr.ImageEditor(
                    label="Draw defects on the image",
                    type="numpy",
                    height=400,
                    brush=gr.Brush(
                        default_size=5,
                        colors=["#000000", "#FF0000", "#8B4513", "#808080", "#FFFFFF"],
                        default_color="#000000",
                        color_mode="fixed"
                    ),
                    eraser=gr.Eraser(default_size=10),
                    layers=False,  # Simple mode without layers
                )
                
                with gr.Row():
                    sketch_model_dropdown = gr.Dropdown(
                        choices=available_models,
                        value=default_model,
                        label="🧠 Model",
                        interactive=True
                    )
                    sketch_category_dropdown = gr.Dropdown(
                        choices=initial_categories,
                        value=initial_categories[0] if initial_categories else "bottle",
                        label="📦 Category",
                        interactive=True
                    )
                
                sketch_analyze_btn = gr.Button(
                    "🔬 Analyze Drawn Defects",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown(BRUSH_COLORS_INFO)
            
            # Right column - Results
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Detection Results")
                
                sketch_result_image = gr.Image(
                    label="Heatmap Visualization",
                    type="numpy",
                    height=400
                )
                
                sketch_result_text = gr.Markdown(value="", label="Results")
                
                gr.Markdown(HEATMAP_INTERPRETATION)
    
    return {
        "image_editor": image_editor,
        "model_dropdown": sketch_model_dropdown,
        "category_dropdown": sketch_category_dropdown,
        "analyze_btn": sketch_analyze_btn,
        "result_image": sketch_result_image,
        "result_image": sketch_result_image,
        "result_text": sketch_result_text,
    }
