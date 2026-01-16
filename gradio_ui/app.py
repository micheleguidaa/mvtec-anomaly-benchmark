"""
Main Gradio app assembly.

This module creates and configures the complete Gradio application
by assembling all tab components and wiring event handlers.
"""

import gradio as gr
import numpy as np
from PIL import Image

from core import get_available_models, MVTEC_CATEGORIES
from gradio_ui.handlers import (
    get_trained_categories,
    predict,
    predict_sketch,
    predict_compare,
    update_categories,
    update_compare_categories,
    get_category_from_sample,
)
from gradio_ui.content import MAIN_HEADER
from gradio_ui.tabs import (
    create_upload_tab,
    create_draw_defects_tab,
    create_compare_tab,
    create_learn_tab,
    create_metrics_tab,
)


def create_app() -> gr.Blocks:
    """
    Creates and returns the Gradio app.
    
    Returns:
        Configured Gradio Blocks application
    """
    available_models = get_available_models()
    default_model = available_models[0] if available_models else "patchcore"
    initial_categories = get_trained_categories(default_model) or MVTEC_CATEGORIES
    
    with gr.Blocks(title="MVTec Anomaly Detection Demo") as app:
        
        # Header
        gr.Markdown(MAIN_HEADER, elem_classes="main-header")
        
        with gr.Tabs():
            # Tab 1: Upload Image
            upload_components = create_upload_tab(
                available_models, default_model, initial_categories
            )
            
            # Tab 2: Draw Defects
            sketch_components = create_draw_defects_tab(
                available_models, default_model, initial_categories
            )
            
            # Tab 3: Compare Models
            compare_components = create_compare_tab(
                available_models, initial_categories
            )
            
            # Tab 4: Learn About Models
            create_learn_tab()
            
            # Tab 5: Model Metrics (self-contained event handlers)
            create_metrics_tab()
        
        # =========================================================================
        # Event handlers for Tab 1 (Upload)
        # =========================================================================
        upload_components["model_dropdown"].change(
            fn=update_categories,
            inputs=[upload_components["model_dropdown"], upload_components["category_dropdown"]],
            outputs=[upload_components["category_dropdown"]],
            show_progress="hidden"
        )
        
        upload_components["analyze_btn"].click(
            fn=predict,
            inputs=[
                upload_components["image_input"],
                upload_components["model_dropdown"],
                upload_components["category_dropdown"]
            ],
            outputs=[
                upload_components["result_image"],
                upload_components["result_text"]
            ]
        )
        
        # Sample gallery selection handler for Upload tab
        def on_upload_sample_select(evt: gr.SelectData, current_model: str, gallery_value: list):
            """Handler for sample image selection in upload tab."""
            if evt is None:
                return None, gr.Dropdown()
            
            # Get the selected index and retrieve original path from gallery data
            idx = evt.index
            if gallery_value and idx < len(gallery_value):
                item = gallery_value[idx]
                if isinstance(item, tuple):
                    image_path, label = item
                    category = get_category_from_sample(label)
                else:
                    image_path = item
                    category = get_category_from_sample(str(image_path))
            else:
                return None, gr.Dropdown()
            
            try:
                img = np.array(Image.open(image_path).convert("RGB"))
            except Exception as e:
                print(f"Error loading image: {e}")
                return None, gr.Dropdown()
            
            trained = get_trained_categories(current_model)
            if trained and category in trained:
                return img, gr.Dropdown(choices=trained, value=category)
            elif trained:
                return img, gr.Dropdown(choices=trained, value=trained[0])
            return img, gr.Dropdown(choices=MVTEC_CATEGORIES, value=category)
        
        upload_components["sample_gallery"].select(
            fn=on_upload_sample_select,
            inputs=[upload_components["model_dropdown"], upload_components["sample_gallery"]],
            outputs=[upload_components["image_input"], upload_components["category_dropdown"]]
        )
        
        # =========================================================================
        # Event handlers for Tab 2 (Sketch)
        # =========================================================================
        sketch_components["model_dropdown"].change(
            fn=update_categories,
            inputs=[sketch_components["model_dropdown"], sketch_components["category_dropdown"]],
            outputs=[sketch_components["category_dropdown"]],
            show_progress="hidden"
        )
        
        sketch_components["analyze_btn"].click(
            fn=predict_sketch,
            inputs=[
                sketch_components["image_editor"],
                sketch_components["model_dropdown"],
                sketch_components["category_dropdown"]
            ],
            outputs=[
                sketch_components["result_image"],
                sketch_components["result_text"]
            ]
        )
        
        # Sample gallery selection handler for Sketch tab
        def on_sketch_sample_select(evt: gr.SelectData, current_model: str, gallery_value: list):
            """Handler for sample image selection in sketch tab."""
            if evt is None:
                return None, gr.Dropdown()
            
            # Get the selected index and retrieve original path from gallery data
            idx = evt.index
            if gallery_value and idx < len(gallery_value):
                item = gallery_value[idx]
                if isinstance(item, tuple):
                    image_path, label = item
                    category = get_category_from_sample(label)
                else:
                    image_path = item
                    category = get_category_from_sample(str(image_path))
            else:
                return None, gr.Dropdown()
            
            try:
                img = np.array(Image.open(image_path).convert("RGB"))
                # For ImageEditor, return the image in the expected format
                editor_value = {"background": img, "layers": [], "composite": img}
            except Exception as e:
                print(f"Error loading image: {e}")
                return None, gr.Dropdown()
            
            trained = get_trained_categories(current_model)
            if trained and category in trained:
                return editor_value, gr.Dropdown(choices=trained, value=category)
            elif trained:
                return editor_value, gr.Dropdown(choices=trained, value=trained[0])
            return editor_value, gr.Dropdown(choices=MVTEC_CATEGORIES, value=category)
        
        sketch_components["sample_gallery"].select(
            fn=on_sketch_sample_select,
            inputs=[sketch_components["model_dropdown"], sketch_components["sample_gallery"]],
            outputs=[sketch_components["image_editor"], sketch_components["category_dropdown"]]
        )
        
        # =========================================================================
        # Event handlers for Tab 3 (Compare)
        # =========================================================================
        compare_components["model_checkboxes"].change(
            fn=update_compare_categories,
            inputs=[compare_components["model_checkboxes"], compare_components["category_dropdown"]],
            outputs=[compare_components["category_dropdown"]],
            show_progress="hidden"
        )
        
        compare_components["compare_btn"].click(
            fn=predict_compare,
            inputs=[
                compare_components["image_input"],
                compare_components["model_checkboxes"],
                compare_components["category_dropdown"]
            ],
            outputs=[
                compare_components["result_image"],
                compare_components["summary"]
            ]
        )
        
        # Sample gallery selection handler for Compare tab
        def on_compare_sample_select(evt: gr.SelectData, selected_models: list, gallery_value: list):
            """Handler for sample image selection in compare tab."""
            if evt is None:
                return None, gr.Dropdown()
            
            # Get the selected index and retrieve original path from gallery data
            idx = evt.index
            if gallery_value and idx < len(gallery_value):
                item = gallery_value[idx]
                if isinstance(item, tuple):
                    image_path, label = item
                    category = get_category_from_sample(label)
                else:
                    image_path = item
                    category = get_category_from_sample(str(image_path))
            else:
                return None, gr.Dropdown()
            
            try:
                img = np.array(Image.open(image_path).convert("RGB"))
            except Exception as e:
                print(f"Error loading image: {e}")
                return None, gr.Dropdown()
            
            # Get common categories for selected models
            from gradio_ui.handlers import get_common_categories
            common = get_common_categories(selected_models)
            if common and category in common:
                return img, gr.Dropdown(choices=common, value=category)
            elif common:
                return img, gr.Dropdown(choices=common, value=common[0])
            return img, gr.Dropdown(choices=MVTEC_CATEGORIES, value=category)
        
        compare_components["sample_gallery"].select(
            fn=on_compare_sample_select,
            inputs=[compare_components["model_checkboxes"], compare_components["sample_gallery"]],
            outputs=[compare_components["image_input"], compare_components["category_dropdown"]]
        )
    
    return app


def launch_app(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = True,
    show_error: bool = True
):
    """
    Creates and launches the Gradio app.
    
    Args:
        server_name: Server hostname (default: "0.0.0.0" for external access)
        server_port: Server port (default: 7860)
        share: Whether to create a public link (default: True)
        show_error: Whether to show errors in UI (default: True)
    """
    app = create_app()
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=show_error
    )
