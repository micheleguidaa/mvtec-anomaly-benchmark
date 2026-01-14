"""
Main Gradio app assembly.

This module creates and configures the complete Gradio application
by assembling all tab components and wiring event handlers.
"""

import gradio as gr

from core import get_available_models, MVTEC_CATEGORIES
from gradio_ui.handlers import (
    get_trained_categories,
    predict,
    predict_sketch,
    predict_compare,
    update_categories,
    update_compare_categories,
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
            inputs=[upload_components["model_dropdown"]],
            outputs=[upload_components["category_dropdown"]]
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
        
        # Allow Enter key on image upload
        upload_components["image_input"].change(
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
        
        # =========================================================================
        # Event handlers for Tab 2 (Sketch)
        # =========================================================================
        sketch_components["model_dropdown"].change(
            fn=update_categories,
            inputs=[sketch_components["model_dropdown"]],
            outputs=[sketch_components["category_dropdown"]]
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
        
        # =========================================================================
        # Event handlers for Tab 3 (Compare)
        # =========================================================================
        compare_components["model_checkboxes"].change(
            fn=update_compare_categories,
            inputs=[compare_components["model_checkboxes"]],
            outputs=[compare_components["category_dropdown"]]
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
