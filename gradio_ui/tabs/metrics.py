"""
Model Metrics Tab component.

Provides the training metrics and performance display interface.
"""

import gradio as gr

from inference import get_available_models, MVTEC_CATEGORIES
from gradio_ui.metrics import generate_metrics_table, get_metrics_summary
from gradio_ui.content import METRICS_HEADER, METRICS_EXPLANATION


def create_metrics_tab():
    """
    Creates the Model Metrics tab.
    
    Returns:
        Dict with tab components for wiring events
    """
    with gr.TabItem("📈 Model Metrics"):
        gr.Markdown(METRICS_HEADER)
        
        with gr.Row():
            metrics_model_filter = gr.Dropdown(
                choices=["All"] + get_available_models(),
                value="All",
                label="🧠 Filter by Model",
                interactive=True
            )
            metrics_category_filter = gr.Dropdown(
                choices=["All"] + MVTEC_CATEGORIES,
                value="All",
                label="📦 Filter by Category",
                interactive=True
            )
            refresh_metrics_btn = gr.Button(
                "🔄 Refresh",
                variant="secondary"
            )
        
        metrics_summary = gr.Markdown(get_metrics_summary())
        
        gr.Markdown("### 📊 All Metrics")
        metrics_table = gr.Markdown(generate_metrics_table())
        
        gr.Markdown(METRICS_EXPLANATION)
        
        # Event handlers for metrics tab
        def update_metrics(model_filter, category_filter):
            return generate_metrics_table(model_filter, category_filter)
        
        metrics_model_filter.change(
            fn=update_metrics,
            inputs=[metrics_model_filter, metrics_category_filter],
            outputs=[metrics_table]
        )
        
        metrics_category_filter.change(
            fn=update_metrics,
            inputs=[metrics_model_filter, metrics_category_filter],
            outputs=[metrics_table]
        )
        
        def refresh_all_metrics():
            return get_metrics_summary(), generate_metrics_table()
        
        refresh_metrics_btn.click(
            fn=refresh_all_metrics,
            inputs=[],
            outputs=[metrics_summary, metrics_table]
        )
    
    return {
        "model_filter": metrics_model_filter,
        "category_filter": metrics_category_filter,
        "refresh_btn": refresh_metrics_btn,
        "summary": metrics_summary,
        "table": metrics_table,
    }
