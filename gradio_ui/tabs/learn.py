"""
Learn About Models Tab component.

Provides educational content about anomaly detection models.
"""

import gradio as gr

from gradio_ui.content import LEARN_ABOUT_MODELS_CONTENT


def create_learn_tab():
    """
    Creates the Learn About Models tab.
    
    Returns:
        None (no components need event wiring)
    """
    with gr.TabItem("📚 Learn About Models"):
        gr.Markdown(LEARN_ABOUT_MODELS_CONTENT)
    
    return None
