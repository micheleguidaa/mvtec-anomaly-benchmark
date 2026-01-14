"""
Gradio UI tab components.

Each tab is defined in its own module for better organization.
"""

from gradio_ui.tabs.upload import create_upload_tab
from gradio_ui.tabs.draw_defects import create_draw_defects_tab
from gradio_ui.tabs.compare import create_compare_tab
from gradio_ui.tabs.learn import create_learn_tab
from gradio_ui.tabs.metrics import create_metrics_tab

__all__ = [
    "create_upload_tab",
    "create_draw_defects_tab",
    "create_compare_tab",
    "create_learn_tab",
    "create_metrics_tab",
]
