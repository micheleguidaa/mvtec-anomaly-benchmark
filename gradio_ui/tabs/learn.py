"""
Learn About Models Tab component.

Provides educational content about anomaly detection models,
loading content from presentazione.md when available.
"""

import gradio as gr
from pathlib import Path

from gradio_ui.content import LEARN_ABOUT_MODELS_CONTENT

# Path to the presentation markdown file (relative to project root)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PRESENTATION_PATH = _PROJECT_ROOT / "presentazione.md"


def _load_presentation_content() -> str:
    """
    Load content from presentazione.md if available, otherwise fallback.
    
    Returns:
        Markdown content string
    """
    try:
        if _PRESENTATION_PATH.exists():
            return _PRESENTATION_PATH.read_text(encoding="utf-8")
    except Exception:
        pass
    # Fallback to built-in content
    return LEARN_ABOUT_MODELS_CONTENT


# Custom CSS for scrollable markdown container
_LEARN_TAB_CSS = """
<style>
.learn-content-container {
    max-height: 80vh;
    overflow-y: auto;
    padding-right: 10px;
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 20px;
    background: var(--background-fill-secondary);
}
.learn-content-container::-webkit-scrollbar {
    width: 8px;
}
.learn-content-container::-webkit-scrollbar-thumb {
    background: var(--neutral-400);
    border-radius: 4px;
}
.learn-content-container::-webkit-scrollbar-track {
    background: var(--background-fill-primary);
}
</style>
"""


def create_learn_tab():
    """
    Creates the Learn About Models tab.
    
    Loads content from presentazione.md at runtime for easy updates.
    Falls back to built-in content if the file is not available.
    
    Returns:
        None (no components need event wiring)
    """
    content = _load_presentation_content()
    
    with gr.TabItem("📚 Learn About Models"):
        # Inject custom CSS + wrap content in scrollable div
        gr.HTML(_LEARN_TAB_CSS)
        gr.Markdown(
            f'<div class="learn-content-container">\n\n{content}\n\n</div>',
            latex_delimiters=[
                {"left": "$$", "right": "$$", "display": True},
                {"left": "$", "right": "$", "display": False},
            ],
        )
    
    return None
