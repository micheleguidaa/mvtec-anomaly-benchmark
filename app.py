#!/usr/bin/env python3
"""
Gradio Demo for MVTec Anomaly Detection.

Provides an interactive interface to test anomaly detection models
on uploaded images.

This is the main entry point. The actual implementation is in the
`gradio_ui` package for better code organization.

Run with: python app.py
"""

from gradio_ui import create_app, launch_app



if __name__ == "__main__":
    # Check for checkpoints and download if missing
    import os
    import subprocess
    import sys
    
    print("🔄 Checking for model checkpoints...")
    subprocess.run([sys.executable, "scripts/download_checkpoints.py"], check=False)

    launch_app(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=True,  # Public link for remote access
        show_error=True
    )
