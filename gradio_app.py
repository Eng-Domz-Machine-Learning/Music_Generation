#!/usr/bin/env python3
"""
Gradio app entry point for music generation.

Run with: python gradio_app.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.interface import launch_app


if __name__ == "__main__":
    print("🎵 Starting Music Generation Gradio App...")
    print("📍 Open http://localhost:7860 in your browser")
    print("Press Ctrl+C to stop the app\n")

    launch_app(share=True)  # Creates public shareable link
