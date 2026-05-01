"""Hugging Face Spaces entry point for music generation app."""

import sys
from pathlib import Path

# Set up paths for HF Spaces environment
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.interface import create_interface

if __name__ == "__main__":
    print("🎵 Starting Music Generation Gradio App on Hugging Face Spaces...")
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # HF Spaces handles sharing
        show_error=True,
    )
