"""Gradio interface for the music generation app."""

import gradio as gr
from pathlib import Path
from typing import List, Tuple

from app.config import (
    APP_TITLE,
    APP_DESCRIPTION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_REPETITION_WINDOW,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_TEMPO,
    DEFAULT_NUM_SAMPLES,
    MAX_SAMPLES,
    MIN_TEMPERATURE,
    MAX_TEMPERATURE,
    MIN_REPETITION_PENALTY,
    MAX_REPETITION_PENALTY,
    TEMPO_CHOICES,
)
from app.inference import get_generator


def generate_music(
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    repetition_window: int,
    no_repeat_ngram_size: int,
    tempo: str,
    seed: int,
) -> Tuple[str, List[str]]:
    """
    Generate music and return status message and file paths.
    
    Args:
        num_samples: Number of samples to generate
        max_new_tokens: Max tokens per sample
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
        repetition_penalty: Repetition penalty
        repetition_window: Repetition window
        no_repeat_ngram_size: N-gram blocking size
        tempo: Tempo control
        seed: Random seed
    
    Returns:
        Tuple of (status message, list of file paths)
    """
    try:
        # Validate inputs
        num_samples = max(1, min(num_samples, MAX_SAMPLES))
        max_new_tokens = max(50, min(max_new_tokens, 2048))
        temperature = max(MIN_TEMPERATURE, min(temperature, MAX_TEMPERATURE))
        top_p = max(0.0, min(top_p, 1.0))
        repetition_penalty = max(MIN_REPETITION_PENALTY, min(repetition_penalty, MAX_REPETITION_PENALTY))
        no_repeat_ngram_size = max(0, min(no_repeat_ngram_size, 10))

        # Get generator and generate
        generator = get_generator()
        output_files = generator.generate(
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            no_repeat_ngram_size=no_repeat_ngram_size,
            tempo=tempo,
            seed=seed,
        )

        if not output_files:
            return "❌ Generation failed: No valid samples produced.", []

        status_msg = f"✅ Successfully generated {len(output_files)} sample(s)!\n\n"
        status_msg += f"📊 Parameters:\n"
        status_msg += f"  • Temperature: {temperature:.2f}\n"
        status_msg += f"  • Top-P: {top_p:.2f}\n"
        status_msg += f"  • Repetition Penalty: {repetition_penalty:.2f}\n"
        status_msg += f"  • Tempo: {tempo}\n"
        status_msg += f"  • Max Tokens: {max_new_tokens}\n\n"
        status_msg += f"📁 Files saved to: {output_files[0].rsplit('/', 1)[0]}"

        return status_msg, output_files

    except Exception as e:
        error_msg = f"❌ Error during generation: {str(e)}"
        return error_msg, []


def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface."""
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as interface:
        # Header
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Generation Parameters")

                # Basic controls
                num_samples = gr.Slider(
                    label="Number of Samples",
                    minimum=1,
                    maximum=MAX_SAMPLES,
                    value=DEFAULT_NUM_SAMPLES,
                    step=1,
                    info="How many MIDI files to generate",
                )

                max_new_tokens = gr.Slider(
                    label="Max Tokens to Generate",
                    minimum=50,
                    maximum=2048,
                    value=DEFAULT_MAX_TOKENS,
                    step=50,
                    info="Longer sequences = longer music",
                )

                # Sampling parameters
                gr.Markdown("#### Sampling Controls")

                temperature = gr.Slider(
                    label="Temperature",
                    minimum=MIN_TEMPERATURE,
                    maximum=MAX_TEMPERATURE,
                    value=DEFAULT_TEMPERATURE,
                    step=0.05,
                    info="Lower = more deterministic, Higher = more creative",
                )

                top_p = gr.Slider(
                    label="Top-P (Nucleus Sampling)",
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_TOP_P,
                    step=0.05,
                    info="Controls diversity of token selection",
                )

                # Music-specific controls
                gr.Markdown("#### Music Controls")

                tempo = gr.Radio(
                    choices=TEMPO_CHOICES,
                    value=DEFAULT_TEMPO,
                    label="Tempo",
                    info="Adjust pacing of generated music",
                )

                # Advanced controls (collapsed)
                with gr.Accordion("Advanced Controls", open=False):
                    repetition_penalty = gr.Slider(
                        label="Repetition Penalty",
                        minimum=MIN_REPETITION_PENALTY,
                        maximum=MAX_REPETITION_PENALTY,
                        value=DEFAULT_REPETITION_PENALTY,
                        step=0.05,
                        info="Higher = less repetition",
                    )

                    repetition_window = gr.Slider(
                        label="Repetition Window",
                        minimum=32,
                        maximum=512,
                        value=DEFAULT_REPETITION_WINDOW,
                        step=32,
                        info="Context for repetition tracking",
                    )

                    no_repeat_ngram_size = gr.Slider(
                        label="No-Repeat N-gram Size",
                        minimum=0,
                        maximum=10,
                        value=DEFAULT_NO_REPEAT_NGRAM_SIZE,
                        step=1,
                        info="Block repeated n-grams (0 = disabled)",
                    )

                    top_k = gr.Slider(
                        label="Top-K",
                        minimum=0,
                        maximum=100,
                        value=DEFAULT_TOP_K,
                        step=1,
                        info="Sample only from top K tokens (0 = disabled)",
                    )

                    seed = gr.Number(
                        label="Random Seed",
                        value=42,
                        precision=0,
                        info="Set for reproducible results",
                    )

            with gr.Column(scale=1):
                gr.Markdown("### 📝 Output")

                # Generate button
                generate_btn = gr.Button("🎵 Generate Music", variant="primary", size="lg")

                # Status message
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=5,
                    placeholder="Click 'Generate Music' to start...",
                )

                # File outputs
                gr.Markdown("#### 📥 Download Generated MIDI Files")
                file_output = gr.File(
                    label="Generated MIDI Files",
                    file_count="multiple",
                    type="filepath",
                )

        # Connect generate button
        generate_btn.click(
            generate_music,
            inputs=[
                num_samples,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                repetition_window,
                no_repeat_ngram_size,
                tempo,
                seed,
            ],
            outputs=[status_output, file_output],
        )

        # Footer
        gr.Markdown(
            """
            ---
            **Notes:**
            - Generation runs on CPU by default (slower but works everywhere)
            - MIDI files are saved and available for download
            - Adjust parameters to explore different musical styles
            """
        )

    return interface


def launch_app(share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860) -> None:
    """Launch the Gradio app."""
    interface = create_interface()
    interface.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True,
    )
