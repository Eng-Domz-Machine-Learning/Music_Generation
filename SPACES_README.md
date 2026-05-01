# 🎵 Piano Music Generation with GPT

Generate symbolic MIDI piano music using a GPT-based deep learning model trained on classical piano compositions from the MAESTRO dataset.

## Features

- **Advanced Sampling**: Top-p (nucleus) sampling, repetition penalty, n-gram filtering
- **Tempo Control**: Generate music at different tempos (slow, normal, fast)
- **Interactive UI**: Easy-to-use Gradio interface with customizable parameters
- **MIDI Download**: Generated samples available for immediate download

## How to Use

1. Adjust parameters on the left panel
2. Click "🎵 Generate Music"
3. Download generated MIDI files from the output
4. Use in your DAW or music player

## Parameters

- **Number of Samples**: 1-5 MIDI files per generation
- **Max Tokens**: Longer sequences = longer music (50-2048)
- **Temperature**: Lower = more consistent, Higher = more creative (0.1-2.0)
- **Top-P**: Nucleus sampling diversity (0.0-1.0)
- **Tempo**: Control pacing (slow/normal/fast)
- **Advanced**: Fine-tune repetition penalty, n-gram blocking, top-k sampling

## Model Details

- **Architecture**: 6-layer GPT decoder with 4 attention heads
- **Parameters**: 11.056M
- **Training Data**: MAESTRO v3.0 (classical piano)
- **Tokenization**: Event-based (288 vocabulary tokens)
  - Velocities: 8 bins
  - Time shifts: 100 bins (0.125s quantization)
  - Notes: 88 pitches (A0-C8)

## Technical Stack

- **PyTorch**: Deep learning framework
- **Gradio**: Web interface
- **pretty_midi**: MIDI I/O

---

**Note**: Generation runs on CPU (slower but works everywhere). Model loading takes ~10-20 seconds on first use.
