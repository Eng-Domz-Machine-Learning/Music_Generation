---
title: Piano Music Generation with GPT
emoji: 🎵
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Piano Music Generation with GPT

Generate expressive symbolic piano music as MIDI using a retrained GPT model and a clean Gradio interface.

This Space is focused on controllable generation: you can adjust creativity, repetition behavior, structure length, and tempo bias from the UI, then download the generated `.mid` files immediately.

## What This Space Does

- Generates one or more MIDI piano samples from a BOS prompt.
- Uses GPT token sampling with:
  - temperature
  - top-k
  - top-p (nucleus)
  - repetition penalty + window
  - no-repeat n-gram filtering
- Applies music-aware constraints during generation (for better event legality).
- Converts generated event tokens into playable MIDI notes.

## Current Model

- Architecture: Decoder-only GPT (`MusicGPT`)
- Config:
  - `d_model=512`
  - `n_layers=8`
  - `n_heads=8`
  - `max_seq_len=256`
  - `vocab_size=288`
- Parameters: ~25.63M
- Checkpoint used by app: `models/gpt/best.pt`

## Tokenization Scheme

Event-based vocabulary (`288` tokens):

- Special: `PAD`, `BOS`, `EOS`, `UNK`
- Velocity bins: 8
- Time shifts: `TIME_SHIFT_1..100` (0.125s quantization)
- Note on/off events for piano range: MIDI pitches `21..108`

## How To Use

1. Set generation controls in the left panel.
2. Click Generate Music.
3. Wait for inference to finish.
4. Download generated MIDI files from the output panel.

## Parameter Guide

- Number of Samples (`1-5`): More files per run.
- Max Tokens (`50-2048`): Larger values usually produce longer pieces.
- Temperature (`0.1-2.0`):
  - lower = safer, more deterministic
  - higher = more varied and exploratory
- Top-P (`0.0-1.0`): Limits sampling to high-probability mass.
- Top-K (`0-100`): Limits sampling to top-k candidates.
- Repetition Penalty (`1.0-2.0`): Reduces repetitive loops.
- Repetition Window (`32-512`): Context used to track repeats.
- No-Repeat N-gram (`0-10`): Blocks repeated n-gram patterns.
- Tempo (`slow`, `normal`, `fast`): Biases time-shift behavior.
- Seed: Reproducibility control.

## Runtime Notes

- Runs on CPU by default in Spaces.
- First request may be slower because model initialization happens on load.
- Output files are generated as MIDI and can be opened in any DAW or MIDI player.

## Repository Layout (Inference-Critical)

- `app.py`: Space entrypoint
- `app/`: interface + inference wrapper
- `src/train_gpt_improved.py`: model definition used for checkpoint loading
- `src/generate_gpt.py`: generation + token-to-note logic
- `models/gpt/best.pt`: inference checkpoint
- `tokenized/id_to_token.json`: ID -> token mapping
- `tokenized/vocab.json`: token -> ID mapping

## Local Run (Optional)

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Dependencies

- torch
- gradio
- pretty_midi

## Limitations

- This is symbolic generation (MIDI), not raw audio waveform synthesis.
- Musical quality depends on sampling settings and prompt conditions.
- Very long generations can still drift structurally, even with repetition controls.

## Acknowledgment

Built for deep learning-based symbolic music generation and iterative model improvement with practical inference controls.
