"""
LSTM Generation Script for Symbolic MIDI Music Generation

Generate MIDI sequences using a trained LSTM model.

Features:
- Load trained LSTM checkpoint
- Generate from BOS or custom prompt
- Temperature sampling
- Top-k sampling
- Max new tokens control
- Token legality safeguards
- Save generated sequences as MIDI files
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Prevent local tokenizer scripts from shadowing stdlib tokenize.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pretty_midi

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import the model and config from train_lstm
from train_lstm import MusicLSTM, LSTMConfig


# ---------------------------------------------------------------------------
# Token ID constants (must match midi_tokenize.py)
# ---------------------------------------------------------------------------

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

# Token type ranges (from vocab structure)
VELOCITY_BIN_MIN = 4
VELOCITY_BIN_MAX = 11  # 8 velocity bins: 4-11
TIME_SHIFT_MIN = 12
TIME_SHIFT_MAX = 111  # TIME_SHIFT_1 to TIME_SHIFT_100
NOTE_ON_MIN = 112
NOTE_ON_MAX = 199  # NOTE_ON_21 to NOTE_ON_108
NOTE_OFF_MIN = 200
NOTE_OFF_MAX = 287  # NOTE_OFF_21 to NOTE_OFF_108

PITCH_MIN = 21
PITCH_MAX = 108
NUM_VELOCITY_BINS = 8
TIME_STEP_SECONDS = 0.125


# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------

def load_id_to_token(path: Path) -> dict:
    """Load id_to_token mapping from JSON."""
    with path.open("r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_token_to_id(path: Path) -> dict:
    """Load token_to_id mapping from JSON."""
    with path.open("r", encoding="utf-8") as f:
        return {k: int(v) for k, v in json.load(f).items()}


def id_to_token_string(token_id: int, id_to_token: dict) -> str:
    """Convert token ID to human-readable string."""
    return id_to_token.get(token_id, "UNK")


def token_string_to_id(token_str: str, token_to_id: dict) -> int:
    """Convert token string to ID."""
    return token_to_id.get(token_str, UNK_TOKEN)


def is_valid_note_on(token_id: int) -> bool:
    """Check if token ID is a valid NOTE_ON token."""
    return NOTE_ON_MIN <= token_id <= NOTE_ON_MAX


def is_valid_note_off(token_id: int) -> bool:
    """Check if token ID is a valid NOTE_OFF token."""
    return NOTE_OFF_MIN <= token_id <= NOTE_OFF_MAX


def get_pitch_from_note_token(token_id: int) -> Optional[int]:
    """Extract pitch from NOTE_ON or NOTE_OFF token."""
    if is_valid_note_on(token_id) or is_valid_note_off(token_id):
        # Token ID = 112 + (pitch - 21) for NOTE_ON
        # Token ID = 200 + (pitch - 21) for NOTE_OFF
        if is_valid_note_on(token_id):
            return (token_id - NOTE_ON_MIN) + PITCH_MIN
        else:
            return (token_id - NOTE_OFF_MIN) + PITCH_MIN
    return None


def get_velocity_bin(token_id: int) -> Optional[int]:
    """Extract velocity bin from VELOCITY token."""
    if VELOCITY_BIN_MIN <= token_id <= VELOCITY_BIN_MAX:
        return token_id - VELOCITY_BIN_MIN
    return None


def velocity_bin_to_midi(velocity_bin: int) -> int:
    """Convert velocity bin (0-7) to MIDI velocity (1-127)."""
    velocity_bin = max(0, min(NUM_VELOCITY_BINS - 1, velocity_bin))
    # Map bin center to MIDI velocity
    return int((velocity_bin + 0.5) * (127.0 / NUM_VELOCITY_BINS))


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits."""
    if temperature <= 0:
        # Argmax sampling (greedy)
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering: set all but top k logits to -inf."""
    if k <= 0:
        return logits

    top_k_values = torch.topk(logits, k, dim=-1)[0]
    min_top_k = top_k_values[..., -1:] if k > 0 else logits
    filtered = torch.where(logits >= min_top_k, logits, torch.full_like(logits, float("-inf")))
    return filtered


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    generator: Optional[torch.Generator] = None,
) -> int:
    """
    Sample a token ID from logits.

    Args:
        logits: (vocab_size,) tensor of unnormalized log-probabilities
        temperature: Sampling temperature (lower = more deterministic)
        top_k: If > 0, only sample from top k tokens
        generator: Optional torch.Generator for reproducibility

    Returns:
        Sampled token ID
    """
    # Apply temperature
    if temperature > 0:
        logits = logits / temperature

    # Apply top-k filtering
    if top_k > 0:
        logits = apply_top_k(logits, top_k)

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Sample
    return torch.multinomial(probs, num_samples=1, generator=generator).item()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: nn.Module,
    device: torch.device,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float = 0.3,
    top_k: int = 0,
    id_to_token: Optional[dict] = None,
    token_to_id: Optional[dict] = None,
    seed: int = 42,
) -> List[int]:
    """
    Generate a sequence of token IDs using the trained LSTM model.

    Args:
        model: Trained LSTM model
        device: torch device
        prompt_ids: Initial token IDs (can be [BOS] or a custom prompt)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        id_to_token: Mapping for token legality checks
        token_to_id: Mapping for token legality checks
        seed: Random seed for reproducibility

    Returns:
        Generated token IDs (including prompt)
    """
    model.eval()

    # Set up random generator
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    random.seed(seed)

    # Initialize with prompt
    generated = prompt_ids.copy()
    input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    # Track active notes for basic legality checks
    active_pitches: set = set()  # Pitches with active NOTE_ON

    for _ in range(max_new_tokens):
        # Forward pass through LSTM
        # LSTM returns (batch, seq, vocab) logits
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, hidden = model(input_ids)

        # Get logits for next token (last position)
        next_token_logits = logits[0, -1, :]  # (vocab_size,)

        # Apply token legality constraints
        if id_to_token is not None and token_to_id is not None:
            # Mask out invalid token IDs (shouldn't happen but be safe)
            vocab_size = next_token_logits.shape[0]
            for token_id in range(vocab_size):
                if token_id > 287:  # Beyond valid vocabulary
                    next_token_logits[token_id] = float("-inf")

                # Optional: discourage NOTE_OFF for pitches not currently active
                # This is a soft constraint (logit penalty, not hard mask)
                if is_valid_note_off(token_id):
                    pitch = get_pitch_from_note_token(token_id)
                    if pitch is not None and pitch not in active_pitches:
                        next_token_logits[token_id] -= 2.0  # Penalty

        # Sample next token
        next_token_id = sample_from_logits(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )

        # Update active pitches tracking
        if is_valid_note_on(next_token_id):
            pitch = get_pitch_from_note_token(next_token_id)
            if pitch is not None:
                active_pitches.add(pitch)
        elif is_valid_note_off(next_token_id):
            pitch = get_pitch_from_note_token(next_token_id)
            if pitch is not None and pitch in active_pitches:
                active_pitches.discard(pitch)

        # Append generated token
        generated.append(next_token_id)

        # Check for EOS
        if next_token_id == EOS_TOKEN:
            break

        # Update input for next step
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    return generated


def tokens_to_notes(token_ids: List[int], id_to_token: dict) -> List[pretty_midi.Note]:
    """
    Convert token IDs to pretty_midi Note objects.

    This decodes the event-based representation back to notes.
    """
    notes = []
    current_step = 0
    current_velocity_bin = 0
    active_notes: dict = {}  # pitch -> (start_step, velocity_bin)

    for token_id in token_ids:
        token_str = id_to_token.get(token_id, "UNK")

        if token_str in ("PAD", "BOS", "EOS"):
            continue

        if token_str.startswith("TIME_SHIFT_"):
            shift = int(token_str.split("_")[-1])
            current_step += shift
            continue

        if token_str.startswith("VELOCITY_"):
            current_velocity_bin = int(token_str.split("_")[-1])
            continue

        if token_str.startswith("NOTE_ON_"):
            pitch = int(token_str.split("_")[-1])
            if PITCH_MIN <= pitch <= PITCH_MAX:
                active_notes[pitch] = (current_step, current_velocity_bin)
            continue

        if token_str.startswith("NOTE_OFF_"):
            pitch = int(token_str.split("_")[-1])
            if pitch in active_notes:
                start_step, velocity_bin = active_notes.pop(pitch)
                if current_step > start_step:
                    velocity = velocity_bin_to_midi(velocity_bin)
                    start_time = start_step * TIME_STEP_SECONDS
                    end_time = current_step * TIME_STEP_SECONDS
                    notes.append(pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=end_time,
                    ))
            continue

    return notes


def save_midi(notes: List[pretty_midi.Note], output_path: Path) -> None:
    """Save notes as a MIDI file."""
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, is_drum=False)
    instrument.notes = notes
    pm.instruments.append(instrument)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(output_path))
    print(f"Saved MIDI: {output_path}", flush=True)


def load_checkpoint(checkpoint_path: Path) -> Tuple[nn.Module, dict]:
    """Load model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # Get config from checkpoint
    config_dict = checkpoint.get("config", {})
    cfg = LSTMConfig(**config_dict) if config_dict else LSTMConfig()

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicLSTM(cfg).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state"])

    return model, checkpoint


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MIDI sequences with trained LSTM.")

    # Model checkpoint
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt file)")
    p.add_argument("--vocab-path", default="tokenized/id_to_token.json", help="Path to id_to_token.json")

    # Generation parameters
    p.add_argument("--prompt", type=str, default=None, help="Optional prompt text (token names separated by commas)")
    p.add_argument("--prompt-file", type=str, default=None, help="File containing prompt token IDs (one per line)")
    p.add_argument("--max-new-tokens", type=int, default=512, help="Maximum tokens to generate")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    p.add_argument("--output-dir", type=str, default="generated/lstm", help="Output directory for MIDI files")
    p.add_argument("--num-samples", type=int, default=5, help="Number of samples to generate")
    p.add_argument("--save-tokens", action="store_true", help="Also save generated token IDs as JSON")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Load vocabulary mappings
    vocab_dir = Path(args.vocab_path).parent
    id_to_token_path = vocab_dir / "id_to_token.json"
    token_to_id_path = vocab_dir / "vocab.json"

    if not id_to_token_path.exists():
        id_to_token_path = Path(args.vocab_path)
    if not token_to_id_path.exists():
        token_to_id_path = vocab_dir / "vocab.json"

    id_to_token = load_id_to_token(id_to_token_path)
    token_to_id = load_token_to_id(token_to_id_path)
    vocab_size = len(id_to_token)
    print(f"Vocabulary size: {vocab_size}", flush=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model, checkpoint = load_checkpoint(Path(args.checkpoint))
    model.to(device)
    model.eval()

    loaded_step = checkpoint.get("step", "unknown")
    print(f"Loaded checkpoint from step {loaded_step}", flush=True)

    # Prepare prompt
    if args.prompt_file and Path(args.prompt_file).exists():
        with open(args.prompt_file, "r") as f:
            prompt_ids = [int(line.strip()) for line in f if line.strip()]
        print(f"Loaded prompt from file: {len(prompt_ids)} tokens", flush=True)
    elif args.prompt:
        prompt_tokens = [t.strip() for t in args.prompt.split(",")]
        prompt_ids = [token_to_id.get(t, UNK_TOKEN) for t in prompt_tokens]
        print(f"Using text prompt: {prompt_tokens}", flush=True)
    else:
        # Default: start with BOS token
        prompt_ids = [BOS_TOKEN]
        print("Using default BOS prompt", flush=True)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples
    print("=" * 60, flush=True)
    print(f"Generating {args.num_samples} samples...", flush=True)
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  temperature: {args.temperature}")
    print(f"  top_k: {args.top_k}")
    print(f"  seed: {args.seed}")
    print("=" * 60, flush=True)

    for sample_idx in range(args.num_samples):
        sample_seed = args.seed + sample_idx

        # Generate tokens
        generated_ids = generate(
            model=model,
            device=device,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            id_to_token=id_to_token,
            token_to_id=token_to_id,
            seed=sample_seed,
        )

        print(f"\nSample {sample_idx + 1}/{args.num_samples}:")
        print(f"  Generated {len(generated_ids)} tokens", flush=True)

        # Decode to notes
        try:
            notes = tokens_to_notes(generated_ids, id_to_token)
            print(f"  Decoded to {len(notes)} notes", flush=True)

            # Save MIDI
            midi_path = output_dir / f"sample_{sample_idx:03d}.mid"
            save_midi(notes, midi_path)
        except Exception as e:
            print(f"  Error decoding to MIDI: {e}", flush=True)

        # Save token IDs if requested
        if args.save_tokens:
            tokens_path = output_dir / f"sample_{sample_idx:03d}_tokens.json"
            with tokens_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "prompt_length": len(prompt_ids),
                    "generated_length": len(generated_ids),
                    "token_ids": generated_ids,
                    "tokens": [id_to_token.get(tid, "UNK") for tid in generated_ids],
                    "config": {
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "seed": sample_seed,
                        "max_new_tokens": args.max_new_tokens,
                    },
                }, f, indent=2)
            print(f"  Saved tokens: {tokens_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Generation complete!", flush=True)
    print(f"Output directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
