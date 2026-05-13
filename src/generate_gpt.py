"""
GPT Generation Script for Symbolic MIDI Music Generation

Generate MIDI sequences using a trained GPT (decoder-only Transformer) model.

Features:
- Load trained GPT checkpoint
- Generate from BOS or custom prompt
- Temperature sampling
- Top-k sampling
- Top-p sampling
- Repetition control
- No-repeat n-gram filtering
- Tempo control (slow/normal/fast)
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
from typing import List, Optional

# Keep the script directory on sys.path so sibling modules like
# train_gpt_improved can be imported when this file is run as a script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pretty_midi

# Import the model and config from train_gpt_improved
from train_gpt_improved import MusicGPT, GPTConfig


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
PROJECT_ROOT = Path(SCRIPT_DIR).parent


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


def resolve_existing_path(path_str: str, extra_roots: Optional[List[Path]] = None) -> Path:
    """Resolve a path relative to common project roots and return the first existing match."""
    path = Path(path_str)
    candidates = [path]

    if not path.is_absolute():
        roots = [Path.cwd(), PROJECT_ROOT, Path(SCRIPT_DIR)]
        if extra_roots:
            roots.extend(extra_roots)
        for root in roots:
            candidates.append(root / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path


def resolve_vocab_paths(vocab_path: Path) -> tuple[Path, Path]:
    """Resolve matching vocab JSON paths from either side of the pair."""
    if vocab_path.name == "id_to_token.json":
        id_to_token_path = vocab_path
        token_to_id_path = vocab_path.with_name("vocab.json")
    elif vocab_path.name == "vocab.json":
        token_to_id_path = vocab_path
        id_to_token_path = vocab_path.with_name("id_to_token.json")
    else:
        id_to_token_path = vocab_path
        token_to_id_path = vocab_path.with_name("vocab.json")

    return id_to_token_path, token_to_id_path


def is_valid_note_on(token_id: int) -> bool:
    """Check if token ID is a valid NOTE_ON token."""
    return NOTE_ON_MIN <= token_id <= NOTE_ON_MAX


def is_valid_note_off(token_id: int) -> bool:
    """Check if token ID is a valid NOTE_OFF token."""
    return NOTE_OFF_MIN <= token_id <= NOTE_OFF_MAX


def get_pitch_from_note_token(token_id: int) -> Optional[int]:
    """Extract pitch from NOTE_ON or NOTE_OFF token."""
    if is_valid_note_on(token_id):
        return (token_id - NOTE_ON_MIN) + PITCH_MIN
    elif is_valid_note_off(token_id):
        return (token_id - NOTE_OFF_MIN) + PITCH_MIN
    return None


def velocity_bin_to_midi(velocity_bin: int) -> int:
    """Convert velocity bin (0-7) to MIDI velocity (1-127)."""
    velocity_bin = max(0, min(NUM_VELOCITY_BINS - 1, velocity_bin))
    return int((velocity_bin + 0.5) * (127.0 / NUM_VELOCITY_BINS))


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering: set all but top k logits to -inf."""
    if k <= 0:
        return logits

    k = min(k, logits.shape[-1])

    top_k_values = torch.topk(logits, k, dim=-1)[0]
    min_top_k = top_k_values[..., -1:] if k > 0 else logits
    filtered = torch.where(logits >= min_top_k, logits, torch.full_like(logits, float("-inf")))
    return filtered


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus sampling: keep the smallest set of tokens whose mass reaches top_p."""
    if top_p <= 0.0 or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    remove_mask = cumulative_probs > top_p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    filtered = logits.clone()
    filtered[sorted_indices[remove_mask]] = float("-inf")
    return filtered


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: List[int],
    penalty: float,
    window_size: int,
) -> torch.Tensor:
    """Penalize tokens that already appeared in the recent context window."""
    if penalty <= 1.0 or not generated_ids:
        return logits

    recent_tokens = generated_ids[-window_size:] if window_size > 0 else generated_ids
    adjusted = logits.clone()

    for token_id in set(recent_tokens):
        if 0 <= token_id < adjusted.shape[-1]:
            if adjusted[token_id] > 0:
                adjusted[token_id] /= penalty
            else:
                adjusted[token_id] *= penalty

    return adjusted


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    generated_ids: List[int],
    ngram_size: int,
) -> torch.Tensor:
    """Block tokens that would create an already-seen n-gram."""
    if ngram_size <= 1 or len(generated_ids) < ngram_size - 1:
        return logits

    prefix = tuple(generated_ids[-(ngram_size - 1):])
    banned_tokens = set()

    for index in range(len(generated_ids) - ngram_size + 1):
        if tuple(generated_ids[index : index + ngram_size - 1]) == prefix:
            banned_tokens.add(generated_ids[index + ngram_size - 1])

    if not banned_tokens:
        return logits

    filtered = logits.clone()
    for token_id in banned_tokens:
        if 0 <= token_id < filtered.shape[-1]:
            filtered[token_id] = float("-inf")

    return filtered


def apply_tempo_bias(
    logits: torch.Tensor,
    tempo: str,
) -> torch.Tensor:
    """Bias time-shift tokens based on desired tempo (slow/normal/fast)."""
    if tempo not in ("slow", "normal", "fast"):
        return logits

    adjusted = logits.clone()

    # TIME_SHIFT tokens range from token ID 12 (TIME_SHIFT_1) to 111 (TIME_SHIFT_100)
    # Larger shift_amount = longer gap = slower music
    # Smaller shift_amount = shorter gap = faster music
    for token_id in range(TIME_SHIFT_MIN, TIME_SHIFT_MAX + 1):
        shift_amount = (token_id - TIME_SHIFT_MIN) + 1

        if tempo == "slow":
            # Stronger "calm" shaping: clearly discourage tiny shifts and
            # strongly favor long gaps so the piece breathes more.
            if shift_amount <= 12:
                adjusted[token_id] -= 1.6
            elif shift_amount <= 20:
                adjusted[token_id] -= 0.9
            elif shift_amount >= 70:
                adjusted[token_id] += 1.2
            elif shift_amount >= 50:
                adjusted[token_id] += 0.9
        elif tempo == "fast":
            # Mildly penalize long shifts, mildly boost short shifts
            if shift_amount >= 50:
                adjusted[token_id] -= 0.5  # Gently discourage long gaps
            elif shift_amount <= 20:
                adjusted[token_id] += 0.8  # Gently encourage short gaps

    return adjusted


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
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
    if temperature <= 0:
        return torch.argmax(logits).item()

    if temperature > 0:
        logits = logits / temperature

    if top_k > 0:
        logits = apply_top_k(logits, top_k)

    if top_p < 1.0:
        logits = apply_top_p(logits, top_p)

    probs = F.softmax(logits, dim=-1)
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
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    repetition_window: int = 128,
    no_repeat_ngram_size: int = 0,
    tempo: str = "normal",
    seed: int = 42,
) -> List[int]:
    """
    Generate a sequence of token IDs using the trained GPT model.

    Uses key-value caching for efficient incremental generation.

    Args:
        model: Trained GPT model
        device: torch device
        prompt_ids: Initial token IDs (can be [BOS] or a custom prompt)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty factor for recently seen tokens
        repetition_window: Number of recent tokens used for repetition control
        no_repeat_ngram_size: Block tokens that would repeat an n-gram
        tempo: Tempo control ("slow", "normal", or "fast")
        seed: Random seed for reproducibility

    Returns:
        Generated token IDs (including prompt)
    """
    model.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    random.seed(seed)

    generated = prompt_ids.copy()
    active_pitches: set = set()  # For basic legality checks
    context_length = getattr(getattr(model, "cfg", None), "max_seq_len", len(generated))

    for _ in range(max_new_tokens):
        # Keep only the most recent context window that the model can represent.
        context_ids = generated[-context_length:] if context_length > 0 else generated
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)

        # Forward pass
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids)

        # Get logits for next token (last position)
        next_token_logits = logits[0, -1, :].clone()  # (vocab_size,)

        # Apply token legality constraints.
        for token_id in range(next_token_logits.shape[0]):
            if is_valid_note_off(token_id):
                pitch = get_pitch_from_note_token(token_id)
                if pitch is not None and pitch not in active_pitches:
                    next_token_logits[token_id] -= 2.0

        if repetition_penalty > 1.0:
            next_token_logits = apply_repetition_penalty(
                next_token_logits,
                generated,
                repetition_penalty,
                repetition_window,
            )

        if no_repeat_ngram_size > 1:
            next_token_logits = apply_no_repeat_ngram(
                next_token_logits,
                generated,
                no_repeat_ngram_size,
            )

        if tempo != "normal":
            next_token_logits = apply_tempo_bias(next_token_logits, tempo)

        # Sample next token
        next_token_id = sample_from_logits(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
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

    return generated


def tokens_to_notes(token_ids: List[int], id_to_token: dict) -> List[pretty_midi.Note]:
    """
    Convert token IDs to pretty_midi Note objects.

    Decodes the event-based representation back to notes.
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


def load_checkpoint(checkpoint_path: Path) -> tuple:
    """Load model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    config_dict = checkpoint.get("config", {})
    cfg = GPTConfig(**config_dict) if config_dict else GPTConfig()

    state_dict = checkpoint["model_state"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicGPT(cfg).to(device)

    # Keep checkpoints from nearby model revisions usable by restoring bias terms
    # when the saved state expects them.
    for block_idx, block in enumerate(model.blocks):
        qkv_bias_key = f"blocks.{block_idx}.attn.qkv.bias"
        proj_bias_key = f"blocks.{block_idx}.attn.proj.bias"

        if qkv_bias_key in state_dict and block.attn.qkv.bias is None:
            old_qkv = block.attn.qkv
            new_qkv = nn.Linear(old_qkv.in_features, old_qkv.out_features, bias=True).to(device)
            new_qkv.weight.data.copy_(old_qkv.weight.data)
            new_qkv.bias.data.zero_()
            block.attn.qkv = new_qkv

        if proj_bias_key in state_dict and block.attn.proj.bias is None:
            old_proj = block.attn.proj
            new_proj = nn.Linear(old_proj.in_features, old_proj.out_features, bias=True).to(device)
            new_proj.weight.data.copy_(old_proj.weight.data)
            new_proj.bias.data.zero_()
            block.attn.proj = new_proj

    load_result = model.load_state_dict(state_dict, strict=False)

    unexpected = list(load_result.unexpected_keys)
    missing = [key for key in load_result.missing_keys if not key.endswith("attn.causal_mask")]

    if missing or unexpected:
        print(f"[checkpoint] missing keys: {missing}", flush=True)
        print(f"[checkpoint] unexpected keys: {unexpected}", flush=True)

    return model, checkpoint


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MIDI sequences with trained GPT.")

    # Model checkpoint
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt file)")
    p.add_argument("--vocab-path", default="tokenized/id_to_token.json", help="Path to id_to_token.json")

    # Generation parameters
    p.add_argument("--prompt", type=str, default=None, help="Optional prompt text (token names separated by commas)")
    p.add_argument("--prompt-file", type=str, default=None, help="File containing prompt token IDs (one per line)")
    p.add_argument("--max-new-tokens", type=int, default=512, help="Maximum tokens to generate")
    p.add_argument("--temperature", type=float, default=0.82, help="Sampling temperature")
    p.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling (1.0 = disabled)")
    p.add_argument("--repetition-penalty", type=float, default=1.15, help="Penalty for recently seen tokens (1.0 = disabled)")
    p.add_argument("--repetition-window", type=int, default=128, help="Number of recent tokens used for repetition penalty")
    p.add_argument("--no-repeat-ngram-size", type=int, default=4, help="Block repeated n-grams (0 or 1 = disabled)")
    p.add_argument("--tempo", type=str, default="normal", choices=["slow", "normal", "fast"], help="Tempo control")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    p.add_argument("--output-dir", type=str, default="generated/gpt", help="Output directory for MIDI files")
    p.add_argument("--num-samples", type=int, default=5, help="Number of samples to generate")
    p.add_argument("--save-tokens", action="store_true", help="Also save generated token IDs as JSON")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    checkpoint_path = resolve_existing_path(args.checkpoint, [PROJECT_ROOT / "models", PROJECT_ROOT / "models" / "gpt"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    vocab_path = resolve_existing_path(args.vocab_path, [PROJECT_ROOT / "tokenized"])
    id_to_token_path, token_to_id_path = resolve_vocab_paths(vocab_path)
    id_to_token_path = resolve_existing_path(str(id_to_token_path), [PROJECT_ROOT / "tokenized"])
    token_to_id_path = resolve_existing_path(str(token_to_id_path), [PROJECT_ROOT / "tokenized"])
    if not id_to_token_path.exists():
        raise FileNotFoundError(f"id_to_token.json not found: {id_to_token_path}")
    if not token_to_id_path.exists():
        raise FileNotFoundError(f"vocab.json not found: {token_to_id_path}")

    # Load vocabulary mappings
    id_to_token = load_id_to_token(id_to_token_path)
    token_to_id = load_token_to_id(token_to_id_path)
    vocab_size = len(id_to_token)
    print(f"Vocabulary size: {vocab_size}", flush=True)

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    model, checkpoint = load_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    loaded_step = checkpoint.get("step", "unknown")
    print(f"Loaded checkpoint from step {loaded_step}", flush=True)

    # Prepare prompt
    if args.prompt_file:
        prompt_file_path = resolve_existing_path(args.prompt_file, [PROJECT_ROOT])
        if not prompt_file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")
        with prompt_file_path.open("r", encoding="utf-8") as f:
            prompt_ids = []
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    token_id = int(stripped)
                except ValueError as exc:
                    raise ValueError(f"Invalid token ID in prompt file at line {line_no}: {stripped}") from exc
                if token_id < 0 or token_id >= vocab_size:
                    raise ValueError(f"Prompt token ID out of range at line {line_no}: {token_id}")
                prompt_ids.append(token_id)
        if not prompt_ids:
            raise ValueError(f"Prompt file is empty: {prompt_file_path}")
        print(f"Loaded prompt from file: {len(prompt_ids)} tokens", flush=True)
    elif args.prompt:
        prompt_tokens = [t.strip() for t in args.prompt.split(",")]
        if not prompt_tokens or any(not token for token in prompt_tokens):
            raise ValueError("Prompt text must contain at least one token name.")

        unknown_tokens = [token for token in prompt_tokens if token not in token_to_id]
        if unknown_tokens:
            raise ValueError(f"Unknown prompt token(s): {', '.join(unknown_tokens)}")

        prompt_ids = [token_to_id[token] for token in prompt_tokens]
        print(f"Using text prompt: {prompt_tokens}", flush=True)
    else:
        prompt_ids = [BOS_TOKEN]
        print("Using default BOS prompt", flush=True)

    if hasattr(model, "cfg") and len(prompt_ids) > model.cfg.max_seq_len:
        print(
            f"Prompt length {len(prompt_ids)} exceeds model context window {model.cfg.max_seq_len}; "
            "the generator will use the most recent tokens.",
            flush=True,
        )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples
    print("=" * 60, flush=True)
    print(f"Generating {args.num_samples} samples...", flush=True)
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  temperature: {args.temperature}")
    print(f"  top_k: {args.top_k}")
    print(f"  top_p: {args.top_p}")
    print(f"  repetition_penalty: {args.repetition_penalty}")
    print(f"  repetition_window: {args.repetition_window}")
    print(f"  no_repeat_ngram_size: {args.no_repeat_ngram_size}")
    print(f"  tempo: {args.tempo}")
    print(f"  seed: {args.seed}")
    print("=" * 60, flush=True)

    for sample_idx in range(args.num_samples):
        sample_seed = args.seed + sample_idx

        generated_ids = generate(
            model=model,
            device=device,
            prompt_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            tempo=args.tempo,
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
