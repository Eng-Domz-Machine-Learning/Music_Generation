"""
Tokenize processed piano MIDI chunks into event-based sequences.

Outputs are written directly into the tokenized/ directory:
- dataset.jsonl
- vocab.json
- id_to_token.json
- stats.json
- skipped.jsonl

The tokenizer uses an event representation with:
- NOTE_ON_<pitch>
- NOTE_OFF_<pitch>
- TIME_SHIFT_<n>
- VELOCITY_<bin>
- PAD / BOS / EOS / UNK
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Sequence

# Prevent this script from shadowing the stdlib `tokenize` module when run directly.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

import pretty_midi
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
INPUT_DIR = Path("data_processed")
OUTPUT_DIR = Path("tokenized")

PITCH_MIN = 21
PITCH_MAX = 108
NUM_VELOCITY_BINS = 8
TIME_STEP_SECONDS = 0.125
MAX_TIME_SHIFT = 100
MIN_TOKENS = 1

SPECIAL_TOKENS = ["PAD", "BOS", "EOS", "UNK"]


class NoteEvent(NamedTuple):
    step: int
    event_type: str
    pitch: int
    velocity_bin: int | None = None


class QuantizedNote(NamedTuple):
    pitch: int
    velocity_bin: int
    start_step: int
    end_step: int


def find_midi_files(root_dir: Path) -> List[Path]:
    """Recursively collect MIDI files under root_dir."""
    if not root_dir.is_dir():
        return []
    files = list(root_dir.rglob("*.mid")) + list(root_dir.rglob("*.midi"))
    return sorted(files)


def velocity_to_bin(velocity: int) -> int:
    """Map a MIDI velocity in 1..127 to a bin index in 0..NUM_VELOCITY_BINS-1."""
    v = max(1, min(127, velocity))
    bin_index = int((v - 1) * NUM_VELOCITY_BINS / 127)
    return max(0, min(NUM_VELOCITY_BINS - 1, bin_index))


def note_to_quantized(
    note: pretty_midi.Note,
    time_step_seconds: float,
) -> QuantizedNote | None:
    """Convert a PrettyMIDI note into a quantized note representation."""
    pitch = max(PITCH_MIN, min(PITCH_MAX, note.pitch))
    start_step = int(round(note.start / time_step_seconds))
    end_step = int(round(note.end / time_step_seconds))

    if end_step <= start_step:
        return None

    return QuantizedNote(
        pitch=pitch,
        velocity_bin=velocity_to_bin(note.velocity),
        start_step=start_step,
        end_step=end_step,
    )


def load_quantized_notes(
    midi_path: Path,
    time_step_seconds: float,
) -> List[QuantizedNote]:
    """Load and quantize notes from a processed MIDI chunk."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes: List[QuantizedNote] = []

    for instrument in pm.instruments:
        for note in instrument.notes:
            quantized = note_to_quantized(note, time_step_seconds)
            if quantized is not None:
                notes.append(quantized)

    notes.sort(key=lambda n: (n.start_step, n.end_step, n.pitch, n.velocity_bin))
    return notes


def build_events(notes: Sequence[QuantizedNote]) -> List[NoteEvent]:
    """Create raw note_on/note_off events from quantized notes."""
    events: List[NoteEvent] = []
    for note in notes:
        events.append(
            NoteEvent(
                step=note.start_step,
                event_type="note_on",
                pitch=note.pitch,
                velocity_bin=note.velocity_bin,
            )
        )
        events.append(
            NoteEvent(
                step=note.end_step,
                event_type="note_off",
                pitch=note.pitch,
            )
        )

    event_order = {"note_off": 0, "note_on": 1}
    events.sort(key=lambda e: (e.step, event_order[e.event_type], e.pitch))
    return events


def emit_time_shifts(delta_steps: int, max_time_shift: int) -> List[str]:
    """Encode a gap in steps as one or more TIME_SHIFT tokens."""
    tokens: List[str] = []
    remaining = delta_steps

    while remaining > 0:
        shift = min(remaining, max_time_shift)
        tokens.append(f"TIME_SHIFT_{shift}")
        remaining -= shift

    return tokens


def tokenize_events(
    events: Sequence[NoteEvent],
    max_time_shift: int,
) -> List[str]:
    """Convert sorted events into a deterministic token sequence."""
    tokens = ["BOS"]
    current_step = 0
    current_velocity_bin: int | None = None

    for event in events:
        if event.step < current_step:
            raise ValueError("Events are not sorted by non-decreasing step.")

        delta_steps = event.step - current_step
        if delta_steps > 0:
            tokens.extend(emit_time_shifts(delta_steps, max_time_shift))
            current_step = event.step

        if event.event_type == "note_on":
            if event.velocity_bin is None:
                raise ValueError("note_on event is missing velocity_bin.")
            if event.velocity_bin != current_velocity_bin:
                tokens.append(f"VELOCITY_{event.velocity_bin}")
                current_velocity_bin = event.velocity_bin
            tokens.append(f"NOTE_ON_{event.pitch}")
        elif event.event_type == "note_off":
            tokens.append(f"NOTE_OFF_{event.pitch}")
        else:
            raise ValueError(f"Unsupported event type: {event.event_type}")

    tokens.append("EOS")
    return tokens


def decode_tokens_to_notes(tokens: Sequence[str]) -> List[QuantizedNote]:
    """Reconstruct quantized notes from a token sequence."""
    current_step = 0
    current_velocity_bin = 0
    active_starts: Dict[int, List[tuple[int, int]]] = defaultdict(list)
    notes: List[QuantizedNote] = []

    for token in tokens:
        if token in {"PAD", "BOS", "EOS"}:
            continue

        if token.startswith("TIME_SHIFT_"):
            shift = int(token.split("_")[-1])
            current_step += shift
            continue

        if token.startswith("VELOCITY_"):
            current_velocity_bin = int(token.split("_")[-1])
            continue

        if token.startswith("NOTE_ON_"):
            pitch = int(token.split("_")[-1])
            active_starts[pitch].append((current_step, current_velocity_bin))
            continue

        if token.startswith("NOTE_OFF_"):
            pitch = int(token.split("_")[-1])
            starts = active_starts[pitch]
            if not starts:
                raise ValueError(f"Unmatched NOTE_OFF for pitch {pitch}.")

            start_step, velocity_bin = starts.pop(0)
            if current_step <= start_step:
                raise ValueError(f"Non-positive note length for pitch {pitch}.")

            notes.append(
                QuantizedNote(
                    pitch=pitch,
                    velocity_bin=velocity_bin,
                    start_step=start_step,
                    end_step=current_step,
                )
            )
            continue

        raise ValueError(f"Unknown token: {token}")

    dangling = {pitch: starts for pitch, starts in active_starts.items() if starts}
    if dangling:
        raise ValueError("Unmatched NOTE_ON events remain after decoding.")

    notes.sort(key=lambda n: (n.start_step, n.end_step, n.pitch, n.velocity_bin))
    return notes


def compare_note_lists(
    expected: Sequence[QuantizedNote],
    actual: Sequence[QuantizedNote],
) -> None:
    """Raise ValueError if two quantized note lists do not match exactly."""
    if len(expected) != len(actual):
        raise ValueError(
            f"Roundtrip note count mismatch: expected {len(expected)}, got {len(actual)}."
        )

    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            raise ValueError(
                "Roundtrip note mismatch at index "
                f"{index}: expected {left}, got {right}."
            )


def validate_token_sequence(tokens: Sequence[str], max_time_shift: int) -> None:
    """Check token sequence shape and token-family constraints."""
    if not tokens:
        raise ValueError("Token sequence is empty.")
    if tokens[0] != "BOS" or tokens[-1] != "EOS":
        raise ValueError("Token sequence must start with BOS and end with EOS.")

    for token in tokens:
        if token in SPECIAL_TOKENS:
            continue

        if token.startswith("NOTE_ON_") or token.startswith("NOTE_OFF_"):
            pitch = int(token.split("_")[-1])
            if not (PITCH_MIN <= pitch <= PITCH_MAX):
                raise ValueError(f"Illegal pitch token: {token}")
            continue

        if token.startswith("VELOCITY_"):
            velocity_bin = int(token.split("_")[-1])
            if not (0 <= velocity_bin < NUM_VELOCITY_BINS):
                raise ValueError(f"Illegal velocity token: {token}")
            continue

        if token.startswith("TIME_SHIFT_"):
            shift = int(token.split("_")[-1])
            if not (1 <= shift <= max_time_shift):
                raise ValueError(f"Illegal time shift token: {token}")
            continue

        raise ValueError(f"Unsupported token encountered: {token}")


def make_base_vocabulary(max_time_shift: int) -> List[str]:
    """Create a deterministic vocabulary order."""
    tokens = list(SPECIAL_TOKENS)
    tokens.extend(f"VELOCITY_{velocity_bin}" for velocity_bin in range(NUM_VELOCITY_BINS))
    tokens.extend(f"TIME_SHIFT_{shift}" for shift in range(1, max_time_shift + 1))
    tokens.extend(f"NOTE_ON_{pitch}" for pitch in range(PITCH_MIN, PITCH_MAX + 1))
    tokens.extend(f"NOTE_OFF_{pitch}" for pitch in range(PITCH_MIN, PITCH_MAX + 1))
    return tokens


def build_vocab(max_time_shift: int) -> tuple[Dict[str, int], Dict[int, str]]:
    """Return token-to-id and id-to-token mappings."""
    vocab_tokens = make_base_vocabulary(max_time_shift)
    token_to_id = {token: idx for idx, token in enumerate(vocab_tokens)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return token_to_id, id_to_token


def encode_tokens(tokens: Sequence[str], token_to_id: Dict[str, int]) -> List[int]:
    """Map token strings to integer IDs using UNK as a fallback."""
    unk_id = token_to_id["UNK"]
    return [token_to_id.get(token, unk_id) for token in tokens]


def verify_encoding_roundtrip(
    tokens: Sequence[str],
    token_ids: Sequence[int],
    id_to_token: Dict[int, str],
) -> None:
    """Ensure token IDs decode back to the same string sequence."""
    decoded = [id_to_token[token_id] for token_id in token_ids]
    if list(tokens) != decoded:
        raise ValueError("Token ID roundtrip mismatch.")


def tokenization_record(
    source_file: Path,
    relative_to: Path,
    tokens: Sequence[str],
    token_ids: Sequence[int],
    num_notes: int,
) -> dict:
    """Build the JSON-serializable record for dataset.jsonl."""
    return {
        "source_file": str(source_file.relative_to(relative_to)).replace("\\", "/"),
        "tokens": list(tokens),
        "token_ids": list(token_ids),
        "num_tokens": len(tokens),
        "num_notes": num_notes,
    }


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def verify_sample_records(
    records: Sequence[dict],
    input_dir: Path,
    time_step_seconds: float,
) -> List[dict]:
    """Roundtrip-check a sample of tokenized records against source MIDIs."""
    failures: List[dict] = []

    for record in records:
        source_path = input_dir / Path(record["source_file"])
        try:
            original_notes = load_quantized_notes(source_path, time_step_seconds)
            decoded_notes = decode_tokens_to_notes(record["tokens"])
            compare_note_lists(original_notes, decoded_notes)
        except Exception as exc:
            failures.append(
                {
                    "source_file": record["source_file"],
                    "reason": str(exc),
                }
            )

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize processed piano-only MIDI chunks into event sequences."
    )
    parser.add_argument(
        "--input-dir",
        default=str(INPUT_DIR),
        help="Directory containing processed MIDI chunks.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory where tokenized outputs will be written.",
    )
    parser.add_argument(
        "--max-time-shift",
        type=int,
        default=MAX_TIME_SHIFT,
        help="Maximum TIME_SHIFT token value before splitting long gaps.",
    )
    parser.add_argument(
        "--time-step-seconds",
        type=float,
        default=TIME_STEP_SECONDS,
        help="Time quantization step in seconds.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=MIN_TOKENS,
        help="Minimum token count required to keep a tokenized file.",
    )
    parser.add_argument(
        "--verify-sample",
        type=int,
        default=0,
        help="Number of tokenized files to roundtrip-check after tokenization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.max_time_shift < 1:
        raise ValueError("--max-time-shift must be at least 1.")
    if args.time_step_seconds <= 0:
        raise ValueError("--time-step-seconds must be positive.")
    if args.min_tokens < 1:
        raise ValueError("--min-tokens must be at least 1.")
    if args.verify_sample < 0:
        raise ValueError("--verify-sample cannot be negative.")

    midi_files = find_midi_files(input_dir)
    if not midi_files:
        print(f"No .mid or .midi files found under: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    token_to_id, id_to_token = build_vocab(args.max_time_shift)

    records: List[dict] = []
    skipped: List[dict] = []
    lengths: List[int] = []

    for midi_path in tqdm(midi_files, desc="Tokenizing MIDI chunks"):
        try:
            notes = load_quantized_notes(midi_path, args.time_step_seconds)
            if not notes:
                raise ValueError("No valid notes found after quantization.")

            events = build_events(notes)
            if not events:
                raise ValueError("No events generated from notes.")

            tokens = tokenize_events(events, args.max_time_shift)
            validate_token_sequence(tokens, args.max_time_shift)

            if len(tokens) < args.min_tokens:
                raise ValueError(
                    f"Token sequence shorter than min_tokens ({len(tokens)} < {args.min_tokens})."
                )

            token_ids = encode_tokens(tokens, token_to_id)
            verify_encoding_roundtrip(tokens, token_ids, id_to_token)

            record = tokenization_record(
                source_file=midi_path,
                relative_to=input_dir,
                tokens=tokens,
                token_ids=token_ids,
                num_notes=len(notes),
            )
            records.append(record)
            lengths.append(len(tokens))
        except Exception as exc:
            skipped.append(
                {
                    "source_file": str(midi_path.relative_to(input_dir)).replace("\\", "/"),
                    "reason": str(exc),
                }
            )

    verification_failures: List[dict] = []
    if args.verify_sample > 0 and records:
        sample_size = min(args.verify_sample, len(records))
        sample_records = random.Random(0).sample(records, sample_size)
        verification_failures = verify_sample_records(
            sample_records,
            input_dir=input_dir,
            time_step_seconds=args.time_step_seconds,
        )

    avg_length = (sum(lengths) / len(lengths)) if lengths else 0.0
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0
    stats = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "time_step_seconds": args.time_step_seconds,
        "max_time_shift": args.max_time_shift,
        "min_tokens": args.min_tokens,
        "total_files_seen": len(midi_files),
        "tokenized_files": len(records),
        "skipped_files": len(skipped),
        "verification_sample_requested": args.verify_sample,
        "verification_sample_checked": min(args.verify_sample, len(records)),
        "verification_failures": len(verification_failures),
        "vocabulary_size": len(token_to_id),
        "average_sequence_length": avg_length,
        "min_sequence_length": min_length,
        "max_sequence_length": max_length,
    }

    write_jsonl(output_dir / "dataset.jsonl", records)
    write_json(output_dir / "vocab.json", token_to_id)
    write_json(output_dir / "id_to_token.json", id_to_token)
    write_json(output_dir / "stats.json", stats)
    write_jsonl(output_dir / "skipped.jsonl", skipped + verification_failures)

    print("Tokenization complete")
    print("-" * 40)
    print(f"Input dir:          {input_dir}")
    print(f"Output dir:         {output_dir}")
    print(f"Total files seen:   {len(midi_files)}")
    print(f"Tokenized files:    {len(records)}")
    print(f"Skipped files:      {len(skipped)}")
    print(f"Verification fails: {len(verification_failures)}")
    print(f"Vocabulary size:    {len(token_to_id)}")
    print(f"Avg seq length:     {avg_length:.2f}")
    print(f"Min seq length:     {min_length}")
    print(f"Max seq length:     {max_length}")


if __name__ == "__main__":
    main()
