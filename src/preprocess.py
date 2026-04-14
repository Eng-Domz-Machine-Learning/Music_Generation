"""
Preprocess a directory of MIDI files for piano-only modeling.

Pipeline for each MIDI file:
1. Keep only piano instruments (program 0–7, non-drums).
2. Merge all piano tracks into a single list of notes.
3. Quantize note start/end times to a 16th-note grid.
4. Clip pitches to the piano range [21, 108].
5. Quantize velocity into 8 discrete bins.
6. Slice into fixed 30-second chunks.
7. Discard chunks with fewer than 10 notes.
8. Save each chunk as a new MIDI file in the output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, List

from tqdm.auto import tqdm
import pretty_midi

# ---------------------------------------------------------------------------
# Configuration defaults (can be overridden by CLI arguments)
# ---------------------------------------------------------------------------
INPUT_DIR = r"C:\path\to\your\raw_midi"
OUTPUT_DIR = r"C:\path\to\your\processed_midi"

PIANO_PROGRAM_MIN = 0
PIANO_PROGRAM_MAX = 7
PITCH_MIN = 21
PITCH_MAX = 108
NUM_VELOCITY_BINS = 8
CHUNK_SECONDS = 30.0
MIN_NOTES_PER_CHUNK = 10


def find_midi_files(root_dir: str | Path) -> List[Path]:
    """Recursively find all .mid and .midi files under root_dir."""
    root = Path(root_dir)
    if not root.is_dir():
        return []
    files = list(root.rglob("*.mid")) + list(root.rglob("*.midi"))
    return sorted(files)


def collect_piano_notes(pm: pretty_midi.PrettyMIDI) -> List[pretty_midi.Note]:
    """Return notes from all piano instruments (program 0–7, non-drums)."""
    notes: List[pretty_midi.Note] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        if not (PIANO_PROGRAM_MIN <= inst.program <= PIANO_PROGRAM_MAX):
            continue
        notes.extend(inst.notes)
    return notes


# Fixed 16th-note step at 120 BPM (seconds per 16th note)
_SIXTEENTH_STEP_120_BPM = 60.0 / 120.0 / 4.0


def make_quantizer(pm: pretty_midi.PrettyMIDI) -> Callable[[float], float]:
    """
    Return a quantizer function that snaps time to the nearest 16th-note grid.

    Uses pm.time_to_beat / pm.beat_to_time when available; otherwise falls back
    to a fixed 120 BPM grid. The returned function is intended to be reused for
    all notes in the same file.
    """
    try:
        _ = pm.time_to_beat(0.0)
        _ = pm.beat_to_time(0.0)
    except Exception:
        step = _SIXTEENTH_STEP_120_BPM
        return lambda t: round(t / step) * step

    def quantize(t: float) -> float:
        beat = pm.time_to_beat(t)
        quantized_beat = round(beat * 4.0) / 4.0
        return pm.beat_to_time(quantized_beat)

    return quantize


def clip_pitch(pitch: int) -> int:
    """Clip MIDI pitch into [PITCH_MIN, PITCH_MAX]."""
    if pitch < PITCH_MIN:
        return PITCH_MIN
    if pitch > PITCH_MAX:
        return PITCH_MAX
    return pitch


def quantize_velocity(velocity: int) -> int:
    """Map velocity (0–127) into NUM_VELOCITY_BINS discrete bins."""
    v = max(1, min(127, velocity))
    bin_index = int((v - 1) * NUM_VELOCITY_BINS / 127)
    bin_index = max(0, min(NUM_VELOCITY_BINS - 1, bin_index))
    # Map bin index back to a representative velocity value in 1–127
    bin_center = (bin_index + 0.5) * (127.0 / NUM_VELOCITY_BINS)
    return int(round(max(1.0, min(127.0, bin_center))))


def process_notes(
    quantize_fn: Callable[[float], float],
    notes: Iterable[pretty_midi.Note],
) -> List[pretty_midi.Note]:
    """Apply quantization, pitch clipping, and velocity binning to notes."""
    processed: List[pretty_midi.Note] = []
    for note in notes:
        start_q = quantize_fn(note.start)
        end_q = quantize_fn(note.end)
        if end_q <= start_q:
            continue

        pitch_q = clip_pitch(note.pitch)
        vel_q = quantize_velocity(note.velocity)

        processed.append(
            pretty_midi.Note(
                velocity=vel_q,
                pitch=pitch_q,
                start=start_q,
                end=end_q,
            )
        )
    processed.sort(key=lambda n: n.start)
    return processed


def split_into_chunks(
    notes: List[pretty_midi.Note],
    chunk_seconds: float = CHUNK_SECONDS,
    min_notes: int = MIN_NOTES_PER_CHUNK,
) -> List[List[pretty_midi.Note]]:
    """
    Split processed notes into fixed-length chunks in time.

    Notes that cross chunk boundaries are clipped to the boundary and
    shifted so each chunk starts at time 0.
    """
    if not notes:
        return []

    max_end = max(n.end for n in notes)
    chunks: List[List[pretty_midi.Note]] = []
    chunk_start = 0.0

    while chunk_start < max_end:
        chunk_end = chunk_start + chunk_seconds
        chunk_notes: List[pretty_midi.Note] = []

        for n in notes:
            if n.end <= chunk_start or n.start >= chunk_end:
                continue
            start = max(n.start, chunk_start) - chunk_start
            end = min(n.end, chunk_end) - chunk_start
            if end <= start:
                continue
            chunk_notes.append(
                pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=start,
                    end=end,
                )
            )

        if len(chunk_notes) >= min_notes:
            chunks.append(chunk_notes)

        chunk_start += chunk_seconds

    return chunks


def save_chunk(
    notes: List[pretty_midi.Note],
    output_path: Path,
) -> None:
    """Create a PrettyMIDI object with a single piano instrument and save it."""
    pm_out = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, is_drum=False)
    instrument.notes = notes
    pm_out.instruments.append(instrument)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm_out.write(str(output_path))


def process_midi_file(
    input_path: Path,
    output_dir: Path,
    input_root: Path,
) -> int:
    """
    Process a single MIDI file and write its chunks.

    Returns the number of chunks written for this file.
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(input_path))
    except Exception:
        # Skip corrupted / unreadable files silently.
        return 0

    piano_notes = collect_piano_notes(pm)
    if not piano_notes:
        return 0

    quantize_fn = make_quantizer(pm)
    processed_notes = process_notes(quantize_fn, piano_notes)
    if not processed_notes:
        return 0

    chunks = split_into_chunks(processed_notes)
    if not chunks:
        return 0

    try:
        parent_rel = input_path.parent.relative_to(input_root)
        prefix = "_".join(parent_rel.parts) if parent_rel.parts else "root"
    except ValueError:
        prefix = input_path.parent.name or "root"
    stem = input_path.stem
    file_prefix = f"{prefix}_{stem}" if prefix != "root" else stem
    chunks_written = 0

    for idx, chunk_notes in enumerate(chunks):
        out_name = f"{file_prefix}_chunk{idx:03d}.mid"
        out_path = output_dir / out_name
        save_chunk(chunk_notes, out_path)
        chunks_written += 1

    return chunks_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a directory of MIDI files (piano-only)."
    )
    parser.add_argument(
        "--input-dir",
        default=INPUT_DIR,
        help="Root directory of raw MIDI files (default: INPUT_DIR constant).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory to write processed MIDI chunks (default: OUTPUT_DIR constant).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    midi_files = find_midi_files(input_dir)
    if not midi_files:
        print(f"No .mid or .midi files found under: {input_dir}")
        return

    total_chunks = 0

    for midi_path in tqdm(midi_files, desc="Processing MIDI files"):
        total_chunks += process_midi_file(midi_path, output_dir, input_dir)

    print(f"Done. Wrote {total_chunks} chunks to {output_dir}")


if __name__ == "__main__":
    main()

