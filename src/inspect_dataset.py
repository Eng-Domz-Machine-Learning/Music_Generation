"""
Inspect a directory of MIDI files: count files, instruments, piano tracks, and duration.
Uses pretty_midi; skips corrupted files gracefully.
"""

import argparse
import statistics
from pathlib import Path

import pretty_midi

# -----------------------------------------------------------------------------
# Configuration: set the directory to scan here
# -----------------------------------------------------------------------------
MIDI_DIR = r"C:\path\to\your\midi\folder"

# Piano programs in General MIDI (0 = Acoustic Grand Piano, 1–7 = other pianos)
PIANO_PROGRAM_MIN = 0
PIANO_PROGRAM_MAX = 7


def find_midi_files(root_dir: str) -> list[Path]:
    """Recursively find all .mid and .midi files under root_dir."""
    root = Path(root_dir)
    if not root.is_dir():
        return []
    files = list(root.rglob("*.mid")) + list(root.rglob("*.midi"))
    return sorted(files)


def load_and_analyze(path: Path) -> dict | None:
    """
    Load a MIDI file with pretty_midi and return stats, or None if corrupted.
    Returns dict with: num_instruments, num_piano, duration_seconds.
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return None

    num_instruments = len(pm.instruments)
    num_piano = sum(
        1
        for inst in pm.instruments
        if not inst.is_drum
        and PIANO_PROGRAM_MIN <= inst.program <= PIANO_PROGRAM_MAX
    )
    duration_seconds = pm.get_end_time()

    return {
        "num_instruments": num_instruments,
        "num_piano": num_piano,
        "duration_seconds": duration_seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a directory of MIDI files.")
    parser.add_argument(
        "--dir",
        default=MIDI_DIR,
        help="Directory to scan for .mid/.midi files (default: MIDI_DIR from script)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = args.dir
    midi_files = find_midi_files(root_dir)
    total_files = len(midi_files)

    if total_files == 0:
        print(f"No .mid or .midi files found under: {root_dir}")
        return

    loaded = []
    corrupted = 0
    total_duration = 0.0
    total_instruments = 0
    files_with_piano = 0
    files_multi_instrument = 0

    for path in midi_files:
        result = load_and_analyze(path)
        if result is None:
            corrupted += 1
            continue
        loaded.append(result)
        total_duration += result["duration_seconds"]
        total_instruments += result["num_instruments"]
        if result["num_piano"] > 0:
            files_with_piano += 1
        if result["num_instruments"] > 1:
            files_multi_instrument += 1

    num_loaded = len(loaded)
    durations = [r["duration_seconds"] for r in loaded]
    avg_duration = total_duration / num_loaded if num_loaded else 0.0
    min_duration = min(durations) if durations else 0.0
    max_duration = max(durations) if durations else 0.0
    median_duration = statistics.median(durations) if durations else 0.0
    avg_instruments = total_instruments / num_loaded if num_loaded else 0.0
    pct_piano = (100.0 * files_with_piano / num_loaded) if num_loaded else 0.0
    pct_multi = (100.0 * files_multi_instrument / num_loaded) if num_loaded else 0.0

    print("MIDI dataset inspection")
    print("-" * 40)
    print(f"Directory:        {root_dir}")
    print(f"Total files:      {total_files}")
    print(f"Loaded:           {num_loaded}")
    print(f"Corrupted:        {corrupted}")
    print(f"Duration (s)      min: {min_duration:.2f}  max: {max_duration:.2f}  median: {median_duration:.2f}  avg: {avg_duration:.2f}")
    print(f"Avg instruments: {avg_instruments:.1f}")
    print(f"Files w/ piano:   {pct_piano:.1f}%")
    print(f"Files w/ >1 inst: {pct_multi:.1f}%")


if __name__ == "__main__":
    main()
