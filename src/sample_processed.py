"""
Load a few random MIDIs from the processed dataset and print duration and note count.
"""

import random
from pathlib import Path

import pretty_midi

# Directory containing processed MIDI chunks (relative to project root or absolute)
PROCESSED_DIR = Path("data_processed")


def find_midi_files(root_dir: Path) -> list[Path]:
    """Collect all .mid and .midi files under root_dir."""
    if not root_dir.is_dir():
        return []
    files = list(root_dir.rglob("*.mid")) + list(root_dir.rglob("*.midi"))
    return files


def main() -> None:
    files = find_midi_files(PROCESSED_DIR)
    if not files:
        print(f"No MIDI files found in {PROCESSED_DIR}")
        return

    sample = random.sample(files, min(5, len(files)))

    print(f"Sampling {len(sample)} file(s) from {PROCESSED_DIR}\n")
    for path in sample:
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
        except Exception as e:
            print(f"  {path.name}: failed to load ({e})")
            continue
        duration = pm.get_end_time()
        note_count = sum(len(inst.notes) for inst in pm.instruments)
        print(f"  {path.name}")
        print(f"    duration: {duration:.2f} s  |  notes: {note_count}")


if __name__ == "__main__":
    main()
