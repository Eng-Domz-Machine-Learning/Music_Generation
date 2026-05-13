"""Configuration settings for the Gradio music generation app."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "gpt"
TOKENIZED_DIR = PROJECT_ROOT / "tokenized"
GENERATED_DIR = PROJECT_ROOT / "generated" / "gradio_output"

# Model settings
CHECKPOINT_PATH = MODELS_DIR / "best.pt"
ID_TO_TOKEN_PATH = TOKENIZED_DIR / "id_to_token.json"
VOCAB_PATH = TOKENIZED_DIR / "vocab.json"

# Generation defaults
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.82
DEFAULT_TOP_K = 0
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_REPETITION_WINDOW = 128
DEFAULT_NO_REPEAT_NGRAM_SIZE = 4
DEFAULT_TEMPO = "normal"
DEFAULT_NUM_SAMPLES = 1
DEFAULT_MOOD_PRESET = "Balanced / Storytelling"

# UI settings
APP_TITLE = "🎵 Piano Music Generation with GPT"
APP_DESCRIPTION = """
Generate symbolic MIDI piano music using a GPT-based deep learning model trained on classical piano compositions.
Adjust generation parameters to control the output's style, tempo, and musical characteristics.
"""

# Constraints
MAX_SAMPLES = 5
MIN_TEMPERATURE = 0.1
MAX_TEMPERATURE = 2.0
MIN_REPETITION_PENALTY = 1.0
MAX_REPETITION_PENALTY = 2.0
TEMPO_CHOICES = ["slow", "normal", "fast"]
TEMPO_UI_CHOICES = [
    ("Calm / Reflective (~70 BPM feel)", "slow"),
    ("Balanced / Storytelling (~95 BPM feel)", "normal"),
    ("Bright / Energetic (~120 BPM feel)", "fast"),
]

# Deterministic post-generation timing scale. Values > 1.0 slow down playback,
# values < 1.0 speed it up. This guarantees audible tempo separation.
TEMPO_TIME_SCALE = {
    "slow": 1.35,
    "normal": 1.00,
    "fast": 0.78,
}

MOOD_PRESETS = {
    "Calm / Reflective": {
        "tempo": "slow",
        "temperature": 0.65,
        "top_p": 0.80,
        "max_new_tokens": 760,
    },
    "Balanced / Storytelling": {
        "tempo": "normal",
        "temperature": 0.82,
        "top_p": 0.90,
        "max_new_tokens": 512,
    },
    "Bright / Energetic": {
        "tempo": "fast",
        "temperature": 0.90,
        "top_p": 0.94,
        "max_new_tokens": 420,
    },
}

# Validation
REQUIRED_FILES = [CHECKPOINT_PATH, ID_TO_TOKEN_PATH, VOCAB_PATH]
