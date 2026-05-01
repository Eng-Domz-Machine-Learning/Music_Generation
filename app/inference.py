"""Model inference wrapper for the Gradio app."""

import sys
from pathlib import Path
import json
import random
from typing import List, Optional, Tuple
import importlib.util

import torch
import torch.nn as nn
import pretty_midi

from app.config import (
    PROJECT_ROOT,
    CHECKPOINT_PATH,
    ID_TO_TOKEN_PATH,
    VOCAB_PATH,
    GENERATED_DIR,
)

# Add src to path so we can import training modules
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_gpt_improved import MusicGPT, GPTConfig

# Import generate_gpt using importlib for robustness
_generate_gpt_path = SRC_DIR / "generate_gpt.py"
_spec = importlib.util.spec_from_file_location("generate_gpt", str(_generate_gpt_path))
_generate_gpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_generate_gpt)
generate = _generate_gpt.generate
tokens_to_notes = _generate_gpt.tokens_to_notes


class MusicGenerator:
    """Wrapper for GPT music generation with caching and error handling."""

    def __init__(self):
        """Initialize the model and tokenizers."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[MusicGPT] = None
        self.id_to_token: Optional[dict] = None
        self.token_to_id: Optional[dict] = None
        self.vocab_size: int = 0

        # Ensure output directory exists
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)

        # Load on init
        self._load_model()
        self._load_tokenizers()

    def _load_model(self) -> None:
        """Load the GPT model checkpoint."""
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

        checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=self.device, weights_only=False)
        config_dict = checkpoint.get("config", {})
        cfg = GPTConfig(**config_dict) if config_dict else GPTConfig()

        self.model = MusicGPT(cfg).to(self.device)
        self.model.eval()

        # Load state dict with compatibility for nearby model revisions
        state_dict = checkpoint["model_state"]
        for block_idx, block in enumerate(self.model.blocks):
            qkv_bias_key = f"blocks.{block_idx}.attn.qkv.bias"
            proj_bias_key = f"blocks.{block_idx}.attn.proj.bias"

            if qkv_bias_key in state_dict and block.attn.qkv.bias is None:
                old_qkv = block.attn.qkv
                new_qkv = nn.Linear(old_qkv.in_features, old_qkv.out_features, bias=True).to(self.device)
                new_qkv.weight.data.copy_(old_qkv.weight.data)
                new_qkv.bias.data.zero_()
                block.attn.qkv = new_qkv

            if proj_bias_key in state_dict and block.attn.proj.bias is None:
                old_proj = block.attn.proj
                new_proj = nn.Linear(old_proj.in_features, old_proj.out_features, bias=True).to(self.device)
                new_proj.weight.data.copy_(old_proj.weight.data)
                new_proj.bias.data.zero_()
                block.attn.proj = new_proj

        load_result = self.model.load_state_dict(state_dict, strict=False)

        unexpected = [k for k in load_result.unexpected_keys if not k.endswith("bias")]
        missing = [k for k in load_result.missing_keys if not k.endswith("attn.causal_mask")]

        if missing or unexpected:
            print(f"[checkpoint] missing keys: {missing}")
            print(f"[checkpoint] unexpected keys: {unexpected}")

    def _load_tokenizers(self) -> None:
        """Load token ID mappings."""
        if not ID_TO_TOKEN_PATH.exists():
            raise FileNotFoundError(f"id_to_token.json not found: {ID_TO_TOKEN_PATH}")
        if not VOCAB_PATH.exists():
            raise FileNotFoundError(f"vocab.json not found: {VOCAB_PATH}")

        with ID_TO_TOKEN_PATH.open("r", encoding="utf-8") as f:
            self.id_to_token = {int(k): v for k, v in json.load(f).items()}

        with VOCAB_PATH.open("r", encoding="utf-8") as f:
            self.token_to_id = {k: int(v) for k, v in json.load(f).items()}

        self.vocab_size = len(self.id_to_token)

    def generate(
        self,
        num_samples: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.82,
        top_k: int = 0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        repetition_window: int = 128,
        no_repeat_ngram_size: int = 4,
        tempo: str = "normal",
        seed: int = 42,
    ) -> List[str]:
        """
        Generate MIDI samples and return file paths.

        Args:
            num_samples: Number of samples to generate
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            repetition_penalty: Penalty for repeated tokens
            repetition_window: Window for repetition tracking
            no_repeat_ngram_size: Block repeated n-grams of this size
            tempo: Tempo control ("slow", "normal", or "fast")
            seed: Random seed

        Returns:
            List of file paths to generated MIDI files
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        output_paths = []
        PAD_TOKEN = 0
        BOS_TOKEN = 1
        EOS_TOKEN = 2

        for sample_idx in range(num_samples):
            sample_seed = seed + sample_idx

            # Generate token sequence
            generated_ids = generate(
                model=self.model,
                device=self.device,
                prompt_ids=[BOS_TOKEN],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                no_repeat_ngram_size=no_repeat_ngram_size,
                tempo=tempo,
                seed=sample_seed,
            )

            # Decode to notes
            try:
                notes = tokens_to_notes(generated_ids, self.id_to_token)
                if not notes:
                    print(f"Sample {sample_idx}: No notes decoded")
                    continue

                # Save MIDI
                output_path = GENERATED_DIR / f"sample_{sample_idx:03d}_{sample_seed}.mid"
                pm = pretty_midi.PrettyMIDI()
                instrument = pretty_midi.Instrument(program=0, is_drum=False)
                instrument.notes = notes
                pm.instruments.append(instrument)
                pm.write(str(output_path))

                output_paths.append(str(output_path))
            except Exception as e:
                print(f"Error decoding sample {sample_idx}: {e}")
                continue

        return output_paths


# Global instance (lazy loaded)
_generator: Optional[MusicGenerator] = None


def get_generator() -> MusicGenerator:
    """Get or create the global generator instance."""
    global _generator
    if _generator is None:
        _generator = MusicGenerator()
    return _generator
