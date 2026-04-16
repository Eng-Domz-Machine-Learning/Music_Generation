# Installation and Execution Instructions

## Quick Start

This document provides step-by-step instructions to install dependencies and run the music generation model comparison.

---

## Prerequisites

- **Python**: 3.9 or higher
- **Conda**: Anaconda or Miniconda installed
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)

---

## Step 1: Activate Conda Environment

```bash
conda activate music_generation_project
```

If the environment doesn't exist yet, create it:

```bash
conda create -n music_generation_project python=3.10
conda activate music_generation_project
```

---

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt contains**:
- `pretty_midi>=0.2.9` - MIDI processing
- `tqdm>=4.66.0` - Progress bars
- `torch>=2.2.0` - PyTorch for deep learning

**Optional**: For faster training with GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 3: Verify Data Exists

Ensure the tokenized dataset exists:

```
tokenized/
  dataset.jsonl      # Main dataset
  vocab.json         # Token to ID mapping
  id_to_token.json   # ID to token mapping
  stats.json         # Dataset statistics
```

If not, run the preprocessing pipeline first:

```bash
# Step 3a: Preprocess raw MIDI files
python src/preprocess.py \
  --input-dir data_raw/maestro \
  --output-dir data_processed

# Step 3b: Tokenize processed files
python src/midi_tokenize.py \
  --input-dir data_processed \
  --output-dir tokenized \
  --verify-sample 10
```

---

## Step 4: Train LSTM Baseline

```bash
python src/train_lstm.py \
  --data-path tokenized/dataset.jsonl \
  --out-dir models/lstm_baseline \
  --batch-size 32 \
  --hidden-size 512 \
  --num-layers 2 \
  --context-len 256 \
  --max-steps 15000 \
  --seed 42
```

**Expected output**:
- Checkpoints saved to `models/lstm_baseline/`
- `best.pt` - Best validation checkpoint
- `final.pt` - Final checkpoint
- `metrics.json` - Training metrics
- `config.json` - Model configuration

**Training time**: ~2-4 hours on GPU, ~8-12 hours on CPU

---

## Step 5: Train GPT (Improved)

```bash
python src/train_gpt_improved.py \
  --data-path tokenized/dataset.jsonl \
  --out-dir models/gpt_improved \
  --batch-size 16 \
  --d-model 256 \
  --n-layers 4 \
  --n-heads 4 \
  --max-seq-len 256 \
  --max-steps 12000 \
  --seed 42
```

**Expected output**:
- Checkpoints saved to `models/gpt_improved/`
- `best.pt` - Best validation checkpoint
- `final.pt` - Final checkpoint
- `metrics.json` - Training metrics
- `config.json` - Model configuration

**Training time**: ~3-5 hours on GPU, ~10-15 hours on CPU

---

## Step 6: Generate Samples from LSTM

```bash
python src/generate_lstm.py \
  --checkpoint models/lstm_baseline/best.pt \
  --output-dir generated/lstm \
  --num-samples 10 \
  --max-new-tokens 512 \
  --temperature 1.0 \
  --top-k 0 \
  --seed 42 \
  --save-tokens
```

**Output**:
- `generated/lstm/sample_000.mid` through `sample_009.mid`
- `generated/lstm/sample_000_tokens.json` through `sample_009_tokens.json`

---

## Step 7: Generate Samples from GPT

```bash
python src/generate_gpt.py \
  --checkpoint models/gpt_improved/best.pt \
  --output-dir generated/gpt \
  --num-samples 10 \
  --max-new-tokens 512 \
  --temperature 1.0 \
  --top-k 0 \
  --seed 42 \
  --save-tokens
```

**Output**:
- `generated/gpt/sample_000.mid` through `sample_009.mid`
- `generated/gpt/sample_000_tokens.json` through `sample_009_tokens.json`

---

## Step 8: Compare Results

### View Metrics

```bash
# LSTM metrics
cat models/lstm_baseline/metrics.json

# GPT metrics
cat models/gpt_improved/metrics.json
```

### Key Metrics to Compare

| Metric | Description | Better |
|--------|-------------|--------|
| `final_test_loss` | Cross-entropy loss on test set | Lower |
| `final_test_perplexity` | exp(loss), interpretable as branching factor | Lower |
| `best_val_loss` | Best validation loss during training | Lower |

### Listen to Samples

Play the generated MIDI files with any MIDI player:

**Windows**:
```bash
# Use Windows Media Player or any MIDI-compatible player
start generated/lstm/sample_000.mid
start generated/gpt/sample_000.mid
```

**With Python** (using pretty_midi):
```python
import pretty_midi
import time

# Load and play (requires audio backend)
pm = pretty_midi.PrettyMIDI('generated/lstm/sample_000.mid')
# Convert to audio
audio = pm.fluidsynth()
# Play using your preferred audio library
```

---

## Alternative: Quick Comparison Script

Create a script to compare metrics:

```python
# compare_models.py
import json

with open('models/lstm_baseline/metrics.json') as f:
    lstm_metrics = json.load(f)

with open('models/gpt_improved/metrics.json') as f:
    gpt_metrics = json.load(f)

print("=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print(f"{'Metric':<25} {'LSTM':>12} {'GPT':>12}")
print("-" * 50)
print(f"{'Test Loss':<25} {lstm_metrics['final_test_loss']:>12.4f} {gpt_metrics['final_test_loss']:>12.4f}")
print(f"{'Test Perplexity':<25} {lstm_metrics['final_test_perplexity']:>12.2f} {gpt_metrics['final_test_perplexity']:>12.2f}")
print(f"{'Best Val Loss':<25} {lstm_metrics['best_val_loss']:>12.4f} {gpt_metrics['best_val_loss']:>12.4f}")
print(f"{'Train Windows':<25} {lstm_metrics['train_windows']:>12,} {gpt_metrics['train_windows']:>12,}")
print(f"{'Parameters':<25} {'~2.5M':>12} {'~2.8M':>12}")
```

Run with:
```bash
python compare_models.py
```

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
# LSTM
python src/train_lstm.py --batch-size 16 ...

# GPT
python src/train_gpt_improved.py --batch-size 8 ...
```

### No CUDA Device

Training will automatically fall back to CPU, but will be much slower. For CPU-only training:
```bash
# Set environment variable before running
set PYTORCH_USE_CUDA=0  # Windows
export PYTORCH_USE_CUDA=0  # Linux/Mac
```

### Import Errors

Make sure you're in the project directory and environment is activated:
```bash
cd C:\Users\Adham\Desktop\Eighth Term\DL\music_generation_project\music_generation_project
conda activate music_generation_project
```

### Data Not Found

Verify the tokenized dataset exists:
```bash
dir tokenized\dataset.jsonl
```

If missing, run preprocessing first (see Step 3).

---

## Configuration Options

### LSTM Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--embed-dim` | 256 | Token embedding dimension |
| `--hidden-size` | 512 | LSTM hidden size |
| `--num-layers` | 2 | Number of LSTM layers |
| `--dropout` | 0.2 | Dropout rate |
| `--context-len` | 256 | Context window size |
| `--stride` | 128 | Window stride for long sequences |
| `--lr` | 3e-4 | Learning rate |
| `--warmup-steps` | 500 | LR warmup steps |

### GPT Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--d-model` | 256 | Model dimension |
| `--n-layers` | 4 | Number of Transformer blocks |
| `--n-heads` | 4 | Number of attention heads |
| `--d-ff` | 1024 | Feed-forward dimension |
| `--dropout` | 0.1 | Dropout rate |
| `--max-seq-len` | 256 | Context window size |
| `--stride` | 128 | Window stride |
| `--lr` | 6e-4 | Learning rate |
| `--warmup-steps` | 600 | LR warmup steps |

### Generation Options

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature` | 1.0 | Sampling temperature (<1 = more deterministic) |
| `--top-k` | 0 | Top-k sampling (0 = disabled) |
| `--max-new-tokens` | 512 | Maximum tokens to generate |
| `--num-samples` | 5 | Number of samples to generate |
| `--seed` | 42 | Random seed |

---

## Expected Directory Structure After Running

```
music_generation_project/
├── src/
│   ├── train_lstm.py
│   ├── generate_lstm.py
│   ├── train_gpt_improved.py
│   ├── generate_gpt.py
│   ├── preprocess.py
│   └── midi_tokenize.py
├── tokenized/
│   ├── dataset.jsonl
│   ├── vocab.json
│   └── ...
├── models/
│   ├── lstm_baseline/
│   │   ├── best.pt
│   │   ├── final.pt
│   │   ├── metrics.json
│   │   └── config.json
│   └── gpt_improved/
│       ├── best.pt
│       ├── final.pt
│       ├── metrics.json
│       └── config.json
├── generated/
│   ├── lstm/
│   │   ├── sample_000.mid
│   │   └── ...
│   └── gpt/
│       ├── sample_000.mid
│       └── ...
├── requirements.txt
├── EXPERIMENT_PLAN.md
├── week_2_report.md
└── README_RUN.md  # This file
```

---

## Summary of Commands

**Full pipeline from scratch**:

```bash
# 1. Activate environment
conda activate music_generation_project

# 2. Install dependencies (if needed)
pip install -r requirements.txt

# 3. Train LSTM
python src/train_lstm.py --out-dir models/lstm_baseline

# 4. Train GPT
python src/train_gpt_improved.py --out-dir models/gpt_improved

# 5. Generate LSTM samples
python src/generate_lstm.py --checkpoint models/lstm_baseline/best.pt --output-dir generated/lstm --num-samples 10

# 6. Generate GPT samples
python src/generate_gpt.py --checkpoint models/gpt_improved/best.pt --output-dir generated/gpt --num-samples 10

# 7. Compare metrics
cat models/lstm_baseline/metrics.json
cat models/gpt_improved/metrics.json
```

---

## Support

For issues or questions:
1. Check the `EXPERIMENT_PLAN.md` for detailed experiment design
2. Review `week_2_report.md` for context and results template
3. Examine the docstrings in each Python file for API documentation
