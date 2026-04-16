# Music Generation Model Comparison: Experiment Plan

## Overview

This document describes the experimental setup for comparing LSTM and GPT (decoder-only Transformer) models for symbolic MIDI music generation using event-based tokenized sequences.

## Dataset

### Source Data
- **Dataset**: MAESTRO v3.0.0 (piano MIDI files)
- **Preprocessing**: 
  1. Piano-only extraction (programs 0-7)
  2. 16th-note quantization
  3. Velocity binning (8 bins)
  4. 30-second chunking (minimum 10 notes per chunk)

### Tokenization
- **Vocabulary size**: 288 tokens
- **Token types**:
  - Special: PAD(0), BOS(1), EOS(2), UNK(3)
  - Velocity: VELOCITY_0 to VELOCITY_7 (8 tokens)
  - Time shift: TIME_SHIFT_1 to TIME_SHIFT_100 (100 tokens)
  - Note on: NOTE_ON_21 to NOTE_ON_108 (88 tokens)
  - Note off: NOTE_OFF_21 to NOTE_OFF_108 (88 tokens)

### Data Statistics
- **Total tokenized files**: 26,784
- **Average sequence length**: ~692 tokens
- **Sequence length range**: 29 - 1,929 tokens

### Data Split
Both models use **identical piece-level splits** to ensure fair comparison:
- **Train**: ~94% of pieces
- **Validation**: ~3% of pieces
- **Test**: ~3% of pieces

Piece-level splitting ensures all chunks from the same musical piece stay in the same split, preventing data leakage.

---

## Model Architectures

### LSTM Baseline

| Component | Specification |
|-----------|---------------|
| Embedding dim | 256 |
| Hidden size | 512 |
| Num layers | 2 |
| Dropout | 0.2 |
| Context length | 256 tokens |
| Parameters | ~2.5M |

**Architecture Details**:
- Token embedding layer
- 2-layer LSTM with dropout between layers
- Linear projection to vocabulary size
- Causal by nature (hidden state only contains past information)

**Key Design Decisions**:
1. **Windowing**: Long sequences are split into overlapping windows (stride=128)
2. **Padding**: Variable-length sequences are padded and masked during batching
3. **Masked loss**: Padded positions don't contribute to loss

### GPT (Improved Decoder-Only Transformer)

| Component | Specification |
|-----------|---------------|
| Model dim (d_model) | 256 |
| Num layers | 4 |
| Num heads | 4 |
| FFN dim (d_ff) | 1024 |
| Dropout | 0.1 |
| Context length | 256 tokens |
| Parameters | ~2.8M |

**Architecture Details**:
- Token embedding + learned positional embedding
- 4 Transformer blocks with causal self-attention
- Pre-normalization (LayerNorm before each sublayer)
- GELU activation in MLP
- Precomputed causal mask for efficiency

**Improvements over original**:
1. **All sequences used**: Original only used sequences > max_seq_len; now all sequences contribute
2. **Proper windowing**: Systematic windowing with stride instead of random sampling
3. **Test split added**: Original had only train/val
4. **Precomputed causal mask**: More efficient than rebuilding every forward pass
5. **Better logging**: More detailed metrics and checkpointing

---

## Training Configuration

### Common Settings

| Hyperparameter | Value |
|----------------|-------|
| Context length | 256 tokens |
| Batch size | 16-32 |
| Optimizer | AdamW |
| Gradient clipping | 1.0 |
| Mixed precision | Yes (CUDA) |
| Early stopping | Yes (patience=12-15) |

### LSTM Training

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 3e-4 |
| Min learning rate | 3e-5 |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Max steps | 15,000 |
| Eval every | 500 steps |

### GPT Training

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 6e-4 |
| Min learning rate | 6e-5 |
| Warmup steps | 600 |
| Weight decay | 0.1 |
| Max steps | 12,000 |
| Eval every | 500 steps |

### Learning Rate Schedule

Both models use **warmup + cosine decay**:
1. **Warmup**: Linear increase from 0 to max_lr over warmup_steps
2. **Decay**: Cosine annealing from max_lr to min_lr

---

## Evaluation Metrics

### Quantitative Metrics

1. **Test Loss** (cross-entropy)
   - Lower is better
   - Measures average negative log-likelihood per token

2. **Test Perplexity**
   - Computed as exp(test_loss)
   - Lower is better
   - More interpretable: "effective branching factor"

3. **Training Efficiency**
   - Steps to convergence
   - Wall-clock training time

### Qualitative Metrics

1. **Musicality**
   - Do generated sequences sound coherent?
   - Are there obvious errors (repeated notes, unnatural rhythms)?

2. **Structure**
   - Does the model capture phrase structure?
   - Are there repeating motifs?

3. **Creativity**
   - Does the model generate novel patterns?
   - Or does it just copy training data?

---

## Generation Configuration

Both models support the same generation parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_new_tokens | 512 | Maximum tokens to generate |
| temperature | 1.0 | Sampling temperature (lower = more deterministic) |
| top_k | 0 | Top-k sampling (0 = disabled) |
| seed | 42 | Random seed for reproducibility |

### Token Legality Safeguards

During generation:
1. Invalid token IDs (> vocab_size) are masked out
2. NOTE_OFF tokens for inactive pitches receive a penalty (-2.0 logits)
3. This helps prevent "orphan" note-offs (releasing notes that weren't played)

---

## Expected Tradeoffs

### LSTM Advantages
- **Simpler architecture**: Easier to understand and debug
- **Lower memory**: Fewer parameters, no attention matrix
- **Sequential nature**: Natural for autoregressive generation

### LSTM Disadvantages
- **Limited context**: Hidden state must compress all history
- **Sequential training**: Cannot parallelize over sequence positions
- **Vanishing gradients**: May struggle with long-range dependencies

### GPT Advantages
- **Self-attention**: Direct access to all previous tokens
- **Parallel training**: All positions computed simultaneously
- **Long-range dependencies**: Better at capturing distant relationships

### GPT Disadvantages
- **Higher memory**: Attention matrix is O(seq_len^2)
- **No inherent position bias**: Must learn positional embeddings
- **More complex**: More hyperparameters to tune

---

## Running the Experiments

### Step 1: Train LSTM Baseline

```bash
cd music_generation_project
conda activate music_generation_project
python src/train_lstm.py --out-dir models/lstm_baseline
```

### Step 2: Train GPT (Improved)

```bash
cd music_generation_project
conda activate music_generation_project
python src/train_gpt_improved.py --out-dir models/gpt_improved
```

### Step 3: Generate Samples from LSTM

```bash
python src/generate_lstm.py \
  --checkpoint models/lstm_baseline/best.pt \
  --output-dir generated/lstm \
  --num-samples 10 \
  --max-new-tokens 512 \
  --temperature 1.0 \
  --seed 42
```

### Step 4: Generate Samples from GPT

```bash
python src/generate_gpt.py \
  --checkpoint models/gpt_improved/best.pt \
  --output-dir generated/gpt \
  --num-samples 10 \
  --max-new-tokens 512 \
  --temperature 1.0 \
  --seed 42
```

### Step 5: Compare Results

1. Compare test loss and perplexity from both models
2. Listen to generated MIDI samples
3. Fill in the results section in `week_2_report.md`

---

## Files Created

```
src/
  train_lstm.py          # LSTM baseline training script
  generate_lstm.py       # LSTM generation script
  train_gpt_improved.py  # Improved GPT training script
  generate_gpt.py        # GPT generation script

models/
  lstm_baseline/         # LSTM checkpoints and metrics
  gpt_improved/          # GPT checkpoints and metrics

generated/
  lstm/                  # LSTM-generated MIDI files
  gpt/                   # GPT-generated MIDI files

EXPERIMENT_PLAN.md       # This document
week_2_report.md         # Results report template
```

---

## Success Criteria

The experiment is successful if:

1. Both models train without errors and converge
2. Test perplexity is reasonable (< 10 for both models)
3. Generated samples are musically coherent (contain actual melodies)
4. GPT shows improvement over LSTM in at least one metric (loss, perplexity, or quality)

---

## Next Steps After Baseline

Once baselines are established:

1. **Scale GPT**: Increase context length to 512 tokens
2. **Ablation studies**: Test impact of windowing stride, batch size
3. **Fine-tuning**: Train on full pieces instead of chunks
4. **Conditional generation**: Add style or composer conditioning
