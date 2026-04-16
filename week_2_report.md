# Week 2 Report: Music Generation Model Comparison

**Project**: Symbolic MIDI Music Generation with Event-Based Tokenization
**Week**: 2
**Date**: [Fill in date range]
**Author**: [Fill in your name]

---

## Executive Summary

This week focused on building and comparing two neural architectures for symbolic music generation:
1. **LSTM Baseline**: A strong autoregressive LSTM language model
2. **GPT (Improved)**: An improved decoder-only Transformer

Both models were trained on the same tokenized MIDI dataset using identical data splits to ensure fair comparison.

---

## What Was Done

### 1. Data Pipeline Review

**Existing Pipeline** (from previous week):
- MAESTRO v3.0.0 piano MIDI files
- 16th-note quantization
- Velocity binning (8 discrete bins)
- Event-based tokenization with 288 tokens
- 26,784 tokenized sequences

**Improvements Made**:
- Added proper **piece-level train/val/test splitting** (prevents data leakage)
- Implemented **windowing strategy** that uses ALL sequences (not just long ones)
- Added **padding and masking** for variable-length batching

### 2. LSTM Baseline Implementation

**File**: `src/train_lstm.py`

**Architecture**:
- Embedding dim: 256
- Hidden size: 512
- 2 LSTM layers
- Dropout: 0.2
- Context length: 256 tokens

**Training Features**:
- AdamW optimizer with gradient clipping
- Learning rate warmup + cosine decay
- Mixed precision (AMP) on CUDA
- Early stopping on validation loss
- Checkpoint saving (best + final)

**Generation**: `src/generate_lstm.py`
- Temperature sampling
- Top-k sampling
- Token legality safeguards

### 3. GPT Trainer Improvements

**File**: `src/train_gpt_improved.py`

**Key Improvements over Original**:
1. **All sequences used**: Original excluded sequences shorter than max_seq_len; new version includes all
2. **Systematic windowing**: Replaced random sampling with deterministic windowing (stride=128)
3. **Test split added**: Now has train/val/test instead of just train/val
4. **Precomputed causal mask**: More efficient attention computation
5. **Better metrics**: Comprehensive logging and JSON metrics export

**Architecture**:
- d_model: 256
- 4 layers, 4 heads
- d_ff: 1024
- Dropout: 0.1
- Context length: 256 tokens

**Generation**: `src/generate_gpt.py`
- Same interface as LSTM for easy comparison

### 4. Documentation

Created:
- `EXPERIMENT_PLAN.md`: Detailed experiment design and running instructions
- `week_2_report.md`: This report

---

## Results

### Training Metrics

| Metric | LSTM | GPT (Improved) |
|--------|------|----------------|
| Train windows | [FILL IN] | [FILL IN] |
| Val windows | [FILL IN] | [FILL IN] |
| Test windows | [FILL IN] | [FILL IN] |
| Best val loss | [FILL IN] | [FILL IN] |
| **Test loss** | [FILL IN] | [FILL IN] |
| **Test perplexity** | [FILL IN] | [FILL IN] |
| Training steps | [FILL IN] | [FILL IN] |
| Training time | [FILL IN] | [FILL IN] |

### Model Sizes

| Model | Parameters |
|-------|------------|
| LSTM | ~2.5M |
| GPT | ~2.8M |

### Qualitative Comparison

**LSTM Generated Samples**:
- [FILL IN: Your observations about LSTM-generated music]
- Example observations: "LSTM samples tended to have simpler rhythms but maintained consistent tonality..."

**GPT Generated Samples**:
- [FILL IN: Your observations about GPT-generated music]
- Example observations: "GPT samples showed more complex rhythmic patterns and better long-range structure..."

### Key Findings

1. **Data Efficiency**: [FILL IN: Did one model converge faster?]

2. **Quality**: [FILL IN: Which model produced more musical samples?]

3. **Tradeoffs**: [FILL IN: Any unexpected observations?]

---

## Technical Details

### Hyperparameters Used

**LSTM**:
- Batch size: 32
- Learning rate: 3e-4 (warmup: 500 steps)
- Max steps: 15,000
- Early stopping patience: 15

**GPT**:
- Batch size: 16
- Learning rate: 6e-4 (warmup: 600 steps)
- Max steps: 12,000
- Early stopping patience: 12

### Generation Settings

Both models were evaluated with:
- Temperature: 1.0
- Top-k: 0 (disabled)
- Max new tokens: 512
- Number of samples: 10

---

## Challenges Encountered

1. **[FILL IN]**: Describe any issues with data loading, training, or generation

2. **[FILL IN]**: Any architectural decisions that required iteration

3. **[FILL IN]**: Computational constraints or resource limitations

---

## Lessons Learned

1. **[FILL IN]**: What worked well?

2. **[FILL IN]**: What would you do differently?

3. **[FILL IN]**: Any insights about the models or data?

---

## Next Steps

### Immediate (Week 3)

1. [ ] Run full training for both models (if not completed)
2. [ ] Generate more samples with different temperatures (0.7, 0.9, 1.2)
3. [ ] Conduct listening evaluation with human judges
4. [ ] Analyze token distributions in generated vs. real data

### Future Work

1. [ ] Scale GPT context length to 512 tokens
2. [ ] Experiment with larger models (more layers, wider)
3. [ ] Try top-p (nucleus) sampling
4. [ ] Implement MIDI playback/visualization tools
5. [ ] Fine-tune on complete pieces rather than chunks

---

## Appendix: Commands Used

### Training LSTM

```bash
conda activate music_generation_project
python src/train_lstm.py \
  --data-path tokenized/dataset.jsonl \
  --out-dir models/lstm_baseline \
  --batch-size 32 \
  --max-steps 15000
```

### Training GPT

```bash
conda activate music_generation_project
python src/train_gpt_improved.py \
  --data-path tokenized/dataset.jsonl \
  --out-dir models/gpt_improved \
  --batch-size 16 \
  --max-steps 12000
```

### Generating Samples

```bash
# LSTM generation
python src/generate_lstm.py \
  --checkpoint models/lstm_baseline/best.pt \
  --output-dir generated/lstm \
  --num-samples 10 \
  --temperature 1.0 \
  --seed 42

# GPT generation
python src/generate_gpt.py \
  --checkpoint models/gpt_improved/best.pt \
  --output-dir generated/gpt \
  --num-samples 10 \
  --temperature 1.0 \
  --seed 42
```

---

## References

1. MAESTRO Dataset: https://magenta.tensorflow.org/datasets/maestro
2. PrettyMIDI Library: https://github.com/craffel/pretty-midi
3. PyTorch Documentation: https://pytorch.org/docs/
