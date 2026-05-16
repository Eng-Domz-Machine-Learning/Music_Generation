# Classical Piano Music Generation

Deep learning project for symbolic classical piano music generation. The system preprocesses raw MIDI files, tokenizes them into event sequences, trains autoregressive sequence models, and serves the final GPT model through a Gradio app that generates downloadable MIDI files.

The final model is a decoder-only Transformer trained on event-tokenized piano data from MAESTRO v3.0.0 and a supplementary Kaggle Classical MIDI dataset.

## Highlights

- End-to-end MIDI pipeline: raw MIDI ingestion, cleaning, tokenization, training, generation, and app deployment.
- Event-based vocabulary with 288 tokens covering note on/off events, time shifts, velocity bins, and special tokens.
- Piece-level train/validation/test splitting to avoid leakage between related segments.
- Final GPT model scaled to about 25.63M parameters and trained for 24,000 steps.
- Generation controls for temperature, top-k, top-p, repetition penalty, no-repeat n-grams, tempo bias, and token legality guidance.
- Gradio interface for generating 1-5 MIDI samples with optional audio preview.
- Hugging Face Space target: `adham2oo3/music-generation-gpt`.

## Results

| Metric | Final GPT |
| --- | ---: |
| Train windows | 147,473 |
| Validation windows | 4,603 |
| Test windows | 4,792 |
| Vocabulary size | 288 |
| Best validation loss | 1.7376 |
| Final test loss | 1.5802 |
| Final test perplexity | 4.8557 |
| Training steps | 24,000 |
| Parameters | ~25.63M |

## Model

| Hyperparameter | Value |
| --- | ---: |
| Architecture | Decoder-only Transformer |
| `d_model` | 512 |
| Layers | 8 |
| Attention heads | 8 |
| Feed-forward dimension | 2,048 |
| Context length | 256 tokens |
| Dropout | 0.1 |

An LSTM baseline was implemented for comparison. The final report notes that the GPT model produced richer rhythm, stronger tonal consistency over longer outputs, and more reliable polyphony, so the final system focuses on the Transformer checkpoint.

## Dataset And Tokenization

The project combines two symbolic piano MIDI sources:

| Source | Raw MIDI files | Processed chunks |
| --- | ---: | ---: |
| MAESTRO v3.0.0 | 1,276 | 24,271 |
| Kaggle Classical MIDI | 295 | 2,513 |
| Total | 1,571 | 26,784 |

Preprocessing stages:

1. Keep piano instruments only.
2. Merge piano tracks into one note stream.
3. Quantize note onsets and offsets to a 16th-note grid.
4. Clip pitches to the standard piano range, MIDI 21-108.
5. Bin velocities into 8 discrete levels.
6. Split pieces into 30-second non-overlapping chunks.
7. Remove chunks with fewer than 10 notes.

Tokenized sequences are split into overlapping 256-token windows with stride 128.

## Project Structure

```text
.
|-- app/                         # Gradio interface and inference wrapper
|-- src/                         # Preprocessing, training, and generation scripts
|-- models/
|   |-- gpt/                     # Final GPT config, metrics, and checkpoints
|   `-- lstm_baseline/           # LSTM baseline artifacts
|-- tokenized/                   # Vocabulary and tokenized dataset artifacts
|-- generated/                   # Generated MIDI samples
|-- Final_progress_report.md
|-- README_RUN.md                # Longer runbook-style execution notes
`-- README.md
```

Large data files and trained checkpoints may be absent from a fresh clone depending on how the repository is distributed. For inference, the required runtime artifacts are:

- `models/gpt/best.pt`
- `tokenized/id_to_token.json`
- `tokenized/vocab.json`

## Setup

Create and activate a Python environment:

```bash
conda create -n music_generation_project python=3.10
conda activate music_generation_project
```

Install core dependencies:

```bash
pip install -r requirements.txt
```

For the Gradio app, install the app dependencies as well:

```bash
pip install -r app_requirements.txt
```

If you are training on an NVIDIA GPU, install a CUDA-compatible PyTorch build for your machine.

## Run The Gradio App

```bash
python app.py
```

Then open:

```text
http://localhost:7860
```

The app exposes controls for sample count, output length, temperature, top-k, top-p, repetition penalty, no-repeat n-gram size, tempo preset, and seed. Generated MIDI files are saved under `generated/gradio_output/` and returned in the UI.

## Generate From The CLI

```bash
python src/generate_gpt.py \
  --checkpoint models/gpt/best.pt \
  --vocab-path tokenized/id_to_token.json \
  --output-dir generated/gpt \
  --num-samples 5 \
  --max-new-tokens 512 \
  --temperature 0.82 \
  --top-p 0.9 \
  --repetition-penalty 1.15 \
  --no-repeat-ngram-size 4 \
  --tempo normal \
  --seed 42 \
  --save-tokens
```

Tempo choices are `slow`, `normal`, and `fast`.

## Preprocess Data

Place raw MIDI data under `data_raw/`, then run:

```bash
python src/preprocess.py \
  --input-dir data_raw/maestro \
  --output-dir data_processed

python src/midi_tokenize.py \
  --input-dir data_processed \
  --output-dir tokenized \
  --verify-sample 10
```

Expected tokenization artifacts include:

- `tokenized/dataset.jsonl`
- `tokenized/vocab.json`
- `tokenized/id_to_token.json`
- `tokenized/stats.json`

## Train Models

Train the LSTM baseline:

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

Train the final-style GPT:

```bash
python src/train_gpt_improved.py \
  --data-path tokenized/dataset.jsonl \
  --out-dir models/gpt \
  --batch-size 16 \
  --d-model 512 \
  --n-layers 8 \
  --n-heads 8 \
  --d-ff 2048 \
  --max-seq-len 256 \
  --max-steps 24000 \
  --seed 42
```

Training outputs are written to the selected model directory:

- `best.pt`
- `final.pt`
- `metrics.json`
- `config.json`

## Validation

A final smoke test was run with:

```bash
python src/generate_gpt.py \
  --checkpoint models/gpt/best.pt \
  --vocab-path tokenized/id_to_token.json \
  --max-new-tokens 32 \
  --num-samples 1 \
  --output-dir generated/smoke_gpt
```

The checkpoint loaded successfully, generation completed, and `generated/smoke_gpt/sample_000.mid` was produced.

## Reports

- [Final_progress_report.md](Final_progress_report.md) - concise final delivery summary.
- [README_RUN.md](README_RUN.md) - detailed installation and execution notes.
- [SPACES_README.md](SPACES_README.md) - Hugging Face Spaces README content.

## Authors

- Adham Ahmed
- Eyad Ahmed
- Belal Mohamed
- Hazem Hassan

Alexandria University, Faculty of Engineering, Department of Computer and Communication Engineering. Deep Learning Course, Spring 2026.
