"""
LSTM Baseline Training Script for Symbolic MIDI Music Generation

This script trains an autoregressive LSTM language model over tokenized MIDI sequences.
The model predicts the next token from previous tokens, exactly like language modeling.

Key features:
- Piece-level train/val/test split (chunks from same piece stay together)
- Variable-length sequences with padding and masking
- Windowing for long sequences (context_len=256, stride=128 by default)
- AdamW optimizer with gradient clipping
- Learning rate warmup + cosine decay
- Mixed precision training on CUDA
- Checkpoint saving (best + last)
- Early stopping on validation loss
- Reproducible seeding
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LSTMConfig:
    """Configuration for LSTM model and training."""
    # Data
    data_path: str = "tokenized/dataset.jsonl"
    out_dir: str = "models/lstm"
    val_ratio: float = 0.03
    test_ratio: float = 0.03

    # Model architecture
    vocab_size: int = 288  # Will be inferred from data
    embed_dim: int = 256
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.2

    # Training
    context_len: int = 256
    stride: int = 128
    batch_size: int = 32
    max_steps: int = 15000
    grad_accum_steps: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Evaluation
    eval_every: int = 500
    eval_batches: int = 30
    save_every: int = 1000
    log_every: int = 50

    # Early stopping
    early_stop_patience: int = 15
    early_stop_min_delta: float = 1e-4

    # Reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------

class MusicLSTM(nn.Module):
    """
    Autoregressive LSTM for next-token prediction.

    Architecture:
    - Token embedding layer
    - Multi-layer LSTM with dropout
    - Linear projection to vocabulary size

    The model uses causal masking implicitly: at each position, the LSTM
    only has access to previous tokens through its hidden state.
    """

    def __init__(self, cfg: LSTMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Embedding: token_id -> dense vector
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)

        # LSTM layers: we use batch_first=True for easier handling
        self.lstm = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        # Output projection: hidden_state -> vocabulary logits
        self.output_proj = nn.Linear(cfg.hidden_size, cfg.vocab_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(cfg.dropout)

        # Weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small random values for stable training."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.normal_(param, mean=0.0, std=0.02)

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden and cell states on the target device."""
        h0 = torch.zeros(self.cfg.num_layers, batch_size, self.cfg.hidden_size, device=device)
        c0 = torch.zeros(self.cfg.num_layers, batch_size, self.cfg.hidden_size, device=device)
        return h0, c0

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token IDs
            hidden: Optional (h_0, c_0) hidden state for incremental decoding

        Returns:
            logits: (batch_size, seq_len, vocab_size) next-token predictions
            hidden: (h_n, c_n) final hidden state
        """
        batch_size, seq_len = input_ids.shape

        if hidden is None:
            hidden = self.init_hidden(batch_size, input_ids.device)

        # Embed tokens: (batch, seq) -> (batch, seq, embed_dim)
        x = self.dropout(self.embedding(input_ids))

        # LSTM forward: (batch, seq, embed_dim) -> (batch, seq, hidden_size)
        # hidden = (h_0, c_0) where each has shape (num_layers, batch, hidden_size)
        output, hidden = self.lstm(x, hidden)

        # Apply dropout and project to vocab
        output = self.dropout(output)
        logits = self.output_proj(output)  # (batch, seq, vocab_size)

        return logits, hidden


# ---------------------------------------------------------------------------
# Dataset and Data Loading
# ---------------------------------------------------------------------------

def source_to_piece_id(source_file: str) -> str:
    """
    Collapse chunked filenames back to a piece-level identifier.

    Example:
        2004/MA_01_chunk000.mid -> 2004/MA_01
        bar_chunk127.midi -> bar
    """
    stem = Path(source_file).stem
    return re.sub(r"_chunk\d+$", "", stem)


def load_dataset(path: Path) -> List[Tuple[str, List[int]]]:
    """
    Load tokenized sequences from JSONL file.

    Returns list of (source_file, token_ids) tuples.
    """
    sequences: List[Tuple[str, List[int]]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            source_file = row.get("source_file")
            token_ids = row.get("token_ids")

            if isinstance(source_file, str) and isinstance(token_ids, list) and len(token_ids) >= 2:
                sequences.append((source_file, [int(x) for x in token_ids]))

            if line_idx % 5000 == 0:
                print(f"[data] loaded {line_idx} lines | sequences={len(sequences)}", flush=True)

    print(f"[data] finished loading | total_sequences={len(sequences)}", flush=True)
    return sequences


def split_data_piece_level(
    sequences: List[Tuple[str, List[int]]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    Split data at the PIECE level, not chunk level.

    This ensures that all chunks from the same musical piece stay in the same split,
    preventing data leakage between train/val/test.

    Returns:
        train_seqs, val_seqs, test_seqs (lists of token_id sequences)
    """
    # Group sequences by piece ID
    grouped: Dict[str, List[List[int]]] = {}
    for source_file, token_ids in sequences:
        piece_id = source_to_piece_id(source_file)
        grouped.setdefault(piece_id, []).append(token_ids)

    # Shuffle piece IDs deterministically
    piece_ids = list(grouped.keys())
    random.Random(seed).shuffle(piece_ids)

    # Calculate target counts
    total_sequences = len(sequences)
    target_val = max(1, int(total_sequences * val_ratio))
    target_test = max(1, int(total_sequences * test_ratio))

    # Assign pieces to splits
    val_piece_ids: set[str] = set()
    test_piece_ids: set[str] = set()

    val_count = 0
    test_count = 0

    for piece_id in piece_ids:
        piece_seq_count = len(grouped[piece_id])

        # Fill test set first
        if test_count < target_test:
            test_piece_ids.add(piece_id)
            test_count += piece_seq_count
        # Then validation set
        elif val_count < target_val:
            val_piece_ids.add(piece_id)
            val_count += piece_seq_count
        # Rest goes to train

    # Build final splits
    train_seqs: List[List[int]] = []
    val_seqs: List[List[int]] = []
    test_seqs: List[List[int]] = []

    for piece_id, piece_seqs in grouped.items():
        if piece_id in test_piece_ids:
            test_seqs.extend(piece_seqs)
        elif piece_id in val_piece_ids:
            val_seqs.extend(piece_seqs)
        else:
            train_seqs.extend(piece_seqs)

    print(f"[split] train={len(train_seqs)} | val={len(val_seqs)} | test={len(test_seqs)} pieces", flush=True)

    if not train_seqs or not val_seqs or not test_seqs:
        raise RuntimeError("Empty split produced - check ratios")

    return train_seqs, val_seqs, test_seqs


class WindowedDataset(Dataset):
    """
    Dataset that creates training windows from sequences.

    For sequences longer than context_len, we create multiple overlapping windows
    using the specified stride. Shorter sequences are kept as-is and will be padded
    during batching.

    Each window produces:
        input_ids = tokens[:-1]  (predict from these)
        target_ids = tokens[1:]  (predict these)
    """

    def __init__(
        self,
        sequences: List[List[int]],
        context_len: int,
        stride: int,
        include_bos_eos: bool = True,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.stride = stride
        self.include_bos_eos = include_bos_eos

        # Generate all windows upfront for deterministic indexing
        self.windows: List[Tuple[int, int, int]] = []  # (seq_idx, start, length)

        for seq_idx, seq in enumerate(sequences):
            seq_len = len(seq)

            if seq_len <= 1:
                continue  # Skip sequences too short

            if seq_len <= context_len + 1:
                # Short sequence: keep as single window
                # Add 1 because we need context_len+1 tokens for context_len predictions
                self.windows.append((seq_idx, 0, seq_len))
            else:
                # Long sequence: create multiple overlapping windows
                start = 0
                while start < seq_len - 1:
                    end = min(start + context_len + 1, seq_len)
                    window_len = end - start
                    if window_len >= 2:  # Need at least 2 tokens
                        self.windows.append((seq_idx, start, window_len))
                    start += stride

                # Ensure we cover the end of the sequence
                if start < seq_len - 1:
                    # Add a final window ending at sequence end
                    final_start = max(0, seq_len - context_len - 1)
                    if final_start < start:  # Avoid duplicate
                        final_len = seq_len - final_start
                        if final_len >= 2:
                            self.windows.append((seq_idx, final_start, final_len))

        print(f"[dataset] created {len(self.windows)} windows from {len(sequences)} sequences", flush=True)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_idx, start, length = self.windows[idx]

        # Get the window tokens
        window = self.data_sequences[seq_idx][start:start + length]

        # Convert to tensors
        # input_ids = all tokens except last
        # target_ids = all tokens except first
        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        target_ids = torch.tensor(window[1:], dtype=torch.long)

        return input_ids, target_ids

    def set_sequences(self, sequences: List[List[int]]) -> None:
        """Set the underlying sequences (needed for DataLoader workers)."""
        self.data_sequences = sequences


def create_dataloaders(
    train_seqs: List[List[int]],
    val_seqs: List[List[int]],
    test_seqs: List[List[int]],
    context_len: int,
    stride: int,
    batch_size: int,
    pin_memory: bool,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/val/test splits."""

    # Create datasets
    train_dataset = WindowedDataset(train_seqs, context_len, stride)
    train_dataset.set_sequences(train_seqs)

    val_dataset = WindowedDataset(val_seqs, context_len, stride)
    val_dataset.set_sequences(val_seqs)

    test_dataset = WindowedDataset(test_seqs, context_len, stride)
    test_dataset.set_sequences(test_seqs)

    # Collate function: pad sequences and create attention mask
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [item[0] for item in batch]
        target_ids = [item[1] for item in batch]

        # Pad sequences to max length in batch
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
        target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)

        # Create mask: 1 for real tokens, 0 for padding
        # This is used to mask the loss
        mask = (input_ids_padded != 0).long()

        return {
            "input_ids": input_ids_padded,
            "target_ids": target_ids_padded,
            "mask": mask,
        }

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def compute_masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    Compute cross-entropy loss with masking.

    Padded positions don't contribute to the loss.
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape for cross_entropy: (batch * seq, vocab), (batch * seq)
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    mask_flat = mask.view(-1)

    # Compute loss for all positions
    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")

    # Apply mask: zero out padded positions
    loss = loss * mask_flat.float()

    # Average over non-padded positions
    num_valid = mask_flat.sum().clamp(min=1)
    return loss.sum() / num_valid


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Learning rate schedule with warmup and cosine decay.

    - Warmup: linear increase from 0 to max_lr over warmup_steps
    - Decay: cosine annealing from max_lr to min_lr
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / max(1, warmup_steps)

    # Cosine decay
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move a batch dictionary to the selected device."""
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def autocast_context(use_amp: bool):
    """Return CUDA autocast when available, otherwise a no-op context."""
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eval_batches: int,
    use_amp: bool,
) -> float:
    """Evaluate model on a subset of data, return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= eval_batches:
            break

        batch = move_to_device(batch, device)
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]
        mask = batch["mask"]

        with autocast_context(use_amp):
            logits, _ = model(input_ids)
            loss = compute_masked_loss(logits, target_ids, mask)

        total_loss += loss.item()
        num_batches += 1

    model.train()

    if num_batches == 0:
        return float("inf")

    return total_loss / num_batches


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LSTM for tokenized MIDI sequences.")
    p.add_argument("--data-path", default=LSTMConfig.data_path)
    p.add_argument("--out-dir", default=LSTMConfig.out_dir)
    p.add_argument("--max-steps", type=int, default=LSTMConfig.max_steps)
    p.add_argument("--batch-size", type=int, default=LSTMConfig.batch_size)
    p.add_argument("--context-len", type=int, default=LSTMConfig.context_len)
    p.add_argument("--stride", type=int, default=LSTMConfig.stride)
    p.add_argument("--embed-dim", type=int, default=LSTMConfig.embed_dim)
    p.add_argument("--hidden-size", type=int, default=LSTMConfig.hidden_size)
    p.add_argument("--num-layers", type=int, default=LSTMConfig.num_layers)
    p.add_argument("--dropout", type=float, default=LSTMConfig.dropout)
    p.add_argument("--lr", type=float, default=LSTMConfig.lr)
    p.add_argument("--min-lr", type=float, default=LSTMConfig.min_lr)
    p.add_argument("--warmup-steps", type=int, default=LSTMConfig.warmup_steps)
    p.add_argument("--weight-decay", type=float, default=LSTMConfig.weight_decay)
    p.add_argument("--grad-clip", type=float, default=LSTMConfig.grad_clip)
    p.add_argument("--val-ratio", type=float, default=LSTMConfig.val_ratio)
    p.add_argument("--test-ratio", type=float, default=LSTMConfig.test_ratio)
    p.add_argument("--eval-every", type=int, default=LSTMConfig.eval_every)
    p.add_argument("--eval-batches", type=int, default=LSTMConfig.eval_batches)
    p.add_argument("--save-every", type=int, default=LSTMConfig.save_every)
    p.add_argument("--log-every", type=int, default=LSTMConfig.log_every)
    p.add_argument("--seed", type=int, default=LSTMConfig.seed)
    p.add_argument("--early-stop-patience", type=int, default=LSTMConfig.early_stop_patience)
    p.add_argument("--early-stop-min-delta", type=float, default=LSTMConfig.early_stop_min_delta)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build config from args
    cfg = LSTMConfig(
        data_path=args.data_path,
        out_dir=args.out_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        context_len=args.context_len,
        stride=args.stride,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        grad_accum_steps=1,  # Not used in LSTM version
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        save_every=args.save_every,
        log_every=args.log_every,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )

    # Set up device and precision
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    pin_memory = device.type == "cuda"

    # Set seeds for reproducibility
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    print("=" * 60, flush=True)
    print(f"[device] selected device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[device] gpu: {torch.cuda.get_device_name(0)}", flush=True)

    # Create output directory
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and split data
    print("=" * 60, flush=True)
    print("Loading dataset...", flush=True)
    sequences = load_dataset(Path(cfg.data_path))

    # Infer vocabulary size from data
    max_id = max(max(token_ids) for _, token_ids in sequences)
    cfg.vocab_size = max_id + 1
    print(f"[data] vocab_size inferred: {cfg.vocab_size}", flush=True)

    # Split data at piece level
    print("[data] splitting at piece level...", flush=True)
    train_seqs, val_seqs, test_seqs = split_data_piece_level(
        sequences, cfg.val_ratio, cfg.test_ratio, cfg.seed
    )

    # Create dataloaders
    print("[data] creating dataloaders...", flush=True)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_seqs, val_seqs, test_seqs,
        context_len=cfg.context_len,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        pin_memory=pin_memory,
    )

    # Count windows
    num_train_windows = len(train_loader.dataset)
    num_val_windows = len(val_loader.dataset)
    num_test_windows = len(test_loader.dataset)

    print(f"[data] windows: train={num_train_windows}, val={num_val_windows}, test={num_test_windows}", flush=True)

    # Create model
    model = MusicLSTM(cfg).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] LSTM parameters: {num_params:,}", flush=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    # Grad scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Training state
    best_val_loss = float("inf")
    no_improve_count = 0
    last_completed_step = 0

    print("=" * 60, flush=True)
    print("Training start", flush=True)
    print("-" * 60, flush=True)
    print(f"Device:            {device}")
    if device.type == "cuda":
        print(f"GPU:               {torch.cuda.get_device_name(0)}")
    print(f"AMP enabled:       {use_amp}")
    print(f"Pin memory:        {pin_memory}")
    print(f"Context length:    {cfg.context_len}")
    print(f"Batch size:        {cfg.batch_size}")
    print(f"Warmup steps:      {cfg.warmup_steps}")
    print(f"Max steps:         {cfg.max_steps}")
    print(f"Early stop:        {cfg.early_stop_patience} evals")
    print("-" * 60, flush=True)

    last_step_time = time.perf_counter()
    step = 0

    # Data iterator for continuous training
    train_iter = iter(train_loader)

    while step < cfg.max_steps:
        # Get next batch (handle epoch boundary)
        try:
            batch = next(train_iter)
        except StopIteration:
            # Epoch ended, restart iterator
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Move to device
        batch = move_to_device(batch, device)
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]
        mask = batch["mask"]

        # Learning rate schedule
        lr = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)
        set_lr(optimizer, lr)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(use_amp):
            logits, _ = model(input_ids)
            loss = compute_masked_loss(logits, target_ids, mask)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Logging
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            now = time.perf_counter()
            step_dt = now - last_step_time
            last_step_time = now

            print(
                f"step {step:06d} | train_loss {loss.item():.4f} | "
                f"lr {lr:.6e} | grad_norm {float(grad_norm):.4f} | "
                f"step_time {step_dt:.2f}s",
                flush=True,
            )

        # Evaluation
        if step > 0 and step % cfg.eval_every == 0:
            val_loss = evaluate(model, val_loader, device, cfg.eval_batches, use_amp)
            print(f"step {step:06d} | val_loss {val_loss:.4f}", flush=True)

            # Check for improvement
            if val_loss < (best_val_loss - cfg.early_stop_min_delta):
                best_val_loss = val_loss
                no_improve_count = 0

                # Save best checkpoint
                best_path = out_dir / "best.pt"
                torch.save({
                    "step": step,
                    "val_loss": val_loss,
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                }, best_path)
                print(f"  -> saved best checkpoint: {best_path}", flush=True)
            else:
                no_improve_count += 1
                print(
                    f"  -> no improvement for {no_improve_count} eval(s) "
                    f"(patience={cfg.early_stop_patience})",
                    flush=True,
                )

                if cfg.early_stop_patience > 0 and no_improve_count >= cfg.early_stop_patience:
                    print("  -> early stopping triggered", flush=True)
                    break

        # Periodic checkpoint
        if step > 0 and step % cfg.save_every == 0:
            last_path = out_dir / f"checkpoint_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "config": asdict(cfg),
            }, last_path)
            print(f"  -> saved checkpoint: {last_path}", flush=True)

        step += 1
        last_completed_step = step

    # Save final checkpoint
    final_path = out_dir / "final.pt"
    torch.save({
        "step": last_completed_step,
        "model_state": model.state_dict(),
        "config": asdict(cfg),
    }, final_path)
    print(f"training complete -> {final_path}", flush=True)

    # Final evaluation on test set
    print("=" * 60, flush=True)
    print("Final evaluation on test set...", flush=True)
    test_loss = evaluate(model, test_loader, device, cfg.eval_batches, use_amp)
    test_perplexity = math.exp(test_loss)

    print(f"Test loss:      {test_loss:.4f}", flush=True)
    print(f"Test perplexity: {test_perplexity:.2f}", flush=True)

    # Save metrics
    metrics = {
        "train_windows": num_train_windows,
        "val_windows": num_val_windows,
        "test_windows": num_test_windows,
        "best_val_loss": best_val_loss,
        "final_test_loss": test_loss,
        "final_test_perplexity": test_perplexity,
        "total_steps": last_completed_step,
        "vocab_size": cfg.vocab_size,
        "num_parameters": num_params,
    }

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"metrics saved -> {metrics_path}", flush=True)

    # Save config
    config_path = out_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"config saved -> {config_path}", flush=True)

    print("=" * 60, flush=True)
    print("Training finished successfully!", flush=True)


if __name__ == "__main__":
    main()
