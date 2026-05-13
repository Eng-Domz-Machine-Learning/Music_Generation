"""
Improved GPT Training Script for Symbolic MIDI Music Generation

This is an improved version of the original train_gpt.py with:
- Proper train/val/test split at piece level
- Windowing that includes ALL sequences (not just those > max_seq_len)
- Precomputed causal attention mask
- Better logging and metrics saving
- Variable-length support with padding (optional)
- Context length of 256 by default (easier baseline)

Architecture:
- Decoder-only Transformer with causal self-attention
- Learned positional embeddings
- GELU activation in MLP
- LayerNorm for stability
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Prevent local tokenizer scripts from shadowing stdlib tokenize.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """Configuration for GPT model and training."""
    # Data
    data_path: str = "Music_Generation/final_tokenized_data/dataset.jsonl"
    out_dir: str = "models/gpt"
    val_ratio: float = 0.03
    test_ratio: float = 0.03

    # Model architecture
    vocab_size: int = 288  # Will be inferred from data
    max_seq_len: int = 256  # Context length
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1

    # Training
    batch_size: int = 16
    max_steps: int = 24000
    grad_accum_steps: int = 1
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 1200
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Evaluation
    eval_every: int = 500
    eval_batches: int = 25
    save_every: int = 1000
    log_every: int = 50

    # Early stopping
    early_stop_patience: int = 12
    early_stop_min_delta: float = 1e-4

    # Reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Uses a precomputed causal mask for efficiency.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, max_seq_len: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Precomputed causal mask: (1, 1, max_seq_len, max_seq_len)
        # Registered as buffer so it's moved to device automatically
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            ).bool(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, d_model) input tensor

        Returns:
            (batch, seq_len, d_model) output tensor
        """
        bsz, seq_len, d_model = x.shape

        # QKV projection and reshape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * self.scale

        # Apply causal mask (slice to actual sequence length)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        att = att.masked_fill(~mask, float("-inf"))

        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply to values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)

        return self.resid_dropout(self.proj(y))


class TransformerBlock(nn.Module):
    """
    Transformer decoder block with pre-normalization.

    Architecture:
        x -> LayerNorm -> Attention -> x + attn
        x -> LayerNorm -> MLP -> x + mlp
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MusicGPT(nn.Module):
    """
    Decoder-only Transformer for next-token prediction.

    Architecture:
    - Token embedding
    - Learned positional embedding
    - N transformer blocks
    - Final LayerNorm
    - Linear projection to vocabulary
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Embeddings
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # Dropout
        self.drop = nn.Dropout(cfg.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.max_seq_len)
            for _ in range(cfg.n_layers)
        ])

        # Final LayerNorm and output projection
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight initialization
        self.apply(self._init_weights)

        # Report parameter count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[model] GPT total parameters: {total_params:,}", flush=True)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with small random values."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            idx: (batch, seq_len) token IDs

        Returns:
            logits: (batch, seq_len, vocab_size) next-token predictions
        """
        bsz, seq_len = idx.shape

        if seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model max_seq_len {self.cfg.max_seq_len}"
            )

        # Get positional embeddings
        pos = torch.arange(seq_len, device=idx.device).unsqueeze(0)

        # Embed tokens and positions
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.head(x)

        return logits


# ---------------------------------------------------------------------------
# Data Loading
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

    print(f"[split] train={len(train_seqs)} | val={len(val_seqs)} | test={len(test_seqs)} sequences", flush=True)

    if not train_seqs or not val_seqs or not test_seqs:
        raise RuntimeError("Empty split produced - check ratios")

    return train_seqs, val_seqs, test_seqs


class WindowedDataset(Dataset):
    """
    Dataset that creates training windows from sequences.

    Key improvement over original:
    - ALL sequences are used, not just those > max_seq_len
    - Short sequences are kept as single windows
    - Long sequences are split into overlapping windows with stride

    Each window produces:
        input_ids = tokens[:-1]  (predict from these)
        target_ids = tokens[1:]  (predict these)
    """

    def __init__(
        self,
        sequences: List[List[int]],
        context_len: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.stride = stride
        self.sequences = sequences

        # Precompute all windows for deterministic indexing
        self.windows: List[Tuple[int, int, int]] = []  # (seq_idx, start, length)

        for seq_idx, seq in enumerate(sequences):
            seq_len = len(seq)

            if seq_len <= 1:
                continue  # Skip sequences too short

            if seq_len <= context_len + 1:
                # Short sequence: keep as single window
                self.windows.append((seq_idx, 0, seq_len))
            else:
                # Long sequence: create overlapping windows
                start = 0
                while start < seq_len - 1:
                    end = min(start + context_len + 1, seq_len)
                    window_len = end - start
                    if window_len >= 2:
                        self.windows.append((seq_idx, start, window_len))
                    start += stride

                # Ensure coverage at end
                if start < seq_len - 1:
                    final_start = max(0, seq_len - context_len - 1)
                    if final_start < start:
                        final_len = seq_len - final_start
                        if final_len >= 2:
                            self.windows.append((seq_idx, final_start, final_len))

        print(
            f"[dataset] created {len(self.windows)} windows from {len(sequences)} sequences "
            f"(context={context_len}, stride={stride})",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_idx, start, length = self.windows[idx]
        seq = self.sequences[seq_idx]
        window = seq[start:start + length]

        # Input = all but last, Target = all but first
        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        target_ids = torch.tensor(window[1:], dtype=torch.long)

        return input_ids, target_ids


def create_dataloaders(
    train_seqs: List[List[int]],
    val_seqs: List[List[int]],
    test_seqs: List[List[int]],
    context_len: int,
    stride: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/val/test splits."""

    train_dataset = WindowedDataset(train_seqs, context_len, stride)
    val_dataset = WindowedDataset(val_seqs, context_len, stride)
    test_dataset = WindowedDataset(test_seqs, context_len, stride)

    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        input_ids = [item[0] for item in batch]
        target_ids = [item[1] for item in batch]

        # Pad to max length in batch
        input_padded = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        target_padded = nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=0)

        # Mask for loss computation (1 = real token, 0 = padding)
        mask = (input_padded != 0).long()

        return {
            "input_ids": input_padded,
            "target_ids": target_padded,
            "mask": mask,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def compute_masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cross-entropy loss with masking.

    Padded positions don't contribute to the loss.
    """
    batch_size, seq_len, vocab_size = logits.shape

    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    mask_flat = mask.view(-1)

    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    loss = loss * mask_flat.float()

    num_valid = mask_flat.sum().clamp(min=1)
    return loss.sum() / num_valid


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Learning rate schedule with warmup and cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eval_batches: int,
    use_amp: bool,
) -> float:
    """Evaluate model on a subset of data."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= eval_batches:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        target_ids = batch["target_ids"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids)
            loss = compute_masked_loss(logits, target_ids, mask)

        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / max(1, num_batches)


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train improved GPT for tokenized MIDI sequences.")

    p.add_argument("--data-path", default=GPTConfig.data_path)
    p.add_argument("--out-dir", default=GPTConfig.out_dir)
    p.add_argument("--max-steps", type=int, default=GPTConfig.max_steps)
    p.add_argument("--batch-size", type=int, default=GPTConfig.batch_size)
    p.add_argument("--max-seq-len", type=int, default=GPTConfig.max_seq_len)
    p.add_argument("--d-model", type=int, default=GPTConfig.d_model)
    p.add_argument("--n-layers", type=int, default=GPTConfig.n_layers)
    p.add_argument("--n-heads", type=int, default=GPTConfig.n_heads)
    p.add_argument("--d-ff", type=int, default=GPTConfig.d_ff)
    p.add_argument("--dropout", type=float, default=GPTConfig.dropout)
    p.add_argument("--lr", type=float, default=GPTConfig.lr)
    p.add_argument("--min-lr", type=float, default=GPTConfig.min_lr)
    p.add_argument("--warmup-steps", type=int, default=GPTConfig.warmup_steps)
    p.add_argument("--weight-decay", type=float, default=GPTConfig.weight_decay)
    p.add_argument("--grad-clip", type=float, default=GPTConfig.grad_clip)
    p.add_argument("--val-ratio", type=float, default=GPTConfig.val_ratio)
    p.add_argument("--test-ratio", type=float, default=GPTConfig.test_ratio)
    p.add_argument("--eval-every", type=int, default=GPTConfig.eval_every)
    p.add_argument("--eval-batches", type=int, default=GPTConfig.eval_batches)
    p.add_argument("--save-every", type=int, default=GPTConfig.save_every)
    p.add_argument("--log-every", type=int, default=GPTConfig.log_every)
    p.add_argument("--seed", type=int, default=GPTConfig.seed)
    p.add_argument("--early-stop-patience", type=int, default=GPTConfig.early_stop_patience)
    p.add_argument("--early-stop-min-delta", type=float, default=GPTConfig.early_stop_min_delta)
    p.add_argument("--stride", type=int, default=128, help="Stride for windowing")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build config
    cfg = GPTConfig(
        data_path=args.data_path,
        out_dir=args.out_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        grad_accum_steps=1,
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

    # Set up device
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # Set seeds
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # Create output directory
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and split data
    print("=" * 60, flush=True)
    print("Loading dataset...", flush=True)
    sequences = load_dataset(Path(cfg.data_path))

    # Infer vocabulary size
    max_id = max(max(token_ids) for _, token_ids in sequences)
    cfg.vocab_size = max_id + 1
    print(f"[data] vocab_size inferred: {cfg.vocab_size}", flush=True)

    # Split at piece level
    print("[data] splitting at piece level...", flush=True)
    train_seqs, val_seqs, test_seqs = split_data_piece_level(
        sequences, cfg.val_ratio, cfg.test_ratio, cfg.seed
    )

    # Create dataloaders
    print("[data] creating dataloaders...", flush=True)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_seqs, val_seqs, test_seqs,
        context_len=cfg.max_seq_len,
        stride=args.stride,
        batch_size=cfg.batch_size,
    )

    num_train_windows = len(train_loader.dataset)
    num_val_windows = len(val_loader.dataset)
    num_test_windows = len(test_loader.dataset)

    print(
        f"[data] windows: train={num_train_windows}, val={num_val_windows}, test={num_test_windows}",
        flush=True,
    )

    # Create model
    model = MusicGPT(cfg).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    # Grad scaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Training state
    best_val_loss = float("inf")
    no_improve_count = 0
    last_completed_step = 0

    print("=" * 60, flush=True)
    print("Training start", flush=True)
    print("-" * 60, flush=True)
    print(f"Device:            {device}")
    print(f"AMP enabled:       {use_amp}")
    print(f"Context length:    {cfg.max_seq_len}")
    print(f"Batch size:        {cfg.batch_size}")
    print(f"Warmup steps:      {cfg.warmup_steps}")
    print(f"Max steps:         {cfg.max_steps}")
    print(f"Early stop:        {cfg.early_stop_patience} evals")
    print("-" * 60, flush=True)

    last_step_time = time.perf_counter()
    step = 0
    train_iter = iter(train_loader)

    while step < cfg.max_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        target_ids = batch["target_ids"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        # Learning rate
        lr = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.lr, cfg.min_lr)
        set_lr(optimizer, lr)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids)
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

            if val_loss < (best_val_loss - cfg.early_stop_min_delta):
                best_val_loss = val_loss
                no_improve_count = 0

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
            checkpoint_path = out_dir / f"checkpoint_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "config": asdict(cfg),
            }, checkpoint_path)
            print(f"  -> saved checkpoint: {checkpoint_path}", flush=True)

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

    # Final test evaluation
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
