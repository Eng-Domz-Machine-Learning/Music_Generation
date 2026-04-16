from __future__ import annotations

import os
import sys
import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import List, Sequence

# Prevent local tokenizer scripts from shadowing stdlib tokenize.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainConfig:
    data_path: str = "tokenized/dataset.jsonl"
    out_dir: str = "models"
    seed: int = 42
    val_ratio: float = 0.03
    max_seq_len: int = 512
    vocab_size: int = 288
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    batch_size: int = 16
    grad_accum_steps: int = 4
    max_steps: int = 12000
    eval_every: int = 200
    eval_batches: int = 25
    save_every: int = 500
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 600
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 1
    early_stop_patience: int = 12
    early_stop_min_delta: float = 1e-4


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.resid_dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
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
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = idx.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError("Sequence length exceeds model max_seq_len.")
        pos = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GPT decoder for tokenized MIDI sequences.")
    p.add_argument("--data-path", default=TrainConfig.data_path)
    p.add_argument("--out-dir", default=TrainConfig.out_dir)
    p.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--grad-accum-steps", type=int, default=TrainConfig.grad_accum_steps)
    p.add_argument("--max-seq-len", type=int, default=TrainConfig.max_seq_len)
    p.add_argument("--d-model", type=int, default=TrainConfig.d_model)
    p.add_argument("--n-layers", type=int, default=TrainConfig.n_layers)
    p.add_argument("--n-heads", type=int, default=TrainConfig.n_heads)
    p.add_argument("--d-ff", type=int, default=TrainConfig.d_ff)
    p.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--min-lr", type=float, default=TrainConfig.min_lr)
    p.add_argument("--warmup-steps", type=int, default=TrainConfig.warmup_steps)
    p.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    p.add_argument("--val-ratio", type=float, default=TrainConfig.val_ratio)
    p.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    p.add_argument("--eval-batches", type=int, default=TrainConfig.eval_batches)
    p.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    p.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=TrainConfig.early_stop_patience,
        help="Number of eval checks without improvement before stopping (0 disables).",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=TrainConfig.early_stop_min_delta,
        help="Minimum val loss improvement to reset early stopping patience.",
    )
    return p.parse_args()


def source_to_piece_id(source_file: str) -> str:
    """
    Collapse chunked filenames back to a piece-level identifier.

    Example:
    - foo_chunk000.mid -> foo
    - bar_chunk127.midi -> bar
    """
    stem = Path(source_file).stem
    return re.sub(r"_chunk\d+$", "", stem)


def load_dataset(path: Path) -> List[tuple[str, List[int]]]:
    sequences: List[tuple[str, List[int]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_file = row.get("source_file")
            ids = row.get("token_ids")
            if isinstance(source_file, str) and isinstance(ids, list) and len(ids) >= 2:
                sequences.append((source_file, [int(x) for x in ids]))
            if line_idx % 2000 == 0:
                print(f"[data] loaded {line_idx} lines | usable_sequences={len(sequences)}", flush=True)
    if not sequences:
        raise RuntimeError(f"No usable token sequences found in {path}")
    print(f"[data] finished loading | total_sequences={len(sequences)}", flush=True)
    return sequences


def split_data(
    sequences: Sequence[tuple[str, List[int]]],
    val_ratio: float,
    seed: int,
) -> tuple[list[list[int]], list[list[int]]]:
    grouped: dict[str, list[list[int]]] = {}
    for source_file, token_ids in sequences:
        piece_id = source_to_piece_id(source_file)
        grouped.setdefault(piece_id, []).append(token_ids)

    piece_ids = list(grouped.keys())
    random.Random(seed).shuffle(piece_ids)

    total_sequences = len(sequences)
    target_val_sequences = max(1, int(total_sequences * val_ratio))

    val_piece_ids: set[str] = set()
    val_count = 0
    for piece_id in piece_ids:
        if val_count >= target_val_sequences and val_piece_ids:
            break
        val_piece_ids.add(piece_id)
        val_count += len(grouped[piece_id])

    train: list[list[int]] = []
    val: list[list[int]] = []
    for piece_id, piece_sequences in grouped.items():
        if piece_id in val_piece_ids:
            val.extend(piece_sequences)
        else:
            train.extend(piece_sequences)

    if not train or not val:
        raise RuntimeError("Piece-level split produced an empty train or validation set.")

    return train, val


def sample_batch(
    sequences: Sequence[List[int]], batch_size: int, seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = [s for s in sequences if len(s) > seq_len]
    if not valid:
        raise RuntimeError(f"No sequences long enough for seq_len={seq_len}.")
    x_list = []
    y_list = []
    for _ in range(batch_size):
        seq = valid[random.randrange(len(valid))]
        start = random.randrange(0, len(seq) - seq_len)
        chunk = seq[start : start + seq_len + 1]
        x_list.append(torch.tensor(chunk[:-1], dtype=torch.long))
        y_list.append(torch.tensor(chunk[1:], dtype=torch.long))
    x = torch.stack(x_list, dim=0).to(device, non_blocking=True)
    y = torch.stack(y_list, dim=0).to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def estimate_val_loss(
    model: nn.Module,
    val_sequences: Sequence[List[int]],
    eval_batches: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = sample_batch(val_sequences, batch_size, seq_len, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def get_lr(step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        data_path=args.data_path,
        out_dir=args.out_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        save_every=args.save_every,
        log_every=args.log_every,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    data_path = Path(cfg.data_path)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    sequences = load_dataset(data_path)
    train_sequences, val_sequences = split_data(sequences, cfg.val_ratio, cfg.seed)

    max_id = max(max(token_ids) for _, token_ids in sequences)
    if max_id + 1 > cfg.vocab_size:
        cfg.vocab_size = max_id + 1

    model = MusicGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    no_improve_evals = 0
    steps_per_update = cfg.grad_accum_steps
    last_completed_step = -1

    print("Training start", flush=True)
    print("-" * 50, flush=True)
    print(f"Device:            {device}", flush=True)
    print(f"Train sequences:   {len(train_sequences)}", flush=True)
    print(f"Val sequences:     {len(val_sequences)}", flush=True)
    print(f"Max seq len:       {cfg.max_seq_len}", flush=True)
    print(f"Vocab size:        {cfg.vocab_size}", flush=True)
    print(f"Model dim/layers:  {cfg.d_model}/{cfg.n_layers}", flush=True)

    last_step_time = time.perf_counter()
    for step in range(cfg.max_steps):
        last_completed_step = step
        lr = get_lr(step, cfg.max_steps, cfg.warmup_steps, cfg.lr, cfg.min_lr)
        set_lr(optimizer, lr)

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for _ in range(steps_per_update):
            x, y = sample_batch(train_sequences, cfg.batch_size, cfg.max_seq_len, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / steps_per_update
            running_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            now = time.perf_counter()
            step_dt = now - last_step_time
            last_step_time = now
            print(
                f"step {step:06d} | train_loss {running_loss:.4f} | lr {lr:.6e} | "
                f"grad_norm {float(grad_norm):.4f} | step_time {step_dt:.2f}s",
                flush=True,
            )

        if step > 0 and step % cfg.eval_every == 0:
            val_loss = estimate_val_loss(
                model,
                val_sequences,
                cfg.eval_batches,
                cfg.batch_size,
                cfg.max_seq_len,
                device,
                use_amp,
            )
            print(f"step {step:06d} | val_loss {val_loss:.4f}", flush=True)
            if val_loss < (best_val - cfg.early_stop_min_delta):
                best_val = val_loss
                no_improve_evals = 0
                best_path = out_dir / "best.pt"
                torch.save(
                    {
                        "step": step,
                        "val_loss": val_loss,
                        "model_state": model.state_dict(),
                        "config": asdict(cfg),
                    },
                    best_path,
                )
                print(f"saved best checkpoint -> {best_path}", flush=True)
            else:
                no_improve_evals += 1
                print(
                    f"no val improvement for {no_improve_evals} eval(s) "
                    f"(patience={cfg.early_stop_patience})",
                    flush=True,
                )
                if cfg.early_stop_patience > 0 and no_improve_evals >= cfg.early_stop_patience:
                    print("early stopping triggered", flush=True)
                    break

        if step > 0 and step % cfg.save_every == 0:
            last_path = out_dir / "last.pt"
            torch.save(
                {
                    "step": step,
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                },
                last_path,
            )
            print(f"saved last checkpoint -> {last_path}", flush=True)

    final_path = out_dir / "final.pt"
    torch.save(
        {
            "step": last_completed_step,
            "model_state": model.state_dict(),
            "config": asdict(cfg),
        },
        final_path,
    )
    print(f"training complete -> {final_path}", flush=True)


if __name__ == "__main__":
    main()
