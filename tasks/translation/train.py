from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from tasks.translation.dataset import build_en_de_dataloaders
from transformer import OriginalTransformer, TransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer for EN->DE translation")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "dataset" / "iwslt2023da.csv"),
        help="Path to iwslt2023da.csv",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-quality", type=int, default=None)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-vocab-size", type=int, default=20000)
    parser.add_argument("--max-src-len", type=int, default=128)
    parser.add_argument("--max-tgt-len", type=int, default=128)

    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pre-ln", action="store_true")
    parser.add_argument("--tie-weights", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    parser.add_argument("--save-dir", type=str, default=str(ROOT_DIR / "checkpoints" / "translation"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: OriginalTransformer,
    dataloader: Any,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        src_ids = batch["src_ids"].to(device)
        src_key_padding_mask = batch["src_key_padding_mask"].to(device)
        tgt_in_ids = batch["tgt_in_ids"].to(device)
        tgt_out_ids = batch["tgt_out_ids"].to(device)
        tgt_key_padding_mask = batch["tgt_key_padding_mask"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(
                src=src_ids,
                tgt=tgt_in_ids,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out_ids.reshape(-1))

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        non_pad_tokens = tgt_out_ids.ne(criterion.ignore_index).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = build_en_de_dataloaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        min_quality=args.min_quality,
        unique_pairs_only=not args.keep_duplicates,
    )
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    src_vocab = data["src_vocab"]
    tgt_vocab = data["tgt_vocab"]

    cfg = TransformerConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pre_ln=args.pre_ln,
        max_len=max(args.max_src_len, args.max_tgt_len) + 2,
    )
    model = OriginalTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        cfg=cfg,
        tie_weights=args.tie_weights,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_vocab.pad_id,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    bad_epochs = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_ppl": [],
        "val_loss": [],
        "val_ppl": [],
        "lr": [],
    }

    print(
        f"train_size={data['train_size']} val_size={data['val_size']} "
        f"src_vocab={len(src_vocab)} tgt_vocab={len(tgt_vocab)}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_loss, val_ppl = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} "
            f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        history["train_loss"].append(float(train_loss))
        history["train_ppl"].append(float(train_ppl))
        history["val_loss"].append(float(val_loss))
        history["val_ppl"].append(float(val_ppl))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step(val_loss)

        latest_ckpt = save_dir / "latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg.__dict__,
                "src_vocab_stoi": src_vocab.stoi,
                "src_vocab_itos": src_vocab.itos,
                "tgt_vocab_stoi": tgt_vocab.stoi,
                "tgt_vocab_itos": tgt_vocab.itos,
                "args": vars(args),
                "val_loss": val_loss,
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
            },
            latest_ckpt,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            best_ckpt = save_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg.__dict__,
                    "src_vocab_stoi": src_vocab.stoi,
                    "src_vocab_itos": src_vocab.itos,
                    "tgt_vocab_stoi": tgt_vocab.stoi,
                    "tgt_vocab_itos": tgt_vocab.itos,
                    "args": vars(args),
                    "val_loss": val_loss,
                    "scheduler_state_dict": scheduler.state_dict(),
                    "history": history,
                },
                best_ckpt,
            )
            print(f"saved best checkpoint: {best_ckpt}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop_patience:
                print(
                    "early stopping triggered: "
                    f"{bad_epochs} epochs without validation improvement"
                )
                break

    metrics_path = save_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_ppl", "val_ppl", "lr"])
        for idx in range(len(history["train_loss"])):
            writer.writerow(
                [
                    idx + 1,
                    history["train_loss"][idx],
                    history["val_loss"][idx],
                    history["train_ppl"][idx],
                    history["val_ppl"][idx],
                    history["lr"][idx],
                ]
            )
    print(f"saved metrics: {metrics_path}")

    print(f"training complete, best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
