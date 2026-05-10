"""
Training script for the conditional DDPM on iCLEVR.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume ckpts/ckpt_epoch010.pt
"""
import argparse
import os 
from pathlib import Path 

import torch 
import torch.nn as nn 
from torch.optim import AdamW 
from torch.optim.lr_scheduler import LambdaLR 
from tqdm import tqdm

from data.dataset import get_train_loader
from diffusion.noise_schedule import NoiseSchedule
from models.ddpm import DDPM
from models.unet import UNet
from utils.config import load_config
from utils.ema import EMA
from utils.early_stopping import EarlyStopping
from utils.utils import save_checkpoint, load_checkpoint

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional DDPM on iCLEVR.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None, 
        help="Path to a checkpoint to resume training from (optional)."
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# LR schedule: linear warmup then constant
# ---------------------------------------------------------------------------
def build_lr_scheduler(optimizer, warmup_steps: int):
    """Linear warmup for ``warmup_steps`` gradient steps, then constant LR.

    Args:
        optimizer (_type_): The optimizer to wrap. 
        warmup_steps (int): Number of gradient steps for linear warmup. 
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    return LambdaLR(optimizer, lr_lambda)

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(cfg: dict, resume_path: str = "") -> None:
    """Run the full DDPM training loop.

    Orchestrates data loading, model construction, EMA, optimisation,
    mixed-precision training, checkpointing, and early stopping.

    Args:
        cfg (dict): Parsed config dictionary from ``load_config``.
        resume_path (str): Optional path to a checkpoint to resume from.
            If empty, falls back to ``cfg["training"]["resume_ckpt"]``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Config aliases ──────────────────────────────────────────────────── #
    path_cfg  = cfg["paths"]
    data_cfg  = cfg["dataset"]
    training_cfg  = cfg["training"]
    early_stopping_cfg = cfg["early_stopping"]

    ckpt_dir = Path(path_cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── DataLoader ──────────────────────────────────────────────────────── #
    loader = get_train_loader(
        train_json=path_cfg["train_json"],
        image_dir=path_cfg["image_dir"],
        objects_json=path_cfg["objects_json"],
        batch_size=training_cfg["batch_size"],
        image_size=data_cfg["image_size"],
        drop_prob=data_cfg["drop_prob"],
        augment=data_cfg["augment"],
        num_workers=data_cfg["num_workers"],
    )

    # ── Model, EMA, DDPM ────────────────────────────────────────────────── #
    model    = UNet(**cfg["model"]).to(device)
    schedule = NoiseSchedule(**cfg["schedule"]).to(device)
    ddpm     = DDPM(model, schedule).to(device)
    ema      = EMA(
        model,
        decay=training_cfg["ema_decay"],
        warmup_steps=training_cfg.get("ema_warmup_steps", 0),
    )

    # ── Optimizer & LR schedule ─────────────────────────────────────────── #
    optimizer = AdamW(model.parameters(), lr=training_cfg["lr"])
    scheduler = build_lr_scheduler(optimizer, training_cfg["lr_warmup_steps"])

    # ── Mixed precision scaler ──────────────────────────────────────────── #
    use_amp = training_cfg["mixed_precision"] and device.type == "cuda"
    scaler  = torch.amp.GradScaler(device="cuda")

    # ── Early stopping ──────────────────────────────────────────────────── #
    early_stop = EarlyStopping(
        patience=early_stopping_cfg["patience"],
        min_delta=early_stopping_cfg["min_delta"],
        min_epochs=early_stopping_cfg["min_epochs"],
    )

    # ── Resume ──────────────────────────────────────────────────────────── #
    start_epoch = 0
    global_step = 0

    resume_path = resume_path or training_cfg.get("resume_ckpt", "")
    if resume_path:
        start_epoch, global_step, _ = load_checkpoint(
            resume_path, ddpm, ema, optimizer, scheduler, early_stop, device
        )

    # ── Training loop ───────────────────────────────────────────────────── #
    for epoch in range(start_epoch + 1, training_cfg["num_epochs"] + 1):
        ddpm.train()
        epoch_loss = 0.0
        torch.cuda.empty_cache()

        for batch_idx, (images, conditions) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            images     = images.to(device)
            conditions = conditions.to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = ddpm.training_loss(images, conditions)

            scaler.scale(loss).backward()

            if training_cfg["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), training_cfg["grad_clip"])

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % training_cfg["log_every"] == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch:04d} | step {global_step:07d} "
                    f"| loss {loss.item():.4f} | lr {lr:.2e}"
                )

        # ── End-of-epoch bookkeeping ─────────────────────────────────────── #
        epoch_loss /= len(loader)
        print(f"Epoch {epoch:04d} complete — avg loss: {epoch_loss:.4f}")

        should_stop = early_stop.step(epoch, epoch_loss)

        # Save best checkpoint whenever loss improves.
        if early_stop.improved:
            save_checkpoint(
                ckpt_dir / "best.pt",
                epoch, ddpm, ema, optimizer, scheduler, early_stop,
                epoch_loss, global_step,
            )
            print(f"  [best] New best loss: {early_stop.best_loss:.4f}")

        # Periodic checkpoint every ``save_every`` epochs.
        if epoch % training_cfg["save_every"] == 0:
            save_checkpoint(
                ckpt_dir / f"ckpt_epoch{epoch:04d}.pt",
                epoch, ddpm, ema, optimizer, scheduler, early_stop,
                epoch_loss, global_step,
            )

        if should_stop:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(no improvement for {early_stopping_cfg['patience']} epochs). "
                f"Best loss: {early_stop.best_loss:.4f}"
            )
            break

    print("Training complete.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    train(cfg, resume_path=args.resume)