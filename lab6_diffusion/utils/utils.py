import os 
import json 
from typing import List, Dict
from pathlib import Path 
import torch 
import torchvision.utils as vutils
from PIL import Image


#---------------------------------------------------------------------------
# Dataset utilities: label map loading and encoding
#---------------------------------------------------------------------------
def load_label_map(objects_json: str | os.PathLike) -> Dict[str, int]:
    """Load the object-name-to-index mapping from objects.json.

    Args:
        objects_json: Path to objects.json.

    Returns:
        Dict mapping object name strings to integer indices (0-23).
    """
    with open(objects_json, "r") as f:
        return json.load(f)
    
def encode_labels(
        labels: List[str], 
        label_map: Dict[str, int], 
        num_classes: int = 24
) -> torch.Tensor:
    """Convert a list of object-name strings to a multi-hot float32 tensor. 

    Args:
        labels (List[str]): List of object name strings, e.g. ["cyan cube", "red sphere"].
        label_map (Dict[str, int]): Mapping from object names to indices.
        num_classes (int, optional): Number of classes. Defaults to 24.

    Returns:
        torch.Tensor: Multi-hot float32 tensor of shape (num_classes,).
    """
    vec = torch.zeros(num_classes, dtype=torch.float32) 
    for name in labels:
        vec[label_map[name]] = 1.0 
    return vec 

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(
        path: Path,
        epoch: int,
        model: torch.nn.Module,
        ema,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        early_stop,
        loss: float,
        global_step: int,
) -> None:
    """Save full training state to disk.

    Persists the raw model weights, EMA shadow weights, optimizer state,
    LR scheduler state, and early-stopping state so that training can be
    resumed exactly from this point.

    Args:
        path (Path): Destination file path (e.g. ``ckpts/ckpt_epoch0010.pt``).
        epoch (int): Current epoch (1-indexed).
        model (torch.nn.Module): Model whose state is saved.
        ema: EMA wrapper (shadow weights + step counter).
        optimizer (torch.optim.Optimizer): Optimizer whose state is saved.
        scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler whose state is saved.
        early_stop: Early-stopping tracker (best loss + counter).
        loss (float): Epoch-averaged loss to record in the checkpoint.
        global_step (int): Total gradient steps taken so far.
    """
    torch.save(
        {
            "epoch":         epoch,
            "global_step":   global_step,
            "loss":          loss,
            "model":         model.state_dict(),
            "ema":           ema.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "early_stopping": early_stop.state_dict(),
        },
        path,
    )
    print(f"  [ckpt] Saved → {path}")

def load_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    ema,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    early_stop,
    device: torch.device,
) -> tuple[int, int, float]:
    """Restore full training state from a checkpoint.

    Args:
        path (str | os.PathLike): Path to the ``.pt`` checkpoint file.
        model (torch.nn.Module): Model to restore raw weights into.
        ema: EMA wrapper to restore shadow weights and step counter into.
        optimizer (torch.optim.Optimizer): Optimizer to restore state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler to restore state into.
        early_stop: Early-stopping tracker to restore into.
        device (torch.device): Device to map tensors to.

    Returns:
        tuple[int, int, float]: ``(epoch, global_step, loss)`` from the checkpoint.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    # Early-stopping state is optional (supports old checkpoints without it).
    if "early_stopping" in ckpt:
        early_stop.load_state_dict(ckpt["early_stopping"])
    else:
        # Fallback: restore just best_loss so patience is tracked correctly.
        early_stop.best_loss = ckpt.get("loss", float("inf"))

    print(f"  [ckpt] Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt["global_step"], ckpt["loss"]

# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------
def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Convert a single CHW tensor in ``[-1, 1]`` to a PIL RGB image.

    Args:
        x (torch.Tensor): Float tensor of shape ``(3, H, W)`` in ``[-1, 1]``.

    Returns:
        Image.Image: Corresponding PIL RGB image in ``[0, 255]``.
    """
    x = x.clamp(-1.0, 1.0)
    x = (x + 1.0) / 2.0              # → [0, 1]
    x = (x * 255.0).byte()           # → [0, 255]
    return Image.fromarray(x.permute(1, 2, 0).cpu().numpy(), mode="RGB")

def save_image_grid(
        images: torch.Tensor,
        path: Path,
        nrow: int = 8,
) -> None:
    """Save a batch of ``[-1, 1]`` tensors as a PNG image grid.

    Args:
        images (torch.Tensor): Float tensor of shape ``(N, 3, H, W)`` in ``[-1, 1]``.
        path (Path): Destination PNG file path.
        nrow (int): Number of images per row in the grid. Defaults to 8.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = vutils.make_grid(
        images.clamp(-1.0, 1.0),
        nrow=nrow,
        normalize=True,    # maps [-1, 1] → [0, 1] for vutils
        value_range=(-1, 1),
    )
    vutils.save_image(grid, path)
    print(f"  [save] Grid → {path}")

def save_individual_images(images: torch.Tensor, out_dir: Path) -> None:
    """Save each image in a batch as a numbered PNG file.

    Images are saved as ``{out_dir}/0.png``, ``{out_dir}/1.png``, … matching
    the index order in the corresponding JSON test file.

    Args:
        images (torch.Tensor): Float tensor of shape ``(N, 3, H, W)`` in ``[-1, 1]``.
        out_dir (Path): Directory to save images into.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        pil = tensor_to_pil(img)
        pil.save(out_dir / f"{i}.png")
    print(f"  [save] {len(images)} individual PNGs → {out_dir}/")