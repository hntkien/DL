"""
Inference script for the conditional DDPM on iCLEVR.

Generates images for both test splits, saves individual PNGs, saves image
grids, produces a denoising-process grid, and reports evaluator accuracy.

Usage:
    # Minimal — uses paths from the config:
    python inference.py --config configs/config.yaml --checkpoint ckpts/best.pt

    # Override output dir and guidance scale:
    python inference.py --config configs/config.yaml \\
                        --checkpoint ckpts/best.pt  \\
                        --output_dir outputs          \\
                        --guidance_scale 2.0

    # Point to a custom evaluator checkpoint (default: ./checkpoint.pth):
    python inference.py --config configs/config.yaml \\
                        --checkpoint ckpts/best.pt   \\
                        --eval_ckpt  ckpts/checkpoint.pth

Expected outputs
----------------
outputs/
├── test/
│   ├── 0.png … N.png          # individual generated images
│   └── grid.png               # full image grid (for the report)
├── new_test/
│   ├── 0.png … M.png
│   └── grid.png
└── denoising_process.png      # timestep strip for ["red sphere", "cyan cylinder", "cyan cube"]
"""
import argparse 
import os 
import shutil 
import sys 
from pathlib import Path
from typing import List, Tuple 

import torch 

from data.dataset import get_test_loader 
from diffusion.noise_schedule import NoiseSchedule
from models.ddpm import DDPM
from utils.config import load_config
from models.unet import UNet
from utils.ema import EMA
from utils.utils import (
    tensor_to_pil,
    save_image_grid,
    save_individual_images,
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Inference with a trained DDPM on iCLEVR.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the DDPM checkpoint to load (e.g. ckpts/best.pt)."
    )
    parser.add_argument(
        "--eval_ckpt", 
        type=str, 
        default="ckpts/checkpoint.pth", 
        help="Path to the evaluator checkpoint (default: ckpts/checkpoint.pth)."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs", 
        help="Directory to save generated images and grids (default: outputs/)."
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        default=1.0, 
        help="Classifier-free guidance scale (default 1.0, i.e. no guidance)."
    )
    parser.add_argument(
        "--batch_size",
        type=int, 
        default=0,
        help="Inference batch size. 0 = use cfg['sampling']['batch_size']. Default: 0.")
    parser.add_argument(
        "--denoise_steps",
        type=int, 
        default=10,
        help="Number of intermediate frames shown in the denoising grid. Default: 10.")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_ddpm_from_checkpoint(
        ckpt_path: str | os.PathLike,
        cfg: dict,
        device: torch.device,
) -> torch.nn.Module:
    """Build a DDPM instance and populate its UNet with the EMA shadow weights.

    The training checkpoint stores both raw model weights (``"model"``) and
    EMA shadow weights (``"ema"``). For inference we always use the EMA weights
    because they produce smoother, higher-quality samples.

    Args:
        ckpt_path (str | os.PathLike): Path to the ``.pt`` checkpoint file.
        cfg (dict): Parsed config dictionary (``cfg["model"]``, ``cfg["schedule"]``).
        device (torch.device): Target device.

    Returns:
        DDPM: DDPM instance with EMA weights loaded, set to eval mode.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    model    = UNet(**cfg["model"]).to(device)
    schedule = NoiseSchedule(**cfg["schedule"]).to(device)
    ddpm     = DDPM(model, schedule).to(device)

    # Prefer EMA shadow weights; fall back to raw model weights if missing.
    if "ema" in ckpt and "shadow" in ckpt["ema"]:
        print("  [ckpt] Loading EMA shadow weights.")
        ddpm.model.load_state_dict(ckpt["ema"]["shadow"])
    elif "model" in ckpt:
        print("  [ckpt] EMA weights not found — loading raw model weights.")
        ddpm.load_state_dict(ckpt["model"])
    else:
        raise KeyError(f"Checkpoint at {ckpt_path} contains neither 'ema' nor 'model' keys.")

    ddpm.eval()
    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("loss",  float("nan"))
    print(f"  [ckpt] Loaded checkpoint from epoch {epoch} (loss={loss:.4f})")
    return ddpm

# ---------------------------------------------------------------------------
# Evaluator setup
# ---------------------------------------------------------------------------
def _ensure_eval_ckpt(eval_ckpt_arg: str) -> bool:
    """Ensure the evaluator checkpoint is accessible at ``./checkpoint.pth``.

    The provided ``evaluator.py`` hardcodes ``torch.load('./checkpoint.pth')``.
    If the user's checkpoint lives elsewhere, we create a temporary symlink so
    the evaluator can find it without modification.

    Args:
        eval_ckpt_arg (str): User-supplied path to the evaluator checkpoint.

    Returns:
        bool: ``True`` if the evaluator checkpoint is ready to use.
    """
    target = Path("./checkpoint.pth")
    src    = Path(eval_ckpt_arg).resolve()

    if not src.exists():
        print(
            f"  [eval] WARNING: evaluator checkpoint not found at {src}. "
            f"Skipping evaluation.", file=sys.stderr)
        return False

    if target.resolve() == src:
        return True  # Already in the right place.

    if target.exists() or target.is_symlink():
        target.unlink()

    try:
        target.symlink_to(src)
        print(f"  [eval] Symlinked {src} → ./checkpoint.pth")
    except OSError:
        # Symlink failed (e.g. Windows without dev mode); copy instead.
        shutil.copy2(src, target)
        print(f"  [eval] Copied {src} → ./checkpoint.pth")

    return True

def load_evaluator(eval_ckpt_arg: str):
    """Load the provided ResNet18 evaluator.

    Args:
        eval_ckpt_arg (str): Path to ``checkpoint.pth`` for the evaluator.
        device (torch.device): Device to run the evaluator on.

    Returns:
        evaluation_model | None: Evaluator instance, or ``None`` if unavailable.
    """
    if not _ensure_eval_ckpt(eval_ckpt_arg):
        return None
    try:
        from evaluation.evaluator import evaluation_model
        evaluator = evaluation_model()
        print("  [eval] Evaluator loaded successfully.")
        return evaluator
    except Exception as exc:
        print(f"  [eval] WARNING: could not load evaluator — {exc}", file=sys.stderr)
        return None
    
# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_images(
        ddpm: DDPM,
        conditions: torch.Tensor,
        batch_size: int,
        image_size: int,
        guidance_scale: float,
        device: torch.device,
) -> torch.Tensor:
    """Run DDPM sampling in mini-batches and return all generated images.

    Splits the condition tensor into chunks of ``batch_size`` to avoid OOM
    on large test sets.

    Args:
        ddpm (DDPM): DDPM model (eval mode, EMA weights).
        conditions (torch.Tensor): Float multi-hot tensor of shape ``(N, num_classes)``.
        batch_size (int): Number of images to sample per forward pass.
        image_size (int): Spatial size of generated images.
        guidance_scale (float): CFG guidance scale ``w``.
        device (torch.device): Device to run sampling on.

    Returns:
        torch.Tensor: Float tensor of shape ``(N, 3, H, W)`` in ``[-1, 1]``.
    """
    all_images: List[torch.Tensor] = []
    N = conditions.shape[0]

    for start in range(0, N, batch_size):
        cond_batch = conditions[start : start + batch_size].to(device)
        imgs = ddpm.sample(
            condition=cond_batch,
            image_size=image_size,
            guidance_scale=guidance_scale,
        )
        all_images.append(imgs.cpu())
        print(f"    Generated {min(start + batch_size, N)}/{N} images …")

    return torch.cat(all_images, dim=0)

@torch.no_grad()
def generate_denoising_process(
        ddpm: DDPM,
        condition: torch.Tensor,
        image_size: int,
        guidance_scale: float,
        num_frames: int,
        device: torch.device,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Sample one image and return its intermediate denoising frames.

    Args:
        ddpm (DDPM): DDPM model (eval mode, EMA weights).
        condition (torch.Tensor): Multi-hot condition of shape ``(1, num_classes)``.
        image_size (int): Spatial size of the generated image.
        guidance_scale (float): CFG guidance scale ``w``.
        num_frames (int): Approximate number of intermediate frames to capture.
        device (torch.device): Device to run sampling on.

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor]]:
            - final image of shape ``(1, 3, H, W)``
            - list of ``num_frames`` intermediate tensors, each ``(1, 3, H, W)``, ordered from most noisy (t≈T) to clean (t=0).
    """
    T     = ddpm.num_timesteps
    every = max(1, T // num_frames)

    final, intermediates = ddpm.sample(
        condition=condition.to(device),
        image_size=image_size,
        guidance_scale=guidance_scale,
        return_immediates=True,
        intermedate_every=every,
    )
    return final.cpu(), [x.cpu() for x in intermediates]

# ---------------------------------------------------------------------------
# Core per-split pipeline
# ---------------------------------------------------------------------------
def run_split(
        split_name: str,
        test_json: str,
        objects_json: str,
        ddpm: DDPM,
        evaluator,
        out_dir: Path,
        batch_size: int,
        image_size: int,
        guidance_scale: float,
        grid_nrow: int,
        device: torch.device,
) -> float:
    """Generate, save, and evaluate images for one test split.

    Args:
        split_name (str): Human-readable name, e.g. ``"test"`` or ``"new_test"``.
        test_json (str): Path to the JSON file for this split.
        objects_json (str): Path to ``objects.json``.
        ddpm (DDPM): DDPM model (eval mode).
        evaluator: ``evaluation_model`` instance, or ``None`` to skip evaluation.
        out_dir (Path): Per-split output directory (e.g. ``outputs/test``).
        batch_size (int): Sampling batch size.
        image_size (int): Spatial size of generated images.
        guidance_scale (float): CFG guidance scale.
        grid_nrow (int): Images per row in the saved grid.
        device (torch.device): Target device.

    Returns:
        float: Evaluator accuracy in ``[0, 1]``, or ``-1.0`` if skipped.
    """
    print(f"\n{'='*60}")
    print(f"  Split: {split_name}  ({test_json})")
    print(f"{'='*60}")

    # ── Load conditions ─────────────────────────────────────────────────── #
    loader = get_test_loader(test_json, objects_json, batch_size=len(
        __import__("json").load(open(test_json))
    ))
    conditions = next(iter(loader))  # (N, num_classes) — full dataset in one batch
    print(f"  Conditions: {conditions.shape}")

    # ── Generate ─────────────────────────────────────────────────────────── #
    print("  Sampling …")
    images = generate_images(
        ddpm, conditions, batch_size, image_size, guidance_scale, device
    )  # (N, 3, H, W) in [-1, 1]

    # ── Save individual PNGs ─────────────────────────────────────────────── #
    save_individual_images(images, out_dir)

    # ── Save grid ────────────────────────────────────────────────────────── #
    save_image_grid(images, out_dir / "grid.png", nrow=grid_nrow)

    # ── Evaluate ──────────────────────────────────────────────────────────── #
    accuracy = -1.0
    if evaluator is not None:
        # Evaluator expects images in [-1, 1] (Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)))
        imgs_eval   = images.clamp(-1.0, 1.0).to(device)
        labels_eval = conditions.to(device)
        accuracy    = evaluator.eval(imgs_eval, labels_eval)
        print(f"\n  ✓ [{split_name}] Evaluator accuracy: {accuracy * 100:.2f}%")
    else:
        print(f"\n  [eval] Skipped (evaluator unavailable).")

    return accuracy

# ---------------------------------------------------------------------------
# Denoising process visualisation
# ---------------------------------------------------------------------------
def run_denoising_viz(
        ddpm: DDPM,
        objects_json: str,
        label_names: List[str],
        out_path: Path,
        image_size: int,
        guidance_scale: float,
        num_frames: int,
        device: torch.device,
) -> None:
    """Generate and save the denoising process strip for a fixed label set.

    The strip shows the reverse diffusion trajectory from pure Gaussian noise
    (leftmost) to the final clean image (rightmost), matching the lab requirement
    of showing the denoising process in a grid.

    Args:
        ddpm (DDPM): DDPM model (eval mode).
        objects_json (str): Path to ``objects.json`` for label encoding.
        label_names (List[str]): Object label strings, e.g.
            ``["red sphere", "cyan cylinder", "cyan cube"]``.
        out_path (Path): Destination PNG path.
        image_size (int): Spatial size.
        guidance_scale (float): CFG guidance scale.
        num_frames (int): Number of intermediate frames to show in the strip.
        device (torch.device): Target device.
    """
    from utils.utils import load_label_map, encode_labels

    print(f"\n{'='*60}")
    print(f"  Denoising process — labels: {label_names}")
    print(f"{'='*60}")

    label_map = load_label_map(objects_json)
    num_classes = len(label_map)
    cond = encode_labels(label_names, label_map, num_classes).unsqueeze(0)  # (1, C)

    _, intermediates = generate_denoising_process(
        ddpm, cond, image_size, guidance_scale, num_frames, device
    )

    # intermediates: list of (1, 3, H, W), noisy → clean
    frames = torch.cat(intermediates, dim=0)  # (num_frames, 3, H, W)
    save_image_grid(frames, out_path, nrow=len(intermediates))
    print(f"  [save] Denoising process ({len(intermediates)} frames) → {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point: parse args, load model, run inference on both test splits."""
    args = parse_args()
    cfg  = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Config aliases ───────────────────────────────────────────────────── #
    path_cfg  = cfg["paths"]
    dataset_cfg  = cfg["dataset"]
    sampling_cfg  = cfg["sampling"]

    output_dir     = Path(args.output_dir or path_cfg["output_dir"])
    guidance_scale = args.guidance_scale if args.guidance_scale > 0 else sampling_cfg["guidance_scale"]
    batch_size     = args.batch_size if args.batch_size > 0 else sampling_cfg["batch_size"]
    image_size     = dataset_cfg["image_size"]
    grid_nrow      = sampling_cfg.get("grid_nrow", 8)
    objects_json   = path_cfg["objects_json"]

    print(f"Output dir    : {output_dir}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Batch size    : {batch_size}")

    # ── Load DDPM ────────────────────────────────────────────────────────── #
    ddpm = load_ddpm_from_checkpoint(args.checkpoint, cfg, device)

    # ── Load evaluator ───────────────────────────────────────────────────── #
    evaluator = load_evaluator(args.eval_ckpt)

    # ── Run both test splits ──────────────────────────────────────────────── #
    results: dict[str, float] = {}

    for split_name, json_key in [("test", "test_json"), ("new_test", "new_test_json")]:
        json_path = path_cfg[json_key]
        acc = run_split(
            split_name=split_name,
            test_json=json_path,
            objects_json=objects_json,
            ddpm=ddpm,
            evaluator=evaluator,
            out_dir=output_dir / split_name,
            batch_size=batch_size,
            image_size=image_size,
            guidance_scale=guidance_scale,
            grid_nrow=grid_nrow,
            device=device,
        )
        results[split_name] = acc

    # ── Summary ──────────────────────────────────────────────────────────── #
    print(f"\n{'='*60}")
    print("  Results Summary")
    print(f"{'='*60}")
    for split, acc in results.items():
        acc_str = f"{acc * 100:.2f}%" if acc >= 0 else "N/A"
        print(f"  {split:>10}: {acc_str}")

    # ── Denoising process visualisation ──────────────────────────────────── #
    # Required by the lab: show the denoising process for this specific label set.
    denoising_labels = ["red sphere", "cyan cylinder", "cyan cube"]
    run_denoising_viz(
        ddpm=ddpm,
        objects_json=objects_json,
        label_names=denoising_labels,
        out_path=output_dir / "denoising_process.png",
        image_size=image_size,
        guidance_scale=guidance_scale,
        num_frames=args.denoise_steps,
        device=device,
    )

    print("\nInference complete.")


if __name__ == "__main__":
    main()