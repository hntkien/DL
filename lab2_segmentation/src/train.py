import os 
import argparse 
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from typing import Tuple 

from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import calculate_hard_dice_score, BCEWithDiceLoss, EarlyStopping

# ========== Argument Parsing ==========
def parse_args() -> argparse.Namespace:
    """Parses command-line arguments to ensure single-command execution without manual intervention.
    
    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train Binary Semantic Segmentation Models")
    parser.add_argument("--data_dir", type=str, default="./dataset/oxford-iiit-pet/", help="Path to dataset root")
    parser.add_argument("--model", type=str, choices=['unet', 'resnet34_unet'], required=True, help="Model architecture to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (Keep low for 8GB VRAM)")
    parser.add_argument("--image_size", type=int, default=256, help="Input image size (images will be resized to this)")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs")
    parser.add_argument("--save_dir", type=str, default="./saved_models/", help="Directory to save .pth checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to a .pth checkpoint to resume training")
    return parser.parse_args()

# ========== Function to Train for One Epoch ==========
def train_epoch(
        model: nn.Module, 
        dataloader: DataLoader, 
        criterion: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        device: torch.device, 
        scaler: torch.GradScaler
) -> float:
    """
    Executes one complete training epoch using Automatic Mixed Precision (AMP).

    Args:
        model (nn.Module): The segmentation model to train.
        dataloader (DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer (e.g., AdamW).
        criterion (nn.Module): The loss function (e.g., BCEWithDiceLoss).
        scaler (torch.GradScaler): PyTorch AMP GradScaler to prevent gradient underflow.
        device (torch.device): The device to compute on ('cuda').

    Returns:
        float: The average training loss for this epoch.
    """
    model.train()
    running_loss = 0.0

    # Wrap dataloader in tqdm to monitor training progress 
    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch in pbar:
        # Extract data from our custom dictionay format 
        images = batch["image"].to(device, non_blocking=True)  # (B, C, H, W)
        targets = batch["mask"].to(device, non_blocking=True)  # (B, 1, H, W)

        # Training pipeline 
        optimizer.zero_grad(set_to_none=True)  # More efficient zeroing of gradients
        with torch.autocast("cuda"):
            logits = model(images)  # Forward pass: (B, 1, H, W)
            loss = criterion(logits, targets)  # Compute loss
        scaler.scale(loss).backward()  # Backpropagate with scaled loss
        scaler.step(optimizer)  # Update weights
        scaler.update()  # Update the scale for next iteration
        running_loss += loss.item()  # Accumulate total loss
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})  # Update progress bar with current loss
    
    # Calculate and return average loss for the epoch
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

# ========== Function to Validate for One Epoch ==========
def validate_epoch(
        model: nn.Module, 
        dataloader: DataLoader, 
        criterion: nn.Module, 
        device: torch.device
) -> Tuple[float, float]:
    """
    Executes one complete validation epoch without gradient tracking.
    
    Args:
        model (nn.Module): The segmentation model to evaluate.
        dataloader (DataLoader): The validation data loader.
        criterion (nn.Module): The loss function to compute validation loss.
        device (torch.device): The device to compute on ('cuda').

    Returns:
        Tuple[float, float]: A tuple containing the average validation loss and the average validation Dice score for the epoch.
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    pbar = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():  # Disable gradient computation for validation
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)

            with torch.autocast("cuda"):
                logits = model(images) 
                loss = criterion(logits, targets)
            
            dice_score = calculate_hard_dice_score(logits, targets)  # Compute Dice score for this batch

            running_loss += loss.item()  # Accumulate total loss
            running_dice += dice_score  # Accumulate total Dice score
            pbar.set_postfix({
                "Val Loss": f"{loss.item():.4f}", 
                "Val Dice": f"{dice_score:.4f}"
            })  # Update progress bar
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)

    return epoch_loss, epoch_dice

# ========== Main Training Loop ==========
def main() -> None:
    """The main execution for the training pipeline. """
    args = parse_args()  # Parse command-line arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    # --- Dataset and DataLoader ---
    train_dataset = OxfordPetDataset(
        data_dir=args.data_dir, 
        split="train",
        image_size=args.image_size
    )
    val_dataset = OxfordPetDataset(
        data_dir=args.data_dir, 
        split="val",
        image_size=args.image_size
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=train_dataset.is_train, 
        num_workers=1, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # --- Model Selection ---
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1, base_c=64).to(device)
    elif args.model == 'resnet34_unet':
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError("Invalid model selected.")

    # --- Loss Function and Optimizer ---
    criterion = BCEWithDiceLoss(bce_weight=0.2, dice_weight=0.8).to(device) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scaler = torch.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=1e-6
    )

    # --- Early Stopping ---
    save_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
    early_stopping = EarlyStopping(
        save_path=save_path, 
        patience=args.patience, 
        mode='max', 
    )

    start_epoch = 0

    # --- Resume from Checkpoint if Provided ---
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            early_stopping.best_score = checkpoint['best_metric']
            early_stopping.val_metric_best = checkpoint['best_metric']
            print(f"Resumed from epoch {checkpoint['epoch']} with best metric {checkpoint['best_metric']:.4f}")
        else:
            raise FileNotFoundError(f"Resume checkpoint not found at '{args.resume}'")
    # --- Training Loop ---
    print(f"Starting training for {args.model.upper()} from epoch {start_epoch} to {args.epochs}...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 20)
        
        # Train and Validate
        torch.cuda.empty_cache()  # Clear GPU memory before each epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device)
        
        # Step the scheduler after validation to adjust learning rate based on epoch count
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice Score: {val_dice:.4f}")
        
        # Trigger Early Stopping check
        early_stopping(
            val_metric=val_dice, 
            model=model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            epoch=epoch
        )
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}. Training halted.")
            break

    print(f"Training complete. Best Validation Dice Score: {early_stopping.val_metric_best:.4f}")

if __name__ == "__main__":
    main()