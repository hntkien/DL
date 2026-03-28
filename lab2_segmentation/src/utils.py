"""
Utility functions and custom loss modules for binary semantic segmentation.
"""
import os 
from typing import Dict, Any 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Differentiable Dice Loss for training.
    
    This computes the continuous version of the Dice Coefficient. Instead of 
    hard 1s and 0s, it uses the raw probabilities (post-sigmoid) to allow 
    gradients to flow back through the network.
    
    Args:
        smooth (float): A small constant added to the numerator and denominator to prevent division by zero and smooth the gradients.
    """
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Dice Loss.
        
        Args:
            logits (torch.Tensor): Unnormalized network predictions of shape (B, 1, H, W).
            targets (torch.Tensor): Binary ground truth masks of shape (B, 1, H, W).
            
        Returns:
            torch.Tensor: The computed Dice loss (scalar).
        """
        # Squeeze the channel dimension if necessary and flatten spatial dimensions
        # We apply sigmoid because the network outputs raw logits
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions, preserve batch dimension\
        # Shape goes from (B, 1, H, W) to (B, H*W)
        probs_flat = probs.view(probs.size(0), -1) 
        targets_flat = targets.view(targets.size(0), -1)
        
        # Intersection: element-wise multiplication
        intersection = (probs_flat * targets_flat).sum(dim=1)  # Sum over spatial dimensions, keep batch dimension
        
        # Denominator: sum of sets
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        # Dice coefficient calculation
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Loss is 1 - Dice (we want to minimize the loss, which maximizes the Dice score)
        return 1.0 - dice_coeff.mean() # Average over batch

class BCEWithDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    
    This takes advantage of BCE's stable pixel-wise gradients and Dice's 
    ability to handle class imbalance and optimize directly for the evaluation metric.
    
    Args:
        bce_weight (float): Weight multiplier for the BCE loss component.
        dice_weight (float): Weight multiplier for the Dice loss component.
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted sum of BCE and Dice Loss.
        
        Args:
            logits (torch.Tensor): Unnormalized network predictions of shape (B, 1, H, W).
            targets (torch.Tensor): Binary ground truth masks of shape (B, 1, H, W).
            
        Returns:
            torch.Tensor: The combined scalar loss.
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        
        combined_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
        return combined_loss

def calculate_hard_dice_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates the exact, hard Dice Score for evaluation and testing.
    
    Args:
        preds (torch.Tensor): Network predictions (logits or probabilities) of shape (B, 1, H, W). 
        targets (torch.Tensor): Binary ground truth masks of shape (B, 1, H, W).
        
    Returns:
        float: The exact Dice score.
    """
    # If inputs are logits, we could apply sigmoid, but simply thresholding at 0.0 is equivalent
    # to thresholding probabilities at 0.5.
    preds_binary = (preds > 0.0).float()
    
    # Preserve batch dimension: (B, H*W)
    preds_flat = preds_binary.view(preds_binary.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    
    intersection = (preds_flat * targets_flat).sum(dim=1)  # Sum over spatial dimensions, keep batch dimension
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    
    # if union == 0:
    #     return 1.0 # Both prediction and target are entirely empty background
    
    # Handle the edge case where both prediction and target are entirely empty
    # If union is 0, Dice should be 1. Otherwise, standard Dice formula.
    dice = torch.where(
        union == 0, 
        torch.ones_like(union), 
        (2.0 * intersection) / union
    )
        
    return dice.mean().item() # Average over batch and return as Python float

def rle_encode(mask: np.ndarray) -> str:
    """
    Converts a binary mask numpy array to Run-Length Encoding (RLE) string 
    in column-major (Fortran) order for Kaggle submission.
    
    Args:
        mask (np.ndarray): 2D binary numpy array (H, W) where 1 is foreground.
        
    Returns:
        str: Space-separated RLE string (start length start length ...).
            Returns an empty string if the mask is entirely background.
    """
    # Flatten in column-major (Fortran) order 
    pixels = mask.flatten(order='F')

    # Pad the array with zeros at the beginning and end to cleanly identify the start and end of sequences of 1s 
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Indices where value changes
    runs[1::2] -= runs[::2]  # Convert to lengths

    return ' '.join(str(x) for x in runs)

class EarlyStopping:
    """
    Halts training when a monitored metric stops improving and handles checkpointing.
    """
    def __init__(
            self, 
            save_path: str, 
            patience: int = 7, 
            mode: str = 'max', 
            delta: float = 0.0
    ) -> None:
        """
        Args:
            save_path (str): The file path to save the .pth checkpoint.
            patience (int): How many epochs to wait after last time metric improved.
            mode (str): 'max' for metrics like Dice Score, 'min' for Loss.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.save_path = save_path
        self.patience = patience
        self.mode = mode
        self.delta = delta
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_best = -float('inf') if mode == 'max' else float('inf')

    def __call__(
            self, 
            val_metric: float, 
            model: torch.nn.Module, 
            optimizer: torch.optim.Optimizer, 
            scheduler: Any, 
            epoch: int
    ) -> None:
        """
        Evaluates the metric and triggers saving or stopping.
        
        Args:
            val_metric (float): The metric to evaluate (e.g., Validation Dice).
            model (torch.nn.Module): The neural network.
            optimizer (torch.optim.Optimizer): The optimizer state.
            scheduler (Any): The learning rate scheduler state.
            epoch (int): The current epoch number.
        """
        # Convert metric to a standardized score where higher is always better for the logic
        score = val_metric if self.mode == 'max' else -val_metric

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_metric, model, optimizer, scheduler, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_metric, model, optimizer, scheduler, epoch)
            self.counter = 0

    def _save_checkpoint(
            self, 
            val_metric: float, 
            model: torch.nn.Module, 
            optimizer: torch.optim.Optimizer, 
            scheduler: Any, 
            epoch: int
    ) -> None:
        """
        Saves the model and training state strictly as a .pth file.
        """
        self.val_metric_best = val_metric
        
        # We save the entire ecosystem to allow resuming
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_metric': val_metric
        }
        
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(state, self.save_path)


# if __name__ == "__main__":
#     # Sanity check
#     print("Testing Loss Functions...")
#     logits = torch.randn(4, 1, 256, 256)
#     targets = torch.randint(0, 2, (4, 1, 256, 256)).float()
    
#     criterion = BCEWithDiceLoss(bce_weight=0.5, dice_weight=0.5)
#     loss = criterion(logits, targets)
#     score = calculate_hard_dice_score(logits, targets)
    
#     print(f"Combined Loss: {loss.item():.4f}")
#     print(f"Hard Dice Score (random logits): {score:.4f}")