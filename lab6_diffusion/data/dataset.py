"""
Dataset classes for the iCLEVR conditional diffusion model.

Splits
------
- Train  : dict {image_filename -> [object_labels]}  (18 009 samples)
- Test   : list of [object_labels]                   (32 samples, no images)

Label representation
--------------------
Each condition is a multi-hot float32 tensor of shape (24,).
Index i is 1.0 when object i is present, 0.0 otherwise.
The zero vector (all zeros) serves as the null/unconditional token for CFG.
"""
import json 
import os 
from pathlib import Path 
from typing import Dict, List, Tuple

import torch 
from PIL import Image 
from torch.utils.data import DataLoader, Dataset 
from torchvision.transforms import v2 as T 
from utils.utils import load_label_map, encode_labels

def build_transform(image_size: int = 64, augment: bool = False) -> T.Compose:
    """Build the standard image transform pipeline.

    Resizes to image_size x image_size, converts to tensor, and normalises
    pixel values from [0, 1] to [-1, 1] (required by the DDPM and evaluator).

    Args:
        image_size: Target spatial resolution (default 64).
        augment: If True, apply random horizontal flip for data augmentation (default False).

    Returns:
        A composed torchvision transform.
    """
    steps = [
        T.Resize(
            (image_size, image_size), 
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        ),
    ]
    if augment:
        steps.append(T.RandomHorizontalFlip())
    steps.extend([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),            # [0, 1]
        T.Normalize(
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5]),   # [-1, 1]
    ])
    return T.Compose(steps)

# ---------------------------------------------------------------------------
# Training dataset  (image + condition)
# ---------------------------------------------------------------------------
class ICLEVRTrainDataset(Dataset):
    """iCLEVR training dataset: returns (image, multi-hot label) pairs.

    Supports classifier-free guidance via random condition dropping: with
    probability `drop_prob` the returned label tensor is replaced by the
    zero vector (the null / unconditional token).

    Args:
        train_json:   Path to train.json.
        image_dir:    Directory that contains the .png image files.
        objects_json: Path to objects.json.
        image_size:   Spatial resolution to resize images to (default 64).
        drop_prob:    Probability of dropping the condition for CFG training.Set to 0.0 to disable (default 0.1).
    """
    def __init__(
            self, 
            train_json: str | os.PathLike, 
            image_dir: str | os.PathLike,
            objects_json: str | os.PathLike,
            image_size: int = 64,
            drop_prob: float = 0.1, 
            augment: bool = True,
    ) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.label_map = load_label_map(objects_json)
        self.transform = build_transform(image_size, augment=augment)
        self.drop_prob = drop_prob 
        self.num_classes = len(self.label_map)

        with open(train_json, "r") as f:
            raw: Dict[str, List[str]] = json.load(f)

        # Store as a list of (filename, labels) for index-based access 
        self.samples = list(raw.items()) 

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one (image, condition) pair.

        Args:
            idx: Sample index.

        Returns:
            image:     Float32 tensor of shape (3, H, W) in [-1, 1].
            condition: Float32 multi-hot tensor of shape (num_classes,).Replaced by zeros with probability `drop_prob`.
        """
        filename, labels = self.samples[idx] 
        image = Image.open(self.image_dir / filename).convert("RGB")
        image = self.transform(image)
        condition = encode_labels(labels, self.label_map, self.num_classes)

        # CFG: Randomly replace condition with the null (zero) vector. 
        if self.drop_prob > 0.0 and torch.rand(1).item() < self.drop_prob:
            condition = torch.zeros_like(condition)

        return image, condition 
    
# ---------------------------------------------------------------------------
# Inference dataset  (condition only — no images)
# ---------------------------------------------------------------------------
class ICLEVRTestDataset(Dataset):
    """iCLEVR test dataset: returns multi-hot label tensors only.

    Used at inference time to provide conditions for image generation.
    There are no ground-truth images in the test splits.

    Args:
        test_json:    Path to test.json or new_test.json.
        objects_json: Path to objects.json.
    """
    def __init__(
            self, 
            test_json: str | os.PathLike, 
            objects_json: str | os.PathLike,
    ) -> None:
        super().__init__()
        self.label_map = load_label_map(objects_json)
        self.num_classes = len(self.label_map)

        with open(test_json, "r") as f:
            self.samples: List[List[str]] = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return the multi-hot condition tensor for sample idx.

        Args:
            idx: Sample index.

        Returns:
            Float32 multi-hot tensor of shape (num_classes,).
        """
        return encode_labels(self.samples[idx], self.label_map, self.num_classes)
    
# ---------------------------------------------------------------------------
# DataLoader 
# ---------------------------------------------------------------------------
def get_train_loader(
        train_json: str | os.PathLike,
        image_dir: str | os.PathLike,
        objects_json: str | os.PathLike,
        batch_size: int = 64,
        image_size: int = 64,
        drop_prob: float = 0.1,
        augment: bool = True,
        num_workers: int = 4,
) -> DataLoader:
    """Build a DataLoader for the training split. 

    Args:
        train_json (str | os.PathLike): Path to train.json
        image_dir (str | os.PathLike): Directory containing iCLEVR .png images. 
        objects_json (str | os.PathLike): Path to objects.json. 
        batch_size (int, optional): Mini-batch size. Defaults to 64.
        image_size (int, optional): Image size. Defaults to 64.
        drop_prob (float, optional): CFG condition drop probability. Defaults to 0.1.
        num_workers (int, optional): Number of worker processes. Defaults to 4.

    Returns:
        DataLoader: A shuffled, pinned-memory DataLoader fot the training split. 
    """
    dataset = ICLEVRTrainDataset(
        train_json=train_json,
        image_dir=image_dir,
        objects_json=objects_json,
        image_size=image_size,
        drop_prob=drop_prob,
        augment=augment,
    )
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
    )

def get_test_loader(
        test_json: str | os.PathLike,
        objects_json: str | os.PathLike,
        batch_size: int = 32,
) -> DataLoader:
    """Build a DataLoader for the test split. 

    Args:
        test_json (str | os.PathLike): Path to test.json or new_test.json.
        objects_json (str | os.PathLike): Path to objects.json.
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        DataLoader: An ordered (no shuffle) DataLoader over the test conditions.
    """
    dataset = ICLEVRTestDataset(
        test_json=test_json,
        objects_json=objects_json,
    )
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True,
)