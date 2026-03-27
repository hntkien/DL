"""
Dataset handler for the Oxford-IIIT Pet Dataset.
Implements custom data loading, preprocessing, and strict binary mask conversion.
"""
import os 
import argparse 
import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader 
from PIL import Image 
import matplotlib.pyplot as plt 
from torchvision import transforms 
from typing import Tuple, Dict, Optional 

class OxfordPetDataset(Dataset):
    """
    Custom Dataset class for the Oxford-IIIT Pet Dataset.
    Handles loading, preprocessing, and strict binary mask conversion.
    """
    def __init__(
            self, 
            data_dir: str, 
            split: str = 'train', 
            transform: Optional[transforms.Compose] = None,
            target_transform: Optional[callable] = None
        ) -> None:
        """
        Initializes the dataset.

        Args:
            data_dir (str): Root directory containing 'images' and 'annotations/trimaps' folders.
            split (str): Dataset split to load ('train', 'val', 'test').
            transform (callable, optional): Optional transform to be applied on a sample image.
            target_transform (callable, optional): Optional transform to be applied on the mask.
        """
        super().__init__() 
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "annotations", "trimaps")

        split_file = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file '{split_file}' not found. Ensure it exists and is correctly formatted.")
        
        self.image_filenames = [] 
        with open(split_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_name = line.strip() + ".jpg"  # Assuming images are .jpg
                self.image_filenames.append(img_name)

        # valid_extensions = ('.jpg', '.jpeg', '.png')
        # self.image_filenames = [
        #     f for f in os.listdir(self.images_dir)
        #     if f.lower().endswith(valid_extensions) and not f.startswith('._')
        # ]
        # Sort to ensure consistent ordering 
        self.image_filenames.sort() 

    def __len__(self) -> int: 
        """
        Returns the total number of samples in the dataset. 
        
        Returns:
            int: Number of samples. 
        """
        return len(self.image_filenames) 
    
    def _process_mask(self, mask_img:Image.Image) -> torch.Tensor:
        """
        Converts the raw trimap to a binary mask according the lab specifications. 
        Tripmap values: 1 (Foreground), 2 (Background), 3 (Boundary). 
        Output values: 1 (Foreground), 0 (Background & Boundary). 

        Args:
            mask_img (Image.Image): The raw trimap image. 

        Returns:
            torch.Tensor: The processed binary mask tensor of shape (1, H, W).
        """
        mask_np = np.array(mask_img) 

        ## Initialise binary mask with zeros (background) 
        binary_mask = np.zeros_like(mask_np, dtype=np.float32) 
        binary_mask[mask_np == 1] = 1.0  # Set foreground pixels to 1
        # Add channel dimension to match (C, H, W) format 
        binary_mask = np.expand_dims(binary_mask, axis=0)
        return torch.from_numpy(binary_mask)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches the image and its corresponding binary mask. 

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The image tensor and its corresponding binary mask tensor.
        """
        img_name = self.image_filenames[idx] 
        img_path = os.path.join(self.images_dir, img_name) 

        # Masks share the same base name but have a .png extension 
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.masks_dir, f"{base_name}.png")

        # Load image 
        image = Image.open(img_path).convert("RGB")

        # Load mask if it exists, (test sets might not have masks available). 
        if os.path.exists(mask_path):
            mask = Image.open(mask_path) 
        else:
            # Fallback for inference on raw test sets without ground truth masks. 
            mask = Image.new("L", image.size, 0)  # Create an empty mask (all background)

        if self.transform:
            image = self.transform(image) 

        # We process the mask manually, but target_transform can handle resizing 
        if self.target_transform:
            mask = self.target_transform(mask)

        mask_tensor = self._process_mask(mask)
        return image, mask_tensor
    
def visualize_sample(
        image_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
) -> None: 
    """
    Visualizes a single image and its corresponding binary mask side-by-side.
    
    Args:
        image_tensor (torch.Tensor): The image tensor of shape (C, H, W).
        mask_tensor (torch.Tensor): The mask tensor of shape (1, H, W).
    """
    img_np = image_tensor.permute(1, 2, 0).numpy()  # Convert to (H, W, C) for visualization
    img_np = np.clip(img_np, 0, 1)  # Ensure pixel values are in [0, 1]
    mask_np = mask_tensor.squeeze(0).numpy()  # Convert to (H, W)

    fig, axes = plt.subplots(1, 2, figsize=(10,5)) 
    axes[0].imshow(img_np) 
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(mask_np, cmap='viridis')
    axes[1].set_title("Binary Mask")
    axes[1].axis('off')

    plt.tight_layout() 
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Oxford-IIIT Pet Dataset Loader")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory (should contain 'images' and 'annotations')")
    parser.add_argument("--split", type=str, default='train', choices=['train', 'val', 'test_unet', 'test_res_unet'], help="Dataset split to load")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for the dataloader")
    parser.add_argument("--img_size", type=int, default=256, help="Image resize dimension")
    
    args = parser.parse_args()

    print(f"Initialising Dataset from: {args.data_dir}")
    print(f"Target Resolution: {args.img_size}x{args.img_size}")

    # Define Transform 
    img_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Resize(
        (args.img_size, args.img_size), 
        interpolation=transforms.InterpolationMode.NEAREST
    )

    dataset = OxfordPetDataset(
        data_dir=args.data_dir,
        split=args.split,
        transform=img_transform,
        target_transform=mask_transform
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset Size: {len(dataset)} samples")
    # Fetch a single batch and visualize the first sample
    images, masks = next(iter(dataloader))
    print(f"Batch Image Shape: {images.shape}, Batch Mask Shape: {masks.shape}")
    # Verify binary constraints
    unique_vals = torch.unique(masks)
    print(f"Unique values in mask tensor: {unique_vals.tolist()}")
    if not all(v in [0.0, 1.0] for v in unique_vals.tolist()):
        print("WARNING: Mask contains invalid values! Your mapping logic is compromised.")
    else:
        print("SUCCESS: Mask is strictly binary.")

    print("Displaying first sample from batch...")
    visualize_sample(images[0], masks[0])