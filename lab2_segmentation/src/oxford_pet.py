"""
Dataset handler for the Oxford-IIIT Pet Dataset.
Implements custom data loading, preprocessing, and strict binary mask conversion.
"""
import os 
import argparse 
import random
import torch 
import numpy as np 
from torch.utils.data import Dataset, DataLoader 
from PIL import Image 
import matplotlib.pyplot as plt 
from torchvision.transforms import v2 as transforms
from typing import Tuple, Optional 

class OxfordPetDataset(Dataset):
    """
    Custom Dataset class for the Oxford-IIIT Pet Dataset.
    Handles loading, preprocessing, and strict binary mask conversion.
    """
    def __init__(
            self, 
            data_dir: str, 
            split: str = 'train', 
            image_size: int = 572,
            # is_train: bool = True, 
            # transform: Optional[transforms.Compose] = None,
            # target_transform: Optional[callable] = None
        ) -> None:
        """Initializes the dataset by reading a split file and validating paths.

        Args:
            data_dir (str): Root directory of the Oxford-IIIT Pet dataset,
                containing ``images/`` and ``annotations/trimaps/``
                subdirectories, plus ``<split>.txt`` files.
            split (str): Name of the split file **without** the ``.txt``
                extension (e.g. ``'train'``, ``'val'``, ``'test_unet'``,
                ``'test_res_unet'``).
            image_size (int): Target spatial size; images and masks are
                resized to ``(image_size, image_size)``.
            is_train (bool): If ``True``, apply random data augmentation
                (horizontal flip, vertical flip, small rotation, colour
                jitter). If ``False``, only deterministic resize is applied.

        Raises:
            FileNotFoundError: If the split file or any referenced image
                file does not exist on disk.
        """
        super().__init__() 
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.mask_size = 388 if image_size == 572 else image_size  # UNet output size after 4 downsamplings
        self.is_train = split == "train"  # Only apply augmentation to training split

        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "annotations", "trimaps")

        # Image base transform (without random flips)
        if self.is_train:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                # transforms.RandomApply(
                #     [transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.3)],
                #     p=0.5
                # ),
                transforms.RandomPhotometricDistort(p=.2),
                transforms.RandomPosterize(bits=2, p=0.2),
                transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.2),
                # transforms.RandomAutocontrast(p=.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))],
                    p=0.1,
                ),
                # transforms.RandomApply(
                #     [transforms.GaussianNoise(mean=0.0, sigma=0.1)],
                #     p=0.5
                # ),
                transforms.ToImage(), # Replaces ToTensor()
                transforms.ToDtype(torch.float32, scale=True), # Scales 0-255 to 0.0-1.0
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToImage(), # Replaces ToTensor()
                transforms.ToDtype(torch.float32, scale=True), # Scales 0-255 to 0.0-1.0
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # Mask base transform (resize with NEAREST)
        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.NEAREST
            ), 
            # transforms.CenterCrop(self.mask_size), 
            transforms.CenterCrop(388) if self.image_size == 572 else transforms.Lambda(lambda x: x),  # Crop to 388x388 to match UNet output size (after 4 downsamplings)
            transforms.ToImage(),  # Convert to image after resizing
            transforms.ToDtype(torch.float32, scale=False)  # Do not divide by 255
        ])

        # --- Load split file ---
        split_file = os.path.join(data_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                f"Ensure it exists inside '{data_dir}'."
            )
        
        self.image_filenames = [] 
        with open(split_file, 'r') as f:
            for line in f:
                img_name = line.strip() + ".jpg"  # Assuming images are .jpg
                if img_name:
                    self.image_filenames.append(img_name)
        # with open(split_file, 'r') as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         img_name = line.strip() + ".jpg"  # Assuming images are .jpg
        #         self.image_filenames.append(img_name)

        # Sort to ensure consistent ordering 
        self.image_filenames.sort() 
        # --- Validate image files exist ---
        self._validate_files() 

    def _validate_files(self) -> None:
        """Checks that all image files listed in the split file exist on disk.

        Raises:
            FileNotFoundError: If any image file is missing.
        """
        missing_images = [] 
        missing_masks = []

        for img_name in self.image_filenames:
            img_path = os.path.join(self.images_dir, img_name)
            if not os.path.exists(img_path):
                missing_images.append(img_path)
            # Check corresponding mask exists (for train/val splits)
            mask_path = os.path.join(self.masks_dir, img_name.replace(".jpg", ".png"))
            if self.split in ['train', 'val'] and not os.path.exists(mask_path):
                missing_masks.append(mask_path)

        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} image(s) listed in "
                f"'{self.split}.txt' were not found. "
                f"First 5: {missing_images[:5]}"
            )

        if missing_masks:
            print(
                f"[WARNING] {len(missing_masks)} mask(s) not found — they "
                f"will default to all-background.  First 5: {missing_masks[:5]}"
            )

    def __len__(self) -> int: 
        """
        Returns the total number of samples in the dataset. 
        
        Returns:
            int: Number of samples. 
        """
        return len(self.image_filenames) 
    
    def _process_mask(self, mask_img:Image.Image) -> Image.Image:
        """
        Converts the raw trimap to a binary mask according the lab specifications. 
        Tripmap values: 1 (Foreground), 2 (Background), 3 (Boundary). 
        Output values: 1 (Foreground), 0 (Background & Boundary). 

        Args:
            mask_img (Image.Image): The raw trimap image. 

        Returns:
            Image.Image: The processed binary mask image.
        """
        mask_np = np.array(mask_img) 

        # --- Initialise binary mask with zeros (background) ---
        binary_mask = np.zeros_like(mask_np, dtype=np.uint8) 
        binary_mask[mask_np == 1] = 1  # Set foreground pixels to 1
        # Add channel dimension to match (C, H, W) format 
        # binary_mask = np.expand_dims(binary_mask, axis=0)
        return Image.fromarray(binary_mask, mode='L')
    
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

        # --- Load images and their corresponding masks ---
        image = Image.open(img_path).convert("RGB")

        # Load mask if it exists, (test sets might not have masks available). 
        if self.split not in ["test_unet", "test_res_unet"]:
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing mask for training/val image: {mask_path}")
            raw_mask = Image.open(mask_path) 
            mask = self._process_mask(raw_mask)
        else:
            # Fallback for inference on raw test sets without ground truth masks. 
            mask = Image.new("L", image.size, 0)  # Create an empty mask (all background)

        # --- Apply transforms ---
        if self.is_train:
            if random.random() > 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            # if random.random() > 0.5:
            #     angle = random.uniform(-15, 15)
            #     image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
            #     mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)
        # Apply base transforms (resize, colour jitter, etc.) to the image
        image = self.img_transform(image) 
        mask = self.mask_transform(mask)

        return {
            'image': image,
            'mask': mask,
            'filename': img_name
        }
    
def visualize_sample(
        image_tensors: torch.Tensor,
        mask_tensors: torch.Tensor,
        num_images: int = 1
) -> None: 
    """
    Visualizes a single image and its corresponding binary mask side-by-side.
    
    Args:
        image_tensors (torch.Tensor): The image tensor of shape (C, H, W).
        mask_tensors (torch.Tensor): The mask tensors of shape (1, H, W).
    """
    img_nps = []
    mask_nps = []
    for i in range(num_images):
        image_tensor = image_tensors[i]
        mask_tensor = mask_tensors[i]
        img_np = image_tensor.permute(1, 2, 0).numpy()  # Convert to (H, W, C) for visualization
        img_np = np.clip(img_np, 0, 1)  # Ensure pixel values are in [0, 1]
        mask_np = mask_tensor.squeeze(0).numpy()  # Convert to (H, W)
        img_nps.append(img_np)
        mask_nps.append(mask_np)
    fig, axes = plt.subplots(2, num_images, figsize=(10,5)) 
    for i in range(num_images):
        axes[0, i].imshow(img_nps[i]) 
        axes[0, i].set_title("Input Image")
        axes[0, i].axis('off')

        axes[1, i].imshow(mask_nps[i], cmap='viridis')
    for i in range(num_images):
        axes[1, i].set_title("Binary Mask")
        axes[1, i].axis('off')

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

    # # Define Transform 
    # img_transform = transforms.Compose([
    #     transforms.Resize((args.img_size, args.img_size)),
    #     transforms.ToTensor(),
    # ])
    # mask_transform = transforms.Resize(
    #     (args.img_size, args.img_size), 
    #     interpolation=transforms.InterpolationMode.NEAREST
    # )

    dataset = OxfordPetDataset(
        data_dir=args.data_dir,
        split=args.split,
        image_size=args.img_size,
        # is_train=True,  # Enable data augmentation
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=dataset.is_train,  # Shuffle only if training
        pin_memory=True,
    )
    print(f"Dataset Size: {len(dataset)} samples")
    # Fetch a single batch and visualize the first sample
    # Iterate through the dataloader to get a batch of images and masks
    for batch in dataloader:
        images = batch['image']
        masks = batch['mask']
        print(f"Batch Image Shape: {images.shape}, Batch Mask Shape: {masks.shape}")
        # Verify binary constraints
        unique_vals = torch.unique(masks)
        print(f"Unique values in mask tensor: {unique_vals.tolist()}")
        if not all(v in [0.0, 1.0] for v in unique_vals.tolist()):
            print("WARNING: Mask contains invalid values! Your mapping logic is compromised.")
        else:
            print("SUCCESS: Mask is strictly binary.")

        print("Displaying first sample from batch...")
        visualize_sample(images, masks, num_images=images.shape[0])
        break