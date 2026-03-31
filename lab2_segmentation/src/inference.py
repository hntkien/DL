import os
import argparse
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Ensure these match your actual file structure exactly
from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import rle_encode

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for single-command inference execution.
    """
    parser = argparse.ArgumentParser(description="Generate Kaggle Submission for Binary Semantic Segmentation")
    parser.add_argument("--data_dir", type=str, default="./dataset/oxford-iiit-pet/", help="Path to dataset root")
    parser.add_argument("--model", type=str, choices=['unet', 'resnet34_unet'], required=True, help="Model architecture")
    parser.add_argument("--image_size", type=int, default=256, help="Input image size (images will be resized to this)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .pth model weights")
    # parser.add_argument("--split", type=str, required=True, help="Test split file (e.g., 'test_unet' or 'test_res_unet')")
    parser.add_argument("--out_csv", type=str, default="submission.csv", help="Output Kaggle submission CSV file")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Inference on device: {device}")

    # 1. Initialize Dataset and DataLoader
    split = "test_unet" if args.model == "unet" else "test_res_unet"
    test_dataset = OxfordPetDataset(
        data_dir=args.data_dir, 
        split=split, 
        image_size=args.image_size
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # 2. Initialize and Load Model
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1, base_c=64).to(device)
    elif args.model == 'resnet34_unet':
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError("Invalid model selected.")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
    
    print(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. Lock Model for Inference
    model.eval()
    
    results = []
    
    print(f"Running inference on {len(test_dataset)} images...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating Predictions"):
            images = batch['image'].to(device, non_blocking=True)
            filenames = batch['filename']
            
            # Forward pass
            logits = model(images)
            # Threshold logits to create binary mask: shape (B, 1, 256, 256)
            preds_binary = (logits > 0.0).float()

            # Iterate through the batch to resize masks back to their original image dimensions
            for i in range(len(filenames)):
                img_name = filenames[i]
                image_id = os.path.splitext(img_name)[0]
                
                # Fetch original dimensions from the disk
                orig_img_path = os.path.join(args.data_dir, "images", img_name)
                with Image.open(orig_img_path) as orig_img:
                    orig_w, orig_h = orig_img.size
                
                # Extract the single prediction tensor: shape (1, 1, 256, 256)
                pred_tensor = preds_binary[i].unsqueeze(0)

                if args.model == 'unet':
                    # Calculate padding: (572 - 388) // 2 = 92
                    diff = args.image_size - pred_tensor.shape[-1]
                    pad_size = diff // 2
                    # Pad (left, right, top, bottom)
                    pred_tensor = F.pad(pred_tensor, (pad_size, pad_size, pad_size, pad_size), value=0)
                
                # Upsample back to original size using NEAREST interpolation to keep it strictly binary
                pred_resized = F.interpolate(
                    pred_tensor, 
                    size=(orig_h, orig_w), 
                    mode='nearest', 
                    # align_corners=False
                )
                
                # Squeeze to (H, W) and convert to numpy for RLE encoding
                pred_np = pred_resized.squeeze().cpu().numpy()
                
                # Apply Fortran-order RLE from your utils
                rle_string = rle_encode(pred_np)
                
                # Append to results
                results.append({"image_id": image_id, "encoded_mask": rle_string})

    # 4. Write to CSV
    print(f"Writing Kaggle submission to {args.out_csv}...")
    with open(args.out_csv, mode='w', newline='') as csv_file:
        fieldnames = ['image_id', 'encoded_mask']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print("Inference complete. Submission file is ready.")

if __name__ == "__main__":
    main()