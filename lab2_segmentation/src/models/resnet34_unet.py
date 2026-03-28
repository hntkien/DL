"""
ResNet34 + UNet architecture for binary semantic segmentation.
Implements a ResNet34 encoder from scratch (no pre-trained weights) 
and a UNet decoder equipped with Convolutional Block Attention Modules (CBAM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel Attention module for CBAM.
    
    Args:
        in_planes (int): Number of input channels.
        reduction_ratio (int): Reduction ratio for the shared MLP.
    """
    def __init__(self, in_planes: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction_ratio, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Channel Attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Attention-scaled tensor of shape (B, C, H, W).
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Spatial Attention module for CBAM.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Spatial Attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Attention-scaled tensor of shape (B, C, H, W).
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module combining Channel and Spatial Attention.
    
    Args:
        in_planes (int): Number of input channels.
    """
    def __init__(self, in_planes: int) -> None:
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CBAM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Attention-scaled tensor of shape (B, C, H, W).
        """
        x = self.ca(x)
        x = self.sa(x)
        return x

class BasicBlock(nn.Module):
    """
    ResNet Basic Block.
    
    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        stride (int): Stride for the first convolution.
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for BasicBlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H_out, W_out).
        """
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class DecoderBlock(nn.Module):
    """
    UNet Decoder Block with CBAM.
    
    Args:
        in_channels (int): Number of input channels (from skip connection + previous decoder output).
        out_channels (int): Number of output channels.
        skip_channels (int): Number of channels from the skip connection.
    """
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int) -> None:
        super().__init__()
        # Project the upsampled input down to 32 channels as per the schematic
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Convolutions after concatenating (32 + skip_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(32+skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for DecoderBlock. Upsamples input and concatenates with skip connection.
        
        Args:
            x (torch.Tensor): Input tensor from previous decoder block.
            skip (torch.Tensor): Skip connection tensor from the encoder.
            
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H_out, W_out).
        """
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = self.project(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.cbam(x)
        return x

class ResNet34_UNet(nn.Module):
    """
    ResNet34 + UNet architecture for binary semantic segmentation.
    Built from scratch. No pre-trained weights allowed.
    
    Args:
        in_channels (int): Number of input channels (default: 3).
        out_channels (int): Number of output channels (default: 1).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()

        # --- ENCODER (ResNet34)  ---
        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        self.layer1 = self._make_layer(64, 64, 3, stride=1)   # Output: 64 channels
        self.layer2 = self._make_layer(64, 128, 4, stride=2)  # Output: 128 channels
        self.layer3 = self._make_layer(128, 256, 6, stride=2) # Output: 256 channels
        self.layer4 = self._make_layer(256, 512, 3, stride=2) # Output: 512 channels (Bottleneck)

        # --- DECODER (UNet style with CBAM)  ---
        # DecoderBlock(in_ch, skip_ch, out_ch)
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # Final mapping blocks to reach output channels [cite: 136, 137]
        self.final_conv = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
        
        self._initialize_weights()

    def _make_layer(
            self, 
            in_channels: int, 
            out_channels: int, 
            blocks: int, 
            stride: int
    ) -> nn.Sequential:
        """Helper to build ResNet layers."""
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ResNet34_UNet.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).
            
        Returns:
            torch.Tensor: Unnormalized logit predictions of shape (B, 1, H, W).
        """
        # Encoder Stage
        e0 = self.relu(self.bn1(self.conv1(x))) # Skip 1 (64 channels)
        x_pool = self.maxpool(e0)
        
        e1 = self.layer1(x_pool) # Skip 2 (64 channels)
        e2 = self.layer2(e1)     # Skip 3 (128 channels)
        e3 = self.layer3(e2)     # Skip 4 (256 channels)
        e4 = self.layer4(e3)     # Bottleneck (512 channels)

        # Decoder Stage
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)

        # Final projection to original size and target classes
        out = self.final_conv(d1)
        # Upsample back to original input resolution (e.g., from 128x128 back to 256x256)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

if __name__ == "__main__":
    # Sanity check for hardware constraints and tensor shapes
    import argparse
    parser = argparse.ArgumentParser(description="Test ResNet34_UNet Architecture")
    parser.add_argument("--batch_size", type=int, default=4, help="Test batch size")
    parser.add_argument("--img_size", type=int, default=256, help="Test image size")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    model = ResNet34_UNet(in_channels=3, out_channels=1).to(device)
    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        if output.shape[2:] != dummy_input.shape[2:]:
            print("WARNING: Input and output spatial dimensions do not match!")
        else:
            print("SUCCESS: Forward pass complete and shapes map correctly.")
    except Exception as e:
        print(f"FAILED: {e}")