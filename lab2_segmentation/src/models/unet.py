import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop

class DoubleConv(nn.Module):
    """
    Standard UNet Double Convolution block: (Conv2d -> BatchNorm -> ReLU) * 2.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
    """
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=0, bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=0, bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DoubleConv block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channel, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channel, H, W).
        """
        return self.net(x)

class UNet(nn.Module):
    """
    Vanilla UNet architecture for binary semantic segmentation.
    
    Implements unpadded convolutions and dynamic center-cropping for skip connections.

    Args:
        in_channels (int): Number of channels in the input image (default: 3 for RGB).
        out_channels (int): Number of output channels (default: 1 for binary segmentation).
        base_c (int): Number of filters in the first convolutional layer.
    """
    def __init__(
            self, 
            in_channels: int = 3, 
            out_channels: int = 1, 
            base_c: int = 64
    ) -> None:
        super().__init__()

        # --- Encoder (Contracting Path) ---
        self.enc1 = DoubleConv(in_channels, base_c)
        self.enc2 = DoubleConv(base_c, base_c * 2)
        self.enc3 = DoubleConv(base_c * 2, base_c * 4)
        self.enc4 = DoubleConv(base_c * 4, base_c * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(base_c * 8, base_c * 16)

        # --- Expansive Path (Decoder) ---
        self.up4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_c * 16, base_c * 8)
        
        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_c * 8, base_c * 4)
        
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_c * 4, base_c * 2)
        
        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_c * 2, base_c)

        # Final output mapping (No sigmoid here; use BCEWithLogitsLoss during training)
        self.final = nn.Conv2d(base_c, out_channels, kernel_size=1)

        # --- Weight Initialisation ---
        self._init_weights()

    def _init_weights(self) -> None:
        """Applies Kaiming (He) normal initialization to all layers.

        * ``Conv2d`` / ``ConvTranspose2d``: Kaiming-normal (fan_out, ReLU).
        * ``BatchNorm2d``: weight = 1, bias = 0.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    @staticmethod
    def _crop_and_concat(
            upsampled: torch.Tensor, 
            bypass: torch.Tensor
    ) -> torch.Tensor:
        """
        Center-crops the bypass (encoder) tensor to match the spatial dimensions of 
        the upsampled (decoder) tensor, then concatenates them.

        Args:
            upsampled (torch.Tensor): The tensor from the expansive path (smaller spatial size).
            bypass (torch.Tensor): The tensor from the contracting path (larger spatial size).

        Returns:
            torch.Tensor: The concatenated tensor of shape (B, C1+C2, H, W).
        """
        # Crop the bypass tensor to match the shape of the upsampled tensor
        _, _, h, w = upsampled.shape
        diff_h = bypass.shape[2] - h
        diff_w = bypass.shape[3] - w
        if diff_h != 0 or diff_w != 0:
            cropped_bypass = center_crop(bypass, output_size=[h, w])
        else:
            cropped_bypass = bypass
        return torch.cat([cropped_bypass, upsampled], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the UNet.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, in_channels, H, W). H and W must be divisible by 16.

        Returns:
            torch.Tensor: Unnormalized logit predictions of shape (B, out_channels, H, W).
        """
        # --- Encoder ---
        e1 = self.enc1(x)                       # (B, 64 , H  ,  W  )
        e2 = self.enc2(self.pool(e1))           # (B, 128, H/2,  W/2)
        e3 = self.enc3(self.pool(e2))           # (B, 256, H/4,  W/4)
        e4 = self.enc4(self.pool(e3))           # (B, 512, H/8,  W/8)

        # --- Bottleneck ---
        bn = self.bottleneck(self.pool(e4))     # (B, 1024, H/16, W/16)

        # --- Decoder with skip connections ---
        d4 = self.up4(bn)
        d4 = self._crop_and_concat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._crop_and_concat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._crop_and_concat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._crop_and_concat(d1, e1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out