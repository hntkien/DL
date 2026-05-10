import torch 
import torch.nn as nn 
from models.blocks import (
    Downsample, 
    ResBlockWithAttention, 
    Swish, 
    Upsample, 
    conv2d, 
    linear
)
from models.condition import TimeEmbedding, ConditionEmbedding
    
# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """Conditional U-Net for 64 x 64 DDPM image generation. 

    Takes a noisy image, the diffusion timestep, and a multi-hot condiiton vector; returns the predicted noise of the same spatial shape. 

    The condition is merge with the time embedding via element-wise addition before the network, so every ResBlock is implicitly conditioned on both signals through its internal time projection. 

    Args:
        in_channels (int): Number of image channels (default 3 for RGB).
        channel (int): Base channel width; deeper stages use multiples of this values (default 128). 
        num_classes (int): Number of condition classes (default 24 for iCLEVR).
        attn_heads (int): Number of self-attention heads at attended resolutions (default 1). 
        use_affine_time (bool): Use affine GroupNorm (AdaGN) conditioning instead of additive conditioning in ResBlocks (default: False).
        dropout (float): Dropout probability in ResBlocks (default 0.0).
    """
    def __init__(
            self, 
            in_channels: int = 3, 
            channel: int = 128, 
            num_classes: int = 24, 
            attn_heads: int = 1,
            use_affine_time: bool = False,
            dropout: float = 0.0, 
            num_groups: int = 32,
            mode: str = "fan_avg",
            distribution: str = "uniform"
    ) -> None: 
        super().__init__() 
        time_dim = channel * 4  # Time embedding dimension (must match the output of ConditionEmbedding)
        self.num_classes = num_classes
        for ch_width in [channel, channel * 2, channel * 4]:
            if ch_width % num_groups != 0:
                raise ValueError(f"Channel width {ch_width} is not divisible by num_groups {num_groups}. Please choose a compatible combination.")
        # ===== Embedding Layers =====
        self.time_emb = nn.Sequential(
            TimeEmbedding(channel), 
            linear(
                in_channels=channel, 
                out_channels=time_dim, 
                mode=mode, 
                distribution=distribution
            ), 
            Swish(), 
            linear(
                in_channels=time_dim, 
                out_channels=time_dim, 
                mode=mode, 
                distribution=distribution
            )
        )
        self.cond_emb = ConditionEmbedding(
            num_classes=num_classes, 
            time_dim=time_dim, 
            mode=mode,
            distribution=distribution
        )

        # Shorthand for repeated kwargs
        def _rb(in_c, out_c, attn=False, num_groups=num_groups, mode=mode, distribution=distribution):
            return ResBlockWithAttention(
                in_channels=in_c, 
                out_channels=out_c, 
                time_dim=time_dim, 
                use_affine_time=use_affine_time, 
                use_attention=attn,
                dropout=dropout,
                attention_heads=attn_heads, 
                num_groups=num_groups,
                mode=mode,
                distribution=distribution 
            )
        
        # Channel aliases for readability (default: 128, 256, 512)
        C1, C2, C4 = channel, channel * 2, channel * 4
        
        # ===== Encoder =====
        # Stage 0 - 64x64, 128 channels 
        self.init_conv = conv2d(
            in_channels=in_channels, 
            out_channels=channel, 
            kernel_size=3, 
            padding=1, 
            mode=mode,
            distribution=distribution
        )
        self.down_0a = _rb(C1, C1) 
        self.down_0b = _rb(C1, C1)
        self.down_01 = Downsample(C1)  # 64 -> 32

        # Stage 1 - 32x32, 256 channels
        self.down_1a = _rb(C1, C2)
        self.down_1b = _rb(C2, C2)
        self.down_12 = Downsample(C2)  # 32 -> 16

        # Stage 2 - 16x16, 256 channels (+ attention) 
        self.down_2a = _rb(C2, C2, attn=True)
        self.down_2b = _rb(C2, C2, attn=True)
        self.down_23 = Downsample(C2)  # 16 -> 8

        # Stage 3 - 8x8, 512 channels (+ attention)
        self.down_3a = _rb(C2, C4, attn=True)
        self.down_3b = _rb(C4, C4, attn=True)

        # ===== Bottleneck =====
        self.mid_a = _rb(C4, C4, attn=True)
        self.mid_b = _rb(C4, C4)

        # ===== Decoder (mirror of encoder; input channels = current + skip) ======
        # C1, C2, C4 = channel, channel * 2, channel * 4   # 128, 256, 512

        # Stage 3 — 8×8  (+ attention)
        # x(C4=512) || f3b(C4=512) → C4   x(C4) || f3a(C4=512) → C4
        # x(C4)     || f23(C2=256) → C4   ← f23 is Downsample output: C2 ch
        self.up_3a = _rb(C4 + C4, C4, attn=True)   # 512+512 → 512
        self.up_3b = _rb(C4 + C4, C4, attn=True)   # 512+512 → 512
        self.up_3c = _rb(C4 + C2, C4, attn=True)   # 512+256 → 512  (f23 is C2)
        self.up_32 = Upsample(C4)                  # → 16×16

        # Stage 2 — 16×16  (+ attention)
        # x(C4=512) || f2b(C2=256) → C2   x(C2) || f2a(C2) → C2   x(C2) || f12(C2) → C2
        self.up_2a = _rb(C4 + C2, C2, attn=True)   # 512+256 → 256
        self.up_2b = _rb(C2 + C2, C2, attn=True)   # 256+256 → 256
        self.up_2c = _rb(C2 + C2, C2, attn=True)   # 256+256 → 256
        self.up_21 = Upsample(C2)                  # → 32×32

        # Stage 1 — 32×32
        # x(C2=256) || f1b(C2=256) → C1   x is now C1 for all remaining blocks
        # x(C1=128) || f1a(C2=256) → C1   ← x is C1 after up_1a; skip is still C2
        # x(C1=128) || f01(C1=128) → C1   ← f01 is Downsample output: C1 ch
        self.up_1a = _rb(C2 + C2, C1)              # 256+256 → 128
        self.up_1b = _rb(C1 + C2, C1)              # 128+256 → 128  (x→C1, skip→C2)
        self.up_1c = _rb(C1 + C1, C1)              # 128+128 → 128  (f01 is C1)
        self.up_10 = Upsample(C1)                  # → 64×64

        # Stage 0 — 64×64
        # x(C1=128) || f0b(C1=128) → C1  (all C1 throughout)
        self.up_0a = _rb(C1 + C1, C1)              # 128+128 → 128
        self.up_0b = _rb(C1 + C1, C1)              # 128+128 → 128
        self.up_0c = _rb(C1 + C1, C1)              # 128+128 → 128

        # ===== Output Projection =====
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, channel), 
            Swish(), 
            conv2d(
                in_channels=128, 
                out_channels=in_channels, 
                kernel_size=3, 
                padding=1, 
                scale=1e-10, 
                mode=mode,
                distribution=distribution
            )
        )

    def forward(
            self, 
            x: torch.Tensor, 
            time: torch.Tensor, 
            condition: torch.Tensor
    ) -> torch.Tensor:
        """Predict the noise added to ``x`` at timestep ``t``. 

        The time and condition embeddings are merged by element-wise addition before the network, producing a single ``emb`` vector that is passed to every ResBlock. 

        Args:
            x (torch.Tensor): _description_
            time (torch.Tensor): _description_
            condition (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        # ── Merged embedding ───────────────────────────────────────────── #
        emb = self.time_emb(time) + self.cond_emb(condition) # (B, time_dim)

        # ── Encoder ────────────────────────────────────────────────────── #
        f0 = self.init_conv(x)        # (B, 128, 64, 64)
        f0a = self.down_0a(f0, emb)   # (B, 128, 64, 64)
        f0b = self.down_0b(f0a, emb)  # (B, 128, 64, 64)
        f01 = self.down_01(f0b)       # (B, 128, 32, 32)

        f1a = self.down_1a(f01, emb)  # (B, 256, 32, 32)
        f1b = self.down_1b(f1a, emb)  # (B, 256, 32, 32)
        f12 = self.down_12(f1b)       # (B, 256, 16, 16)

        f2a = self.down_2a(f12, emb)  # (B, 256, 16, 16)
        f2b = self.down_2b(f2a, emb)  # (B, 256, 16, 16)
        f23 = self.down_23(f2b)       # (B, 256, 8, 8)

        f3a = self.down_3a(f23, emb)  # (B, 512, 8, 8)
        f3b = self.down_3b(f3a, emb)  # (B, 512, 8, 8)

        # ── Bottleneck ─────────────────────────────────────────────────── #
        x = self.mid_a(f3b, emb)       # (B, 512, 8, 8)
        x = self.mid_b(x, emb)         # (B, 512, 8, 8)

        # ── Decoder ────────────────────────────────────────────────────── #
        x = self.up_3a(torch.cat([x, f3b], dim=1), emb)  # (B, 1024, 8, 8) -> (B, 512, 8, 8)
        x = self.up_3b(torch.cat([x, f3a], dim=1), emb)  # (B, 1024, 8, 8) -> (B, 512, 8, 8)
        x = self.up_3c(torch.cat([x, f23], dim=1), emb)  # (B, 768, 8, 8) -> (B, 512, 8, 8)
        x = self.up_32(x)                                  # (B, 512, 8, 8) -> (B, 512, 16, 16)

        x = self.up_2a(torch.cat([x, f2b], dim=1), emb)  # (B, 512, 16, 16) -> (B, 256, 16, 16)
        x = self.up_2b(torch.cat([x, f2a], dim=1), emb)  # (B, 512, 16, 16) -> (B, 256, 16, 16)
        x = self.up_2c(torch.cat([x, f12], dim=1), emb)  # (B, 512, 16, 16) -> (B, 256, 16, 16)
        x = self.up_21(x)                                  # (B, 256, 16, 16) -> (B, 256, 32, 32)

        x = self.up_1a(torch.cat([x, f1b], dim=1), emb)  # (B, 512, 32, 32) -> (B, 128, 32, 32)
        x = self.up_1b(torch.cat([x, f1a], dim=1), emb)  # (B, 384, 32, 32) -> (B, 128, 32, 32)
        x = self.up_1c(torch.cat([x, f01], dim=1), emb)  # (B, 256, 32, 32) -> (B, 128, 32, 32)
        x = self.up_10(x)                                 # (B, 128, 32, 32) -> (B, 128, 64, 64)

        x = self.up_0a(torch.cat([x, f0b], dim=1), emb)  # (B, 256, 64, 64) -> (B, 128, 64, 64)
        x = self.up_0b(torch.cat([x, f0a], dim=1), emb)  # (B, 256, 64, 64) -> (B, 128, 64, 64)
        x = self.up_0c(torch.cat([x, f0], dim=1), emb)   # (B, 256, 64, 64) -> (B, 128, 64, 64)

        return self.out(x)  # (B, in_channels, 64, 64)

# ---------------------------------------------------------------------------
# Quick shape check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = UNet(
        in_channels=3, 
        channel=128, 
        num_classes=24, 
        attn_heads=1, 
        use_affine_time=False, 
        dropout=0.0,
        num_groups=32,
        mode="fan_avg",
        distribution="uniform"
    )
    B = 2
    img   = torch.randn(B, 3, 64, 64)
    t     = torch.tensor([100, 500])
    cond  = torch.zeros(B, 24)
    cond[0, [2, 9]] = 1.0   # blue cube + red sphere
    cond[1, [0]]    = 1.0   # gray cube

    out = model(img, t, cond)
    print(f"Input : {tuple(img.shape)}")
    print(f"Output: {tuple(out.shape)}")
    assert out.shape == img.shape, "Output shape mismatch!"
    print("Shape check passed.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.1f}M")