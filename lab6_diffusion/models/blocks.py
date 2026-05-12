import math 
from typing import Optional
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------
@torch.no_grad() 
def variance_scaling_init(
    tensor: torch.Tensor, 
    scale: float = 1.0, 
    mode: str = "fan_avg", 
    distribution: str = "uniform"
) -> torch.Tensor:
    """In-lace variance-scaling initisalisation (Glorot/He family)

    Args:
        tensor (torch.Tensor): Parameter tensor to intialise in-place. 
        scale (float, optional): Variance multiplier applied after fan normalisation. Defaults to 1.0.
        mode (str, optional): One of ``"fan_in"``, ``"fan_out"``, or ``"fan_avg"``. Defaults to "fan_avg".
        distribution (str, optional): One of ``"uniform"`` or ``"normal"``. Defaults to "uniform".

    Returns:
        torch.Tensor: The intialised tensor (same object, modified in-place).
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        scale /= fan_in
    elif mode == "fan_out":
        scale /= fan_out
    elif mode == "fan_avg":
        scale /= (fan_in + fan_out) / 2.0

    else:
        raise ValueError(f"Invalid mode {mode}, expected one of 'fan_in', 'fan_out', 'fan_avg'.")

    if distribution == "uniform":
        bound = math.sqrt(3 * scale)
        return tensor.uniform_(-bound, bound)
    elif distribution == "normal":
        std = math.sqrt(scale)
        return tensor.normal_(0, std)
    else:
        raise ValueError(f"Invalid distribution {distribution}, expected 'uniform' or 'normal'.")

def conv2d(
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 0,
        bias: bool = True,
        scale: float = 1.0, 
        mode: str = "fan_avg",
        distribution: str = "uniform"
) -> nn.Conv2d:
    """Variance-scaling-initialised 2-D convolution. 

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size (assumed square).
        stride (int, optional): Convolution stride. Defaults to 1.
        padding (int, optional): Convolution padding. Defaults to 0.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
        scale (float, optional): Variance multiplier. Defaults to 1.0.
        mode (str, optional): Fan mode for variance scaling. Defaults to "fan_avg".
        distribution (str, optional): Distribution for variance scaling. Defaults to "uniform".

    Returns:
        nn.Conv2d: Initialised convolutional layer.
    """
    conv = nn.Conv2d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        bias=bias
    )
    variance_scaling_init(
        conv.weight, 
        scale=scale, 
        mode=mode, 
        distribution=distribution
    )
    if bias:
        nn.init.zeros_(conv.bias)
    return conv

def linear(
        in_channels: int, 
        out_channels: int, 
        scale: float = 1.0, 
        mode: str = "fan_avg",
        distribution: str = "uniform"
) -> nn.Linear:
    """Variance-scaling-initialised linear layer. 

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale (float, optional): Variance multiplier. Defaults to 1.0.
        mode (str, optional): Fan mode for variance scaling. Defaults to "fan_avg".
        distribution (str, optional): Distribution for variance scaling. Defaults to "uniform".

    Returns:
        nn.Linear: Initialised linear layer.
    """
    lin = nn.Linear(
        in_features=in_channels, 
        out_features=out_channels, 
    )
    variance_scaling_init(
        lin.weight, 
        scale=scale, 
        mode=mode, 
        distribution=distribution
    )
    nn.init.zeros_(lin.bias)
    return lin

# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
class Swish(nn.Module):
    """Swish (SiLU) acticvation: x * sigmoid(x)

    Used throughout the U-Net following the original DDPM implementation. 
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Swish activation element-wise.

        Args:
            x (torch.Tensor): Input tensor. 

        Returns:
            torch.Tensor: Output tensor.
        """
        return F.silu(x) 
    
# ---------------------------------------------------------------------------
# Spatial up / downsampling
# ---------------------------------------------------------------------------
class Upsample(nn.Sequential):
    """2x nearest-neighbour upsample followed by a 3x3 convolution.

    The extra conv smooths the checkerboard artefacts that nearest-neighbour
    upsampling can introduce.

    Args:
        channel: Number of input (and output) channels.
    """
    def __init__(self, channel: int) -> None:
        super().__init__(
            nn.Upsample(scale_factor=2, mode="nearest"), 
            conv2d(channel, channel, kernel_size=3, padding=1)
        )

class Downsample(nn.Sequential):
    """Strided 3x3 convolution for 2x spatial downsampling.

    Preferred over pooling because it is learnable and preserves information
    better for diffusion model decoders.

    Args:
        channel: Number of input (and output) channels.
    """
    def __init__(self, channel: int) -> None:
        super().__init__(
            conv2d(channel, channel, kernel_size=3, stride=2, padding=1)
        )
    
# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Time-conditioned residual block (GroupNorm + Conv + time projection).

    Supports two conditioning modes:

    * **Additive** (``use_affine_time=False``): The time/condition embedding is
      projected to ``out_channel`` and *added* to the intermediate feature map before the second GroupNorm. Straightforward and used by Ho et al. (2020).

    * **Affine** (``use_affine_time=True``): The embedding is projected to ``2 x out_channel`` and split into scale (γ) and shift (β) parameters for an affine transformation of the second GroupNorm (AdaGN), similar to Dhariwal & Nichol (2021). More expressive but slightly heavier.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        time_dim (int): Dimension of the merged time+condition embedding.
        use_affine_time (bool): Use affine GroupNorm conditioning (default False).
        dropout (float): Dropout probability on the second conv (default 0).
        num_groups (int): Number of groups for GroupNorm (default 32). Should divide both in_channel and out_channel.
    """
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            time_dim: int, 
            use_affine_time: bool = False, 
            dropout: float = 0.0, 
            num_groups: int = 32, 
            mode: str = "fan_avg",
            distribution: str = "uniform"
    ) -> None: 
        super().__init__() 
        self.use_affine_time = use_affine_time

        if in_channels % num_groups != 0 or out_channels % num_groups != 0:
            raise ValueError(f"num_groups={num_groups} must divide both in_channels={in_channels} and out_channels={out_channels}.")
        
        time_out_dim = out_channels 
        time_scale = 1.0
        norm_affine = True 

        if use_affine_time:
            time_out_dim *= 2 # Split into gamma and beta 
            time_scale = 1e-10 # Near zero init -> identity at start 
            norm_affine = False # Don't learn affine params in GroupNorm if using AdaGN
        
        self.norm1 = nn.GroupNorm(
            num_groups=num_groups, 
            num_channels=in_channels
        )
        self.act1 = Swish() 
        self.conv1 = conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            padding=1, 
            mode=mode,
            distribution=distribution
        )

        self.time_proj = nn.Sequential(
            Swish(), 
            linear(
                in_channels=time_dim, 
                out_channels=time_out_dim, 
                scale=time_scale, 
                mode=mode,
                distribution=distribution
            )
        )

        self.norm2 = nn.GroupNorm(
            num_groups=num_groups, 
            num_channels=out_channels, 
            affine=norm_affine
        )
        self.act2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            padding=1, 
            scale=1e-10, 
            mode=mode,
            distribution=distribution
        )
        
        self.skip = (
            conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=1,
                mode=mode,
                distribution=distribution
            )
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with time + condition embedding injection.

        Args:
            x (torch.Tensor): Input feature map of shape (B, in_channel, H, W).
            emb (torch.Tensor): Merged time+condition embedding of shape (B, time_dim).

        Returns:
            torch.Tensor: Output feature map of shape (B, out_channel, H, W).
        """
        B = x.shape[0] 
        h = self.conv1(self.act1(self.norm1(x)))
        time_emb = self.time_proj(emb)  # (B, time_out_dim)
        
        if self.use_affine_time:
            gamma, beta = time_emb.view(B, -1, 1, 1).chunk(2, dim=1)  # (B, out_channel) each
            h = self.norm2(h) * (1 + gamma) + beta
        else:
            h = h + self.time_proj(emb).view(B, -1, 1, 1)  # (B, out_channel)
            h = self.norm2(h) 

        h = self.conv2(self.dropout(self.act2(h)))
        skip = self.skip(x) if self.skip is not None else x 
        return h + skip

# ---------------------------------------------------------------------------
# Self-attention block
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial feature maps (QKV via 1x1 conv).

    Flattens the spatial dimensions into a sequence, runs scaled dot-product
    attention, then reshapes back. Applied at lower resolutions (≤ 16x16) to
    keep the quadratic cost manageable.

    Args:
        in_channel (int): Number of input (and output) channels.
        n_head (int): Number of attention heads (default 1).
    """
    def __init__(
            self, 
            in_channels: int, 
            n_heads: int = 1,
            num_groups: int = 32,
            mode: str = "fan_avg",
            distribution: str = "uniform"
    ) -> None:
        super().__init__() 
        self.n_heads = n_heads 
        if in_channels % n_heads != 0:
            raise ValueError(f"in_channels={in_channels} must be divisible by n_heads={n_heads} for multi-head attention.")
        self.norm = nn.GroupNorm(
            num_groups=num_groups, 
            num_channels=in_channels
        )
        self.qkv = conv2d(
            in_channels=in_channels, 
            out_channels=in_channels * 3, 
            kernel_size=1, 
            mode=mode,
            distribution=distribution
        )
        self.out = conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=1, 
            scale=1e-10, 
            mode=mode,
            distribution=distribution
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention with residual connection. 

        Args:
            x (torch.Tensor): Feature map of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Attended feature map of shape (B, in_channels, H, W).
        """
        B, C, H, W = x.shape 
        head_dim = C // self.n_heads

        norm = self.norm(x) 
        qkv = self.qkv(norm).view(B, self.n_heads, 3 * head_dim, H, W)  # (B, n_heads, 3*head_dim, H, W)
        q, k, v = qkv.chunk(3, dim=2)  # Each (B, n_heads, head_dim, H, W)

        # Scaled dot-product: (B, n_heads, H, W, H, W) 
        attn = torch.einsum("bnchw,bncyx->bnhwyx", q, k) / math.sqrt(head_dim)
        attn = attn.view(B, self.n_heads, H, W, -1)
        attn = torch.softmax(attn, dim=-1)  # (B, n_heads, H, W, H*W)
        attn = attn.view(B, self.n_heads, H, W, H, W)

        out = torch.einsum("bnhwyx,bncyx->bnchw", attn, v).contiguous()  # (B, n_heads, head_dim, H, W)
        out = self.out(out.view(B, C, H, W))  # (B, C, H, W)

        return x + out

# ---------------------------------------------------------------------------
# Cross-attention block
# ---------------------------------------------------------------------------
class CrossAttention(nn.Module):
    """Multi-head cross-attention from spatial features to a context sequence.

    Reference:
        Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion
        Models" (CVPR 2022), §3.3 (Conditioning Mechanisms).

    Args:
        in_channels (int): Channel count of the spatial feature map.
        context_dim (int): Per-token dimension of the context sequence.
        n_heads (int): Number of attention heads. ``in_channels`` must be
            divisible by ``n_heads``.
        num_groups (int): GroupNorm groups for the spatial pre-norm.
        mode (str): Variance-scaling fan mode for layer init.
        distribution (str): Variance-scaling distribution for layer init.
    """
    def __init__(
            self,
            in_channels: int,
            context_dim: int,
            n_heads: int = 1,
            num_groups: int = 32,
            mode: str = "fan_avg",
            distribution: str = "uniform",
    ) -> None:
        super().__init__()
        if in_channels % n_heads != 0:
            raise ValueError(
                f"in_channels={in_channels} must be divisible by n_heads={n_heads}."
            )
        self.n_heads = n_heads
        self.head_dim = in_channels // n_heads
        self.scale = self.head_dim ** -0.5

        # Pre-norms (transformer-style): GroupNorm on spatial Q-source,
        # LayerNorm on the context-sequence K/V source.
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.norm_ctx = nn.LayerNorm(context_dim)

        # Q: 1×1 conv keeps spatial layout (matches SelfAttention style).
        self.to_q = conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            mode=mode,
            distribution=distribution,
        )
        # K, V: linear over the token dimension.
        self.to_k = linear(
            in_channels=context_dim,
            out_channels=in_channels,
            mode=mode,
            distribution=distribution,
        )
        self.to_v = linear(
            in_channels=context_dim,
            out_channels=in_channels,
            mode=mode,
            distribution=distribution,
        )
        # Output projection — zero-init for an identity start.
        self.to_out = conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            scale=1e-10,
            mode=mode,
            distribution=distribution,
        )

    def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply cross-attention with a residual connection.

        Args:
            x (torch.Tensor): Spatial feature map of shape (B, in_channels, H, W).
            context (torch.Tensor): Context sequence of shape (B, N, context_dim).
                Tokens for absent classes should already be zeroed by the
                upstream embedding module.
            key_padding_mask (Optional[torch.Tensor]): Bool tensor of shape
                (B, N). ``True`` keeps a token, ``False`` masks it. Rows that
                are entirely False are unmasked (softmax fallback); this is
                safe because the corresponding V rows are zero.

        Returns:
            torch.Tensor: Output feature map of shape (B, in_channels, H, W).
        """
        B, C, H, W = x.shape
        N = context.shape[1]
        HW = H * W
        d = self.head_dim
        n = self.n_heads

        # Pre-norm.
        h = self.norm(x)
        ctx = self.norm_ctx(context)

        # Project. Q is (B, C, H, W); K, V are (B, N, C).
        q = self.to_q(h)
        k = self.to_k(ctx)
        v = self.to_v(ctx)

        # Reshape to multi-head: bring (heads, head_dim) to the front.
        # Q: (B, C, H, W) → (B, n, d, HW)
        q = q.view(B, n, d, HW)
        # K, V: (B, N, C) → (B, n, d, N)
        k = k.view(B, N, n, d).permute(0, 2, 3, 1).contiguous()
        v = v.view(B, N, n, d).permute(0, 2, 3, 1).contiguous()

        # Scores: (B, n, HW, N) = Qᵀ · K
        attn = torch.einsum("bndq,bndk->bnqk", q, k) * self.scale

        if key_padding_mask is not None:
            # If a sample has no active class (CFG null), every key is masked;
            # softmax over -inf would NaN. Fallback: leave that row unmasked.
            # The V projection of zero context is zero, so the attention
            # output is still zero — the desired behaviour.
            all_inactive = ~key_padding_mask.any(dim=1, keepdim=True)  # (B, 1)
            kpm = key_padding_mask | all_inactive                       # (B, N)
            mask = ~kpm[:, None, None, :]                               # (B, 1, 1, N)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)

        # Aggregate: (B, n, HW, N) × (B, n, d, N) → (B, n, d, HW)
        out = torch.einsum("bnqk,bndk->bndq", attn, v)
        # Back to (B, C, H, W)
        out = out.contiguous().view(B, C, H, W)
        out = self.to_out(out)

        return x + out

# ---------------------------------------------------------------------------
# Combined residual + (optional) attention block
# ---------------------------------------------------------------------------
class ResBlockWithAttention(nn.Module):
    """Residual block with optional self-attention and optional cross-attention.

    Block ordering (Latent-Diffusion / Stable-Diffusion pattern):

        x ─► ResBlock(time-conditioned) ─► [SelfAttention] ─► [CrossAttention(context)] ─► out

    Self-attention captures global spatial context within the feature map;
    cross-attention injects external conditioning (per-class context tokens).
    Both are skipped when their respective flags are off, so the block can
    represent the additive-only baseline (cross-attn off) and the
    cross-attention variant (cross-attn on at attention resolutions) with the
    same class.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        time_dim (int): Dimension of the time embedding consumed by the ResBlock.
        dropout (float): Dropout probability inside the ResBlock.
        use_attention (bool): If True, apply self-attention after the ResBlock.
        attention_heads (int): Number of self-attention heads.
        use_affine_time (bool): Use AdaGN-style affine modulation in the ResBlock.
        num_groups (int): GroupNorm groups.
        mode (str): Variance-scaling fan mode for layer init.
        distribution (str): Variance-scaling distribution for layer init.
        use_cross_attention (bool): If True, apply cross-attention after the
            (optional) self-attention. Requires ``context_dim`` to be set and
            ``context`` to be passed to ``forward``.
        context_dim (int | None): Per-token dimension of the context sequence
            consumed by cross-attention. Required when ``use_cross_attention``
            is True.
        cross_attention_heads (int): Number of cross-attention heads.
    """
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            time_dim: int, 
            dropout: float = 0.0, 
            use_attention: bool = False, 
            attention_heads: int = 1,
            use_affine_time: bool = False, 
            num_groups: int = 32,
            mode: str = "fan_avg",
            distribution: str = "uniform",
            use_cross_attention: bool = False,
            context_dim: Optional[int] = None,
            cross_attention_heads: int = 1,
    ) -> None: 
        super().__init__() 
        self.res = ResBlock(
            in_channels=in_channels, 
            out_channels=out_channels, 
            time_dim=time_dim, 
            use_affine_time=use_affine_time, 
            dropout=dropout, 
            num_groups=num_groups,
            mode=mode,
            distribution=distribution
        )
        if use_attention:
            self.attn = SelfAttention(
                in_channels=out_channels, 
                n_heads=attention_heads, 
                num_groups=num_groups,
                mode=mode,
                distribution=distribution
            )
        if use_cross_attention:
            if context_dim is None:
                raise ValueError(
                    "context_dim must be provided when use_cross_attention=True."
                )
            self.cross_attn = CrossAttention(
                in_channels=out_channels,
                context_dim=context_dim,
                n_heads=cross_attention_heads,
                num_groups=num_groups,
                mode=mode,
                distribution=distribution,
            )

    def forward(
            self,
            x: torch.Tensor,
            emb: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input feature map of shape (B, in_channels, H, W).
            emb (torch.Tensor): Time embedding of shape (B, time_dim). In the
                additive conditioning path this is the time+condition sum; in
                the cross-attention path it is the time embedding alone.
            context (Optional[torch.Tensor]): Context sequence of shape
                (B, N, context_dim) for cross-attention. Ignored when this
                block has no cross-attention layer.
            key_padding_mask (Optional[torch.Tensor]): Bool tensor of shape
                (B, N) marking which context tokens are present. Forwarded to
                cross-attention when applicable.

        Returns:
            torch.Tensor: Output feature map of shape (B, out_channels, H, W).
        """
        x = self.res(x, emb)
        if hasattr(self, "attn"):
            x = self.attn(x)
        if hasattr(self, "cross_attn"):
            if context is None:
                raise ValueError(
                    "ResBlockWithAttention has cross-attention enabled but "
                    "no context tensor was provided."
                )
            x = self.cross_attn(x, context, key_padding_mask=key_padding_mask)
        return x