import torch 
import torch.nn as nn
import math
from models.blocks import linear, Swish

# ---------------------------------------------------------------------------
# Time (sinusoidal) embedding
# ---------------------------------------------------------------------------
class TimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the diffusion timestep.

    Maps a scalar timestep t ∈ {1, …, T} to a dense vector of dimension
    ``dim`` using sine and cosine functions at different frequencies,
    following Vaswani et al. (2017) and Ho et al. (2020).

    Args:
        dim: Output embedding dimension (must be even).
    """
    def __init__(self, dim: int) -> None:
        super().__init__() 
        self.dim = dim 
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed a batch of integer timesteps. 

        Args:
            t (torch.Tensor): Integer timestep tensor of shape (B,).

        Returns:
            torch.Tensor: Float32 emebdding tensor of shape (B, dim). 
        """
        sinusoid = torch.outer(t.float(), self.inv_freq)  # (B, dim/2)
        emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)  # (B, dim)
        return emb
    
# ---------------------------------------------------------------------------
# Condition projection
# ---------------------------------------------------------------------------
class ConditionEmbedding(nn.Module):
    """Projects a multi-hot label vector into the time-embedding space. 

    A two-layer MLP (with Swish activation) maps the binary label vecor to a dense vector of dimension ``time_dim``, matching the shape of the time embedding so the two can be element-wise added. 

    For classifier-free guidance, the null condition is represented by the all-zeros vector; when ``condition`` is all zeros, this module outputs a near-zero vector (because the first linear layer has zero-initialised bias), leaving the time signal intact. 

    Args:
        num_classes (int): Number of object classes (24 for iCLEVR). 
        time_dim (int): Output dimension; must match the dimension of the time embedding.
    """
    def __init__(
            self, 
            num_classes: int, 
            time_dim: int, 
            mode: str = "fan_avg",
            distribution: str = "uniform"
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            linear(
                in_channels=num_classes, 
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
    
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """Project a multi-hot condition vector to the embedding space. 

        Args:
            condition (torch.Tensor): Float32 multi-hot vector of shape (B, num_classes). Pass the zero vector for unconditional (CFG null) generation. 

        Returns:
            torch.Tensor: Float32 vector of shape (B, time_dim) representing the projected condition.
        """
        return self.proj(condition)
    
# ---------------------------------------------------------------------------
# Class context embedding — cross-attention path
# ---------------------------------------------------------------------------
class ClassContextEmbedding(nn.Module):
    """Multi-hot label → per-class token sequence for cross-attention conditioning.
 
    Each of the ``num_classes`` object classes owns a learnable embedding vector(``class_emb``). A shared two-layer MLP (Swish) refines the per-class vectors before being broadcast across the batch. The multi-hot ``condition`` then acts as a mask: each sample's token at class index ``i`` is zeroed out if class ``i`` is absent, so the resulting tensor encodes "which classes are present in this sample" as a fixed-length sequence.
 
    Reference:
        Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion
        Models" (CVPR 2022)
 
    Args:
        num_classes (int): Number of object classes (24 for iCLEVR).
        context_dim (int): Per-token embedding dimension; the dimension of the
            keys and values consumed by cross-attention.
        mode (str): Variance-scaling fan mode for linear init.
        distribution (str): Variance-scaling distribution for linear init.
    """
    def __init__(
            self,
            num_classes: int,
            context_dim: int,
            mode: str = "fan_avg",
            distribution: str = "uniform",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.context_dim = context_dim
        # One learnable token per class. Initialised small so the network
        # starts close to "no conditioning" (consistent with zero-init to_out).
        self.class_emb = nn.Parameter(torch.randn(num_classes, context_dim) * 0.02)
        # Lightweight per-class MLP (shared across classes). Mirrors the
        # additive ConditionEmbedding's two-linear-with-Swish capacity.
        self.proj = nn.Sequential(
            linear(
                in_channels=context_dim,
                out_channels=context_dim,
                mode=mode,
                distribution=distribution,
            ),
            Swish(),
            linear(
                in_channels=context_dim,
                out_channels=context_dim,
                mode=mode,
                distribution=distribution,
            ),
        )
 
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """Build the per-sample class-token sequence.
 
        Args:
            condition (torch.Tensor): Float32 multi-hot tensor of shape
                ``(B, num_classes)``; values in {0, 1}.
 
        Returns:
            torch.Tensor: Context tensor of shape ``(B, num_classes, context_dim)``.
                Rows corresponding to absent classes are exactly zero.
        """
        # (num_classes, context_dim) — same for every sample in the batch.
        tokens = self.proj(self.class_emb)
        # Broadcast to batch and mask absent classes via the multi-hot indicator.
        # (1, N, D) * (B, N, 1) → (B, N, D)
        return tokens.unsqueeze(0) * condition.unsqueeze(-1)