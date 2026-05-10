import torch 
import torch.nn as nn
import math
from blocks import linear, Swish

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