import math
from typing import Literal 
import torch 
import torch.nn as nn

ScheduleType = Literal["linear", "cosine"]

def linear_beta_schedule(
        beta_start: float, 
        beta_end: float, 
        num_timesteps: int, 
) -> torch.Tensor:
    """Linearly-spaced Beta scheduler. 

    Ramps Beta uniformly from ``beta_start`` to ``beta_end`` over ``num_timesteps`` steps. The original paper uses beta_1 = 1e-4 and beta_T = 0.02 for T = 1,000. 

    Args:
        beta_start (float): Starting value for Beta.
        beta_end (float): Ending value for Beta.
        num_timesteps (int): Number of timesteps.

    Returns:
        torch.Tensor: Linearly-spaced Beta values.
    """
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)

def cosine_beta_schedule(
        num_timesteps: int, 
        s: float = 0.008, 
) -> torch.Tensor:
    """Cosine Beta scheduler. 

    Ramps Beta according to a cosine function, as proposed in Nichol & Dhariwal (2021). The original paper uses s = 0.008 and T = 1,000. 

    Args:
        num_timesteps (int): Number of timesteps.
        s (float, optional): Small offset to prevent singularities. Defaults to 0.008.

    Returns:
        torch.Tensor: Cosine-scheduled Beta values.
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
    f = torch.cos(((t / num_timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999)

class NoiseSchedule(nn.Module):
    """Precompute noise schedule for DDPM trianing and inference. 

    Registers all derived quantities a sbuffers so they are automatically moved to the correct device with ``.to(device)`` and excluded from gradient computation. 
    """
    def __init__(
            self, 
            schedule: ScheduleType = "linear",
            num_timesteps: int = 1000, 
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            cosine_s: float = 0.008,
    ) -> None: 
        super().__init__() 
        self.num_timesteps = num_timesteps
        self.schedule = schedule

        if schedule == "linear":
            betas = linear_beta_schedule(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps, s=cosine_s)
        else:
            raise ValueError(f"Unsupported schedule type: {schedule}")
        
        betas = betas.float() 
        alphas = 1.0 - betas 
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # ===== Register buffers =====
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", 
            torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "betas_over_sqrt_one_minus_alphas_cumprod", 
            betas / torch.sqrt(1.0 - alphas_cumprod)
        )
        # Posterior variance 
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped", 
            torch.log(torch.clip(posterior_variance, min=1e-20))
        )
    
    # ---------------------------------------------------------------------- #
    # Forward process                                                        #
    # ---------------------------------------------------------------------- #
    def q_sample(
            self, 
            x0: torch.Tensor, 
            t: torch.Tensor, 
            noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Diffuse an image ``x0`` to timestep ``t`` by adding noise. 

        Args:
            x0 (torch.Tensor): Original image tensor of shape (B, C, H, W).
            t (torch.Tensor): Timestep tensor of shape (B,) with integer values in [0, num_timesteps-1].
            noise (torch.Tensor, optional): Noise tensor of shape (B, C, H, W). If None, sampled from standard normal distribution. Defaults to None.

        Returns:
            torch.Tensor: Noisy image tensor of shape (B, C, H, W).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise
    # ---------------------------------------------------------------------- #
    # Reverse process                                                        #
    # ---------------------------------------------------------------------- #
    def predict_x0_from_noise(
            self, 
            xt: torch.Tensor, 
            t: torch.Tensor, 
            noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """Predict the original image ``x0`` from a noisy image ``xt`` at timestep ``t`` and the noise added. 

        Args:
            xt (torch.Tensor): Noisy image tensor of shape (B, C, H, W).
            t (torch.Tensor): Timestep tensor of shape (B,) with integer values in [0, num_timesteps-1].
            noise_pred (torch.Tensor): Predicted noise tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Predicted original image tensor of shape (B, C, H, W).
        """
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, xt.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        return (xt - sqrt_1mab * noise_pred) / sqrt_ab
    
    def q_posterior_mean(
            self, 
            xt: torch.Tensor, 
            t: torch.Tensor,
            noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the mean of the posterior distribution q(xt-1 | xt, x0) using the predicted noise. 

        Args:
            xt (torch.Tensor): Noisy image tensor at timestep t of shape (B, C, H, W).
            t (torch.Tensor): Timestep tensor of shape (B,) with integer values in [0, num_timesteps-1].
            noise_pred (torch.Tensor): Predicted noise tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Posterior mean tensor of shape (B, C, H, W).
        """
        recip_sqrt_a = self._extract(self.sqrt_recip_alphas, t, xt.shape)
        beta_coef = self._extract(
            self.betas_over_sqrt_one_minus_alphas_cumprod, 
            t, 
            xt.shape
        )
        return recip_sqrt_a * (xt - beta_coef * noise_pred)
    
    @staticmethod
    def _extract(
            buffer: torch.Tensor, 
            t: torch.Tensor, 
            shape: torch.Size
    ) -> torch.Tensor:
        """"Gather per-timestep scalars and broadcast to a spatial shape.

        Args:
            buffer (torch.Tensor): 1-D tensor of shape (num_timesteps,) containing precomputed values.
            t (torch.Tensor): Timestep tensor of shape (B,) with integer values in [0, num_timesteps-1].
            shape (torch.Size): Shape of the target tensor to determine the output shape.
        Returns:
            torch.Tensor: Tensor of shape (B, 1, 1, 1) containing the gathered values broadcasted to the spatial dimensions.
        """
        vals = buffer[t] 
        return vals.view(t.shape[0], *([1] * (len(shape) - 1)))
    
    def __repr__(self) -> str:
        return (
            f"NoiseSchedule(schedule={self.schedule!r}, "
            f"T={self.num_timesteps}, "
            f"β∈[{self.betas.min():.2e}, {self.betas.max():.2e}])"
        )
    
# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for sched in ("linear", "cosine"):
        ns = NoiseSchedule(schedule=sched)
        print(ns)
        assert ns.alphas_cumprod[0] > 0.99,  "ᾱ_0 should be ≈ 1 (little noise)"
        assert ns.alphas_cumprod[-1] < 0.01, "ᾱ_T should be ≈ 0 (almost pure noise)"
        # Forward process roundtrip
        x0 = torch.randn(4, 3, 64, 64)
        t  = torch.randint(0, 1000, (4,))
        eps = torch.randn_like(x0)
        xt  = ns.q_sample(x0, t, noise=eps)

        # Verify the noisy sample has the correct statistics
        sqrt_ab   = ns.sqrt_alphas_cumprod[t].view(4, 1, 1, 1)
        sqrt_1mab = ns.sqrt_one_minus_alphas_cumprod[t].view(4, 1, 1, 1)
        expected  = sqrt_ab * x0 + sqrt_1mab * eps
        assert torch.allclose(xt, expected), "q_sample mismatch"
        print(f"  ᾱ range : [{ns.alphas_cumprod.min():.4f}, {ns.alphas_cumprod.max():.4f}]")
        print(f"  β  range : [{ns.betas.min():.2e},  {ns.betas.max():.2e}]")
        print(f"  q_sample : OK\n")