"""DDIM sampler with classifier-free guidance.

References:
    Song et al., "Denoising Diffusion Implicit Models" (2020),
    https://arxiv.org/abs/2010.02502.
"""
from typing import List, Tuple
import torch
import torch.nn as nn

from diffusion.noise_schedule import NoiseSchedule


@torch.no_grad()
def ddim_sample(
        model: nn.Module,
        schedule: NoiseSchedule,
        condition: torch.Tensor,
        image_size: int = 64,
        num_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        clip_x0: bool = True,
        return_intermediates: bool = False,
        intermediate_every: int = 5,
) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
    """Generate samples using DDIM with classifier-free guidance.

    Builds a uniformly-spaced sub-sequence of ``num_steps`` timesteps from the
    schedule's ``num_timesteps`` and integrates the deterministic (η=0) or
    partially-stochastic (η>0) reverse process. ``η=1`` recovers DDPM
    statistics (with sub-sampled timesteps).

    The update at step ``t -> t_prev`` is::

        x̂_0 = (x_t - sqrt(1-ᾱ_t) · ε̃) / sqrt(ᾱ_t)            (predict x_0)
        σ_t = η · sqrt((1-ᾱ_{t_prev}) / (1-ᾱ_t)) · sqrt(1 - ᾱ_t/ᾱ_{t_prev})
        dir = sqrt(1 - ᾱ_{t_prev} - σ_t²) · ε̃                  (direction to x_t)
        x_{t_prev} = sqrt(ᾱ_{t_prev}) · x̂_0 + dir + σ_t · z

    where ``ε̃ = ε_uncond + w · (ε_cond - ε_uncond)`` is the CFG-combined noise.

    Args:
        model (nn.Module): The trained noise-prediction U-Net.
        schedule (NoiseSchedule): Pre-computed noise schedule.
        condition (torch.Tensor): Multi-hot label tensor of shape (B, num_classes).
        image_size (int): Spatial size of generated images. Defaults to 64.
        num_steps (int): Number of DDIM steps (sub-sampled from
            ``schedule.num_timesteps``). Defaults to 50.
        eta (float): DDIM stochasticity. 0.0 = deterministic, 1.0 ≈ DDPM.
            Defaults to 0.0.
        guidance_scale (float): CFG weight ``w``; 1.0 disables guidance.
            Defaults to 1.0.
        clip_x0 (bool): Clip the predicted x̂_0 to ``[-1, 1]`` before each step
            for numerical stability. Defaults to True.
        return_intermediates (bool): If True, also return intermediate samples.
            Defaults to False.
        intermediate_every (int): Save an intermediate every N DDIM steps.
            Defaults to 5.

    Returns:
        torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
            Generated images of shape (B, 3, H, W) in [-1, 1], or
            ``(images, intermediates)`` if ``return_intermediates`` is True.
    """
    device = condition.device
    B = condition.shape[0]
    T = schedule.num_timesteps

    # Uniformly-spaced sub-sequence including 0 and T-1; ascending.
    # e.g., T=1000, num_steps=50 → [0, 20, ..., 980, 999]
    seq = torch.linspace(0, T - 1, num_steps, dtype=torch.long).tolist()
    # Pair each step (descending) with the next (smaller) step we transition to.
    # The last step uses t_prev = -1, which we treat as ᾱ_prev = 1.
    seq_rev = list(reversed(seq))
    seq_prev = seq_rev[1:] + [-1]

    x = torch.randn(B, 3, image_size, image_size, device=device)
    null_cond = torch.zeros_like(condition)
    intermediates: List[torch.Tensor] = []

    for i, (t_cur, t_prev) in enumerate(zip(seq_rev, seq_prev)):
        t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

        # --- CFG noise prediction --- #
        if guidance_scale > 1.0:
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t_batch, t_batch], dim=0)
            c_in = torch.cat([condition, null_cond], dim=0)
            eps_cond, eps_uncond = model(x_in, t_in, c_in).chunk(2, dim=0)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = model(x, t_batch, condition)

        # --- Gather ᾱ values --- #
        a_cur = schedule.alphas_cumprod[t_cur]
        a_prev = (
            schedule.alphas_cumprod[t_prev]
            if t_prev >= 0
            else torch.tensor(1.0, device=device, dtype=a_cur.dtype)
        )

        # --- Predict x_0 from x_t and ε --- #
        sqrt_a_cur = a_cur.sqrt()
        sqrt_1m_a_cur = (1.0 - a_cur).clamp(min=0).sqrt()
        x0_hat = (x - sqrt_1m_a_cur * eps) / sqrt_a_cur.clamp(min=1e-8)
        if clip_x0:
            x0_hat = x0_hat.clamp(-1.0, 1.0)

        # --- DDIM noise variance σ_t --- #
        # σ_t² = η² · (1-ᾱ_prev)/(1-ᾱ_cur) · (1 - ᾱ_cur/ᾱ_prev)
        sigma = (
            eta
            * ((1.0 - a_prev) / (1.0 - a_cur).clamp(min=1e-8)).sqrt()
            * (1.0 - a_cur / a_prev.clamp(min=1e-8)).clamp(min=0).sqrt()
        )

        # --- Direction towards x_t_prev --- #
        dir_coef = (1.0 - a_prev - sigma ** 2).clamp(min=0).sqrt()
        dir_xt = dir_coef * eps

        # --- Sample x_{t_prev} --- #
        noise = torch.randn_like(x) if (t_prev >= 0 and eta > 0) else torch.zeros_like(x)
        x = a_prev.sqrt() * x0_hat + dir_xt + sigma * noise

        if return_intermediates and ((num_steps - 1 - i) % intermediate_every == 0):
            intermediates.append(x.detach().cpu())

    if return_intermediates:
        return x, intermediates
    return x