from typing import Optional, List, Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.noise_schedule import NoiseSchedule
from diffusion.sampler import ddim_sample
from models.unet import UNet


SamplerKind = Literal["ddpm", "ddim"]

class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Models (DDPM) wrapper.

    Wraps a noise-prediction U-Net and a pre-computed noise schedule. Conditioning is supplied as a multi-hot label vector that matches the channel layout in ``objects.json``. Supports DDPM and DDIM samplers and optional Min-SNR-γ loss weighting (Hang et al. 2023, https://arxiv.org/abs/2303.09556).

    Args:
        model (UNet): The noise-prediction U-Net.
        schedule (NoiseSchedule): The pre-computed noise schedule.
        min_snr_gamma (float | None): If not None and > 0, weight the per-sample ε-prediction loss by ``min(γ, SNR(t)) / SNR(t)``. ``None`` or 0 disables weighting (uniform per-t loss). Defaults to None.
    """
    def __init__(
            self, 
            model: UNet,
            schedule: NoiseSchedule, 
            min_snr_gamma: Optional[float] = None
    ) -> None: 
        super().__init__() 
        self.model = model 
        self.schedule = schedule
        self.num_timesteps = schedule.num_timesteps
        self.num_classes = model.num_classes 
        self.min_snr_gamma = min_snr_gamma

    def training_loss(self, x0: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Compute the ε-prediction loss for one batch, optionally Min-SNR weighted.

        With ``min_snr_gamma`` set (recommended γ=5), each sample's MSE is multiplied by ``min(γ, SNR(t)) / SNR(t)`` where ``SNR(t) = ᾱ_t / (1 - ᾱ_t)``. This downweights easy small-t samples that otherwise dominate the average loss, focusing gradients on the harder high-t regime where conditioning matters most.

        CFG-style condition dropping is performed upstream in the dataset (``drop_prob``); ``condition`` may already contain zero rows here.

        Args:
            x0 (torch.Tensor): Clean image batch of shape (B, C, H, W).
            condition (torch.Tensor): Multi-hot class vector of shape (B, num_classes). May contain all-zero rows (CFG null token).

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        B = x0.shape[0] 
        t = torch.randint(0, self.num_timesteps, (B,), device=x0.device)  # Random timesteps for each sample in the batch
        noise = torch.randn_like(x0)  # Sample noise
        x_t = self.schedule.q_sample(x0, t, noise=noise)  # Diffuse the clean image to get the noisy image at timestep t
        noise_pred = self.model(x_t, t, condition) 

        # Per-sample MSE averaged over (C, H, W); shape (B,)
        mse = (noise_pred - noise).pow(2).mean(dim=[1, 2, 3])

        if self.min_snr_gamma is not None and self.min_snr_gamma > 0:
            alphas_cumprod = self.schedule.alphas_cumprod[t]              # (B,)
            snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp(min=1e-8)  # (B,)
            weight = torch.clamp(snr, max=self.min_snr_gamma) / snr        # (B,)
            return (weight * mse).mean()
        
        return mse.mean()
    
    @torch.no_grad()
    def sample(
            self,
            condition: torch.Tensor,
            image_size: int = 64,
            guidance_scale: float = 2.0,
            sampler: SamplerKind = "ddpm",
            ddim_steps: int = 50,
            ddim_eta: float = 0.0,
            return_immediates: bool = False,
            intermedate_every: int = 100,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate images with classifier-free guidance.

        Dispatches to the DDPM ancestral sampler or the DDIM sampler.

        Args:
            condition (torch.Tensor): Multi-hot label tensor of shape (B, num_classes).
            image_size (int): Spatial size of the generated images. Defaults to 64.
            guidance_scale (float): CFG weight ``w``. ``1.0`` disables guidance. Defaults to 2.0.
            sampler (str): ``"ddpm"`` for ancestral sampling (T steps) or ``"ddim"`` for DDIM (``ddim_steps`` steps). Defaults to ``"ddpm"``.
            ddim_steps (int): Number of DDIM steps. Ignored when ``sampler="ddpm"``. Defaults to 50.
            ddim_eta (float): DDIM stochasticity. 0.0 = deterministic. Defaults to 0.0.
            return_immediates (bool): If True, also return intermediate xₜ samples. Defaults to False.
            intermedate_every (int): Interval between recorded intermediates (in the original timestep index for DDPM, in DDIM-step index for DDIM). Defaults to 100.

        Returns:
            torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
                Generated images, or (images, intermediates) if requested.
        """
        if sampler == "ddim":
            return ddim_sample(
                model=self.model,
                schedule=self.schedule,
                condition=condition,
                image_size=image_size,
                num_steps=ddim_steps,
                eta=ddim_eta,
                guidance_scale=guidance_scale,
                return_intermediates=return_immediates,
                intermediate_every=max(1, ddim_steps // max(1, intermedate_every // (self.num_timesteps // ddim_steps))),
            )
        elif sampler == "ddpm":
            return self._sample_ddpm(
                condition=condition,
                image_size=image_size,
                guidance_scale=guidance_scale,
                return_immediates=return_immediates,
                intermedate_every=intermedate_every,
            )
        else:
            raise ValueError(f"Unknown sampler '{sampler}'. Use 'ddpm' or 'ddim'.")
    
    @torch.no_grad() 
    def _sample_ddpm(
        self,
        condition: torch.Tensor, 
        image_size: int = 64, 
        guidance_scale: float = 2.0, 
        return_immediates: bool = False, 
        intermedate_every: int = 100,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]: 
        """Generate images via ancestral DDPM sampling with CFG.

        At each timestep, two U-Net forward passes are run — one conditioned
        on the supplied label vector, one on the zero (null) vector — and
        the noise predictions are combined as

            ε̃ = ε_uncond + w · (ε_cond - ε_uncond)

        where `w = guidance_scale`. The combined ε̃ is then used to compute
        the posterior mean μ̃_t, and the next sample x_{t-1} is drawn from
        N(μ̃_t, β̃_t · I) (deterministic at t = 0).

        Args:
            condition (torch.Tensor): Multi-hot lablel tensor of shape (B, num_classes), values in {0, 1}. 
            image_size (int, optional): Spatial size of the generated images. Defaults to 64.
            guidance_scale (float, optional): Guidance scale for classifier-free guidance. Defaults to 2.0.
            return_immediates (bool, optional): Whether to return intermediate samples. Defaults to False.
            intermedate_every (int, optional): Frequency of returning intermediate samples. Defaults to 100.

        Returns:
            torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]: If `return_immediates` is False, returns a batch of generated images of shape (B, C, H, W). If True, returns a tuple of (final_images, intermediates) where `final_images` is the same as before and `intermediates` is a list of noisy images at the specified intervals.
        """
        device = condition.device 
        B = condition.shape[0] 

        # Start from pure Gaussian noise 
        x = torch.randn(B, 3, image_size, image_size, device=device) 
        null_cond = torch.zeros_like(condition) 
        intermediates: List[torch.Tensor] = []

        # Reverse process 
        for step in reversed(range(self.num_timesteps)):
            t = torch.full((B,), step, device=device, dtype=torch.long) 

            # --- CFG: Combine conditional and unconditional predictions ---
            if guidance_scale > 1.0:
                # Batch the two forward passes for efficiency.
                x_in = torch.cat([x, x], dim=0)  # (2B, C, H, W)
                t_in = torch.cat([t, t], dim=0)  # (2B,)
                c_in = torch.cat([condition, null_cond], dim=0)   # (2B, num_classes)
                eps_cond, eps_uncond = self.model(x_in, t_in, c_in).chunk(2, dim=0)  # Each is (B, C, H, W)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)  # (B, C, H, W)
            else:
                eps = self.model(x, t, condition)  # (B, C, H, W)

            # --- Posterior mean and reverse-step sampling ---
            mean = self.schedule.q_posterior_mean(x, t, eps)  # (B, C, H, W)

            if step > 0:
                noise = torch.randn_like(x)
                log_var = self.schedule.posterior_log_variance_clipped[t]  # (B,)
                x = mean + (0.5 * log_var).exp().view(B, 1, 1, 1) * noise 
            else:
                x = mean 

            # Save intermediates if requested
            if return_immediates and (step % intermedate_every == 0 or step == 0):
                intermediates.append(x.detach().cpu())
        
        if return_immediates:
            return x, intermediates
        else:
            return x
        
    # ---------------------------------------------------------------------- #
    # Build a DDPN from a config dict                                        #
    # ---------------------------------------------------------------------- #
    @classmethod
    def from_config(cls, config: dict) -> "DDPM":
        """Instantiate a DDPM from a parsed YAML config dictionary. 

        Expected layout:
            cfg["model"]    → kwargs for `UNet`
            cfg["schedule"] → kwargs for `NoiseSchedule`

        Args:
            config (dict): Parsed config dictionary. 

        Returns:
            DDPM: A new `DDPM` instance. 
        """
        model = UNet(**config["model"])
        schedule = NoiseSchedule(**config["schedule"])
        min_snr_gamma = config.get("training", {}).get("min_snr_gamma", None)
        return cls(model, schedule, min_snr_gamma=min_snr_gamma)
    
# ---------------------------------------------------------------------------
# Quick smoke test
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
    schedule = NoiseSchedule(schedule="linear", num_timesteps=20)
    ddpm     = DDPM(model, schedule)
    x0   = torch.randn(2, 3, 64, 64)
    cond = torch.zeros(2, 24)
    cond[0, [2, 9]] = 1.0
    cond[1, [0]]    = 1.0
    loss = ddpm.training_loss(x0, cond)
    print(f"training_loss : {loss.item():.4f}")

    samples = ddpm.sample(cond, guidance_scale=2.0)
    print(f"samples shape : {tuple(samples.shape)}  range=[{samples.min():.2f}, {samples.max():.2f}]")
    assert samples.shape == (2, 3, 64, 64)
    print("DDPM smoke test passed.")