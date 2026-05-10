from typing import Optional, List, Tuple 
import torch 
import torch.nn as nn
import torch.nn.functional as F 

from diffusion.noise_schedule import NoiseSchedule
from models.unet import UNet

class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Models (DDPM) implementation.

    Wraps a noise-prediction U-Net and a pre-computed noise schedule. Conditioning is supplied as a multi-hot label vector that matches the channel layout in `objects.json`. 

    Args: 
        model (nn.Module): The noise-prediction model, typically a U-Net.
        schedule (NoiseSchedule): The noise schedule defining the forward diffusion process.
    """
    def __init__(
            self, 
            model: UNet,
            schedule: NoiseSchedule
    ) -> None: 
        super().__init__() 
        self.model = model 
        self.schedule = schedule
        self.num_timesteps = schedule.num_timesteps
        self.num_classes = model.num_classes 

    def training_loss(self, x0: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Compute the simple noise-prediction MSE loss for one batch. 

        CFG-style condition dropping is perfomed upstream in the dataset (`drop_prob`) so `condition` may already contain zero rows here. 

        Args:
            x0 (torch.Tensor): Clean image batch of shape (B, C, H, W).
            condition (torch.Tensor): Multi-hot class vector of shape (B, num_classes) for classifier-free guidance. May contain zero rows for dropped conditions.

        Returns:
            torch.Tensor: Scalar loss tensor (mean MSE over all elements and batch).
        """
        B = x0.shape[0] 
        t = torch.randint(0, self.num_timesteps, (B,), device=x0.device)  # Random timesteps for each sample in the batch
        noise = torch.randn_like(x0)  # Sample noise
        x_t = self.schedule.q_sample(x0, t, noise=noise)  # Diffuse the clean image to get the noisy image at timestep t
        noise_pred = self.model(x_t, t, condition) 

        return F.mse_loss(noise_pred, noise)  # MSE loss between the predicted noise and the true noise
    
    @torch.no_grad() 
    def sample(
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
        intermediates = [] 

        # Reverse process 
        for step in reversed(range(self.num_timesteps)):
            t = torch.full((B,), step, device=device, dtype=torch.long) 

            # --- CFG: Combine conditional and unconditional predictions ---
            if guidance_scale != 1.0:
                # Batch the two forward passes for efficiency.
                x_in = torch.cat([x, x], dim=0)  # (2B, C, H, W)
                t_in = torch.cat([t, t], dim=0)  # (2B,)
                c_in = torch.cat([null_cond, condition], dim=0)  # (2B, num_classes)
                eps_uncond, eps_cond = self.model(x_in, t_in, c_in).chunk(2, dim=0)  # Each is (B, C, H, W)
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
        return cls(model, schedule)
    
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