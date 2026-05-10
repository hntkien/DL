"""Exponential Moving Average (EMA) of model parameters.

Used during DDPM training to maintain a shadow copy of the UNet weights.
The EMA model is used exclusively for inference/sampling, while the raw
model continues to receive gradient updates.

Reference:
    Ho et al., "Denoising Diffusion Probabilistic Models" (2020), §3.3.
    https://arxiv.org/abs/2006.11239
"""
import copy 
import torch 
import torch.nn as nn 

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model weights updated as::

        shadow ← decay x shadow + (1 - decay) x param

    The shadow (EMA) weights are used for sampling; raw weights are used for
    gradient updates. A ``warmup_steps`` guard prevents the shadow from being
    pulled too far toward the initial random weights early in training.

    Args:
        model (nn.Module): The model whose parameters to track.
        decay (float): EMA decay rate. Typical value is 0.9999. Defaults to 0.9999.
        warmup_steps (int): Number of update calls before full ``decay`` kicks in. During warmup the effective decay is ``min(decay, (1+step)/(10+step))``, matching the schedule used in the original DDPM codebase. Defaults to 0 (no warmup).
    """
    def __init__(
            self, 
            model: nn.Module, 
            decay: float = 0.9999, 
            warmup_steps: int = 0,
    ) -> None:
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"Decay must be in [0.0, 1.0), got {decay}")
        self.decay = decay 
        self.warmup_steps = warmup_steps 
        self._step = 0 
        self.shadow = copy.deepcopy(model) 
        self.shadow.eval() 
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad 
    def update(self, model: nn.Module) -> None:
        """Update shadow weights form current model parameters. 

        Applies the EMA update to every parameter tensor. Buffers (e.g., BatchNorm running stats) are copied directly without smoothing, matching standard practise. 
        Args:
            model (nn.Module): The model being trained (source of live weights).
        """
        self._step += 1 

        # Ramp decay up from 0 during warmup to avoid anchoring to random init. 
        if self.warmup_steps > 0: 
            effective_decay = min(
                self.decay, 
                (1.0 + self._step) / (10.0 + self._step)  # Matches original DDPM codebase
            )
        else:
            effective_decay = self.decay

        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(effective_decay).add_(param.data, alpha=1.0 - effective_decay)
        
        # Copy non-parameter buffers directly 
        for s_buffer, buffer in zip(self.shadow.buffers(), model.buffers()):
            s_buffer.data.copy_(buffer.data)

    def state_dict(self) -> dict:
        """Return the full EMA state for checkpointing. 

        Returns:
            dict: Contains ``shadow`` (model state dict), ``decay``,``warmup_steps``, and ``_step``.
        """
        return {
            "shadow":        self.shadow.state_dict(),
            "decay":         self.decay,
            "warmup_steps":  self.warmup_steps,
            "_step":         self._step,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from a checkpoint. 

        Args:
            state_dict (dict): Must contain the same keys as returned by ``state_dict()``.
        """
        self.shadow.load_state_dict(state_dict["shadow"])
        self.decay        = state_dict.get("decay", self.decay)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self._step        = state_dict.get("_step", 0)

    def __repr__(self) -> str:
        return (
            f"EMA(decay={self.decay}, warmup_steps={self.warmup_steps}, "
            f"step={self._step})"
        )