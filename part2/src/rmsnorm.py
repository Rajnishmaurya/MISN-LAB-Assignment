"""
Root Mean Square Layer Normalization (RMSNorm)
Reference: Zhang and Sennrich, Root Mean Square Layer Normalization, NeurIPS 2019.

RMSNorm normalizes by the root mean square of the activations (no mean centering,
no bias). This is simpler and faster than LayerNorm while performing comparably.

Formula: RMSNorm(x) = (x / RMS(x)) * weight
         where RMS(x) = sqrt(mean(x^2) + eps)
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMSNorm: normalize activations by their RMS, then scale by a learnable weight.

    Unlike LayerNorm, there is:
    - No mean subtraction (re-centering)
    - No bias parameter (re-scaling only via weight)
    This reduces computation and parameters while maintaining performance.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x.pow(2).mean(-1, keepdim=True) = mean of squared activations
        # rsqrt = 1 / sqrt
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in float32 for numerical stability, then cast back
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
