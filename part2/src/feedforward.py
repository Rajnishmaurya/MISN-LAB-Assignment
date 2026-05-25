"""
Feed-Forward Network (FFN) used in both BERT and GPT blocks.

Standard two-layer MLP with GELU activation:
    FFN(x) = W2(GELU(W1(x)))

Using GELU (Gaussian Error Linear Unit) consistent with BERT/GPT-2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Two-layer FFN with GELU activation.
    No bias in linear layers (consistent with modern LLM practice).

    Args:
        hidden_dim : input/output dimension (d_model)
        ffn_dim    : intermediate dimension (typically 4 * hidden_dim)
        dropout    : dropout on intermediate activations
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1      = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2      = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.gelu(self.w1(x))))
