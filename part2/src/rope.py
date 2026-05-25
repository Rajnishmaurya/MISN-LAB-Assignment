"""
Rotary Positional Embeddings (RoPE)
Reference: Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding, 2021.

RoPE encodes position by rotating the query/key vectors in complex space.
No positional embedding parameters — purely functional.
"""

import torch


def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the complex frequency tensor used in RoPE.

    For each pair of dimensions (2i, 2i+1), the rotation frequency is:
        theta_i = 1 / (theta ^ (2i / head_dim))

    Returns a complex tensor of shape (max_seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    # freqs[i] = 1 / theta^(2i / head_dim)  shape: (head_dim // 2,)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # positions: (max_seq_len,)
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # outer product: (max_seq_len, head_dim // 2)
    freqs = torch.outer(t, freqs)
    # complex form: e^(i * freq) = cos(freq) + i*sin(freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis  # (max_seq_len, head_dim // 2), complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    Args:
        xq: (batch, seq_len, num_heads, head_dim)
        xk: (batch, seq_len, num_kv_heads, head_dim)
        freqs_cis: (seq_len, head_dim // 2)  complex

    Returns:
        xq_out, xk_out: same shape as inputs, with RoPE applied
    """
    # Reshape to complex: (..., head_dim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis: (seq_len, head_dim // 2) → broadcast over batch and heads
    # xq_: (batch, seq_len, num_heads, head_dim // 2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
