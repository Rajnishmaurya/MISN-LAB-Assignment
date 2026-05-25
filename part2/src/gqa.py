"""
Grouped Query Attention (GQA)
Reference: Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models
           from Multi-Head Checkpoints, EMNLP 2023.

GQA interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA):
  - MHA: num_kv_heads == num_heads  (each Q head has its own K, V)
  - MQA: num_kv_heads == 1          (all Q heads share one K, V)
  - GQA: 1 < num_kv_heads < num_heads (groups of Q heads share K, V)

Memory efficiency: K, V cache size reduced by (num_heads / num_kv_heads)x.
num_heads must be divisible by num_kv_heads.

RoPE is applied to Q and K in self-attention.
Cross-attention does NOT apply RoPE (K/V come from encoder context vectors).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import apply_rotary_emb


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention with optional RoPE and optional causal mask.

    Parameters
    ----------
    hidden_dim   : model dimension (d_model)
    num_heads    : number of query heads
    num_kv_heads : number of key/value heads (must divide num_heads)
    dropout      : attention dropout probability
    is_causal    : if True, applies causal (autoregressive) mask in self-attention
    is_cross_attn: if True, this layer is cross-attention (Q from one source,
                   K/V from another); RoPE is NOT applied in cross-attention
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        is_causal: bool = False,
        is_cross_attn: bool = False,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = hidden_dim // num_heads
        self.groups       = num_heads // num_kv_heads  # Q heads per KV head
        self.is_causal    = is_causal
        self.is_cross_attn = is_cross_attn
        self.scale        = math.sqrt(self.head_dim)

        # Q always uses full num_heads
        self.q_proj = nn.Linear(hidden_dim, num_heads    * self.head_dim, bias=False)
        # K, V use reduced num_kv_heads
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads   * self.head_dim, hidden_dim,  bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x             : (B, T, C) — query source (decoder hidden states or encoder)
            freqs_cis     : (T, head_dim // 2) complex — RoPE frequencies for positions 0..T-1
            context       : (B, S, C) — for cross-attention: encoder outputs (K, V source)
                            If None, this is self-attention
            key_padding_mask: (B, S) bool — True where positions should be ignored
        Returns:
            (B, T, C) — attention output
        """
        B, T, C = x.shape

        # ── Project Q, K, V ──────────────────────────────────────────────────
        q = self.q_proj(x)                              # (B, T, num_heads * head_dim)
        kv_src = context if context is not None else x  # cross-attn or self-attn
        S = kv_src.shape[1]
        k = self.k_proj(kv_src)                         # (B, S, num_kv_heads * head_dim)
        v = self.v_proj(kv_src)

        # Reshape to (B, seq, num_heads, head_dim)
        q = q.view(B, T, self.num_heads,    self.head_dim)
        k = k.view(B, S, self.num_kv_heads, self.head_dim)
        v = v.view(B, S, self.num_kv_heads, self.head_dim)

        # ── Apply RoPE to Q and K (only in self-attention) ────────────────────
        if not self.is_cross_attn and freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # ── Transpose to (B, num_heads, seq, head_dim) ───────────────────────
        q = q.transpose(1, 2)   # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)   # (B, num_kv_heads, S, head_dim)
        v = v.transpose(1, 2)   # (B, num_kv_heads, S, head_dim)

        # ── Expand KV heads to match Q heads (GQA repeat-interleave) ─────────
        if self.groups > 1:
            # Each KV head serves self.groups Q heads
            k = k.repeat_interleave(self.groups, dim=1)  # (B, num_heads, S, head_dim)
            v = v.repeat_interleave(self.groups, dim=1)

        # ── Scaled dot-product attention ──────────────────────────────────────
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, S)

        # Causal mask: upper triangle → -inf
        if self.is_causal:
            causal_mask = torch.triu(
                torch.full((T, T), float('-inf'), device=x.device), diagonal=1
            )
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

        # Padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (B, S) True = ignore
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn_weights = F.softmax(attn_weights, dim=-1).nan_to_num(0.0)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)              # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, C)
        return self.o_proj(out)
