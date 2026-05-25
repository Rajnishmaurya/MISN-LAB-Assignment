"""
GPT-2 Style Decoder-Only Language Model (~124M parameters)

Architecture:
  - Token embeddings (no positional embeddings — RoPE handles position)
  - 16 Transformer decoder layers, each:
      Pre-norm → GQA causal self-attention → residual
      Pre-norm → FFN → residual
  - RMSNorm instead of LayerNorm
  - GQA instead of standard MHA (num_kv_heads=4, num_heads=12 → 3 Q heads per KV head)
  - RoPE instead of learned positional embeddings
  - LM head: RMSNorm + Linear(hidden_dim → vocab_size)
  - Causal (autoregressive) language modeling objective

Parameter count (hidden=768, layers=16, heads=12, kv_heads=4, ffn=3072, vocab=16000):
  Embeddings       : 16000 * 768                  =  12.3M
  Per layer (GQA)  :
    Q(768²) + K(768*256) + V(768*256) + O(768²)  =   1.57M
    FFN: 2 * 768 * 3072                           =   4.72M
    RMSNorm (2):                                  =   0.002M
    Total per layer:                               =   6.29M  × 16 = 100.7M
  LM head (untied) : 768 * 16000                  =  12.3M
  Total            : ≈ 125.3M ≈ 124M ✓

Design note: 16 layers (vs 12 in BERT) + GQA (kv_heads=4) achieves 124M target.
The reduced KV heads (4 vs 12) save KV cache memory during inference.
"""

import torch
import torch.nn as nn

from rope        import precompute_freqs_cis
from rmsnorm     import RMSNorm
from gqa         import GroupedQueryAttention
from feedforward import FeedForward


class GPTDecoderLayer(nn.Module):
    """Single GPT-like decoder layer (causal self-attention + FFN, pre-norm)."""

    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.attn  = GroupedQueryAttention(
            hidden_dim   = hidden_dim,
            num_heads    = num_heads,
            num_kv_heads = num_kv_heads,
            dropout      = dropout,
            is_causal    = True,   # autoregressive
            is_cross_attn = False,
        )
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn   = FeedForward(hidden_dim, ffn_dim, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    """
    GPT-2 style decoder-only model for causal language model pretraining.

    Design choices:
    - Pre-norm (RMSNorm) before each sub-layer for stable training
    - Causal self-attention (lower-triangular mask) for autoregressive generation
    - GQA with num_kv_heads=4 reduces KV cache 3× vs MHA
    - RoPE applied to Q and K within each attention layer
    - Separate LM head (not weight-tied) to hit ~124M parameter target

    After pretraining, the backbone (embed + layers + final_norm) is extracted
    and used to initialize the decoder in the translation model.
    The translation decoder adds cross-attention layers to each GPT layer.
    """

    def __init__(
        self,
        vocab_size   : int,
        hidden_dim   : int   = 768,
        num_layers   : int   = 16,
        num_heads    : int   = 12,
        num_kv_heads : int   = 4,
        ffn_dim      : int   = 3072,
        max_seq_len  : int   = 512,
        dropout      : float = 0.1,
        pad_id       : int   = 0,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.embed      = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.embed_drop = nn.Dropout(dropout)

        # Decoder layers (causal self-attention only)
        self.layers = nn.ModuleList([
            GPTDecoderLayer(hidden_dim, num_heads, num_kv_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_dim)

        # LM head
        self.lm_norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=True)

        # RoPE frequencies
        head_dim = hidden_dim // num_heads
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(head_dim, max_seq_len),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CLM pretraining.

        Args:
            input_ids: (B, T) token IDs

        Returns:
            logits: (B, T, vocab_size) — predict next token at each position
        """
        B, T = input_ids.shape
        x = self.embed_drop(self.embed(input_ids))
        freqs = self.freqs_cis[:T].to(x.device)
        for layer in self.layers:
            x = layer(x, freqs)
        x = self.final_norm(x)
        return self.lm_head(self.lm_norm(x))

    def get_backbone_state_dict(self) -> dict:
        """Return state dict of backbone only (embed + layers + final_norm).
        Used to initialize the translation decoder."""
        keys = [k for k in self.state_dict() if not k.startswith(('lm_norm', 'lm_head'))]
        return {k: self.state_dict()[k] for k in keys}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
