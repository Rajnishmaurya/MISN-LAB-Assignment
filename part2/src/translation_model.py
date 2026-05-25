"""
Encoder-Decoder Translation Model

Architecture:
  Encoder : pretrained BERT-like model (bidirectional, 12 layers)
  Decoder : GPT-backbone + cross-attention layers (16 layers)

Each decoder layer has THREE sub-layers:
  1. Causal self-attention (from GPT backbone — pretrained)
  2. Cross-attention      (new — Q from decoder, K/V from encoder outputs)
  3. FFN                  (from GPT backbone — pretrained)

Weight initialization for fine-tuning:
  - Encoder: load from BERT MLM checkpoint (BERTModel.state_dict)
  - Decoder self-attention + FFN: load from GPT CLM checkpoint (GPTModel backbone)
  - Decoder cross-attention: random initialization (learned from scratch)
  - Translation head: random initialization

The cross-attention does NOT apply RoPE (encoder outputs are context vectors,
not position-dependent keys). Only decoder self-attention uses RoPE.

This design allows the model to leverage:
  - BERT's bidirectional contextual representations for the source language
  - GPT's strong generative prior for the target language
"""

import torch
import torch.nn as nn

from rope        import precompute_freqs_cis
from rmsnorm     import RMSNorm
from gqa         import GroupedQueryAttention
from feedforward import FeedForward


class TranslationDecoderLayer(nn.Module):
    """
    Decoder layer for translation: causal self-attn + cross-attn + FFN.

    Self-attention and FFN are initialized from the pretrained GPT backbone.
    Cross-attention is randomly initialized.
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int,
                 num_kv_heads_cross: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()

        # Sub-layer 1: causal self-attention (from GPT)
        self.norm1      = RMSNorm(hidden_dim)
        self.self_attn  = GroupedQueryAttention(
            hidden_dim, num_heads, num_kv_heads,
            dropout=dropout, is_causal=True, is_cross_attn=False
        )

        # Sub-layer 2: cross-attention (new, random init)
        self.norm_cross = RMSNorm(hidden_dim)
        self.cross_attn = GroupedQueryAttention(
            hidden_dim, num_heads, num_kv_heads_cross,
            dropout=dropout, is_causal=False, is_cross_attn=True
        )

        # Sub-layer 3: FFN (from GPT)
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn   = FeedForward(hidden_dim, ffn_dim, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x          : torch.Tensor,       # (B, T, C) decoder hidden states
        freqs_cis  : torch.Tensor,       # (T, head_dim//2) RoPE for target positions
        enc_out    : torch.Tensor,       # (B, S, C) encoder outputs
        enc_pad_mask: torch.Tensor | None = None,  # (B, S) encoder padding
    ) -> torch.Tensor:
        # 1. Causal self-attention on target tokens
        x = x + self.drop(self.self_attn(self.norm1(x), freqs_cis=freqs_cis))
        # 2. Cross-attention: Q from x, K/V from encoder
        x = x + self.drop(self.cross_attn(
            self.norm_cross(x),
            freqs_cis=None,
            context=enc_out,
            key_padding_mask=enc_pad_mask,
        ))
        # 3. FFN
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TranslationModel(nn.Module):
    """
    Full encoder-decoder translation model.

    Usage:
        model = TranslationModel(config)
        model.load_pretrained(bert_ckpt, gpt_ckpt)
        logits = model(src_ids, tgt_ids)
    """

    def __init__(
        self,
        vocab_size         : int,
        enc_hidden_dim     : int   = 768,
        enc_num_layers     : int   = 12,
        enc_num_heads      : int   = 12,
        enc_num_kv_heads   : int   = 12,
        enc_ffn_dim        : int   = 3072,
        dec_hidden_dim     : int   = 768,
        dec_num_layers     : int   = 16,
        dec_num_heads      : int   = 12,
        dec_num_kv_heads   : int   = 4,
        dec_num_kv_heads_cross: int = 4,
        dec_ffn_dim        : int   = 3072,
        max_seq_len        : int   = 128,
        dropout            : float = 0.1,
        pad_id             : int   = 0,
    ):
        super().__init__()
        self.pad_id      = pad_id
        self.dec_hidden  = dec_hidden_dim

        # ── Encoder (BERT backbone) ──────────────────────────────────────────
        self.enc_embed     = nn.Embedding(vocab_size, enc_hidden_dim, padding_idx=pad_id)
        self.enc_embed_drop = nn.Dropout(dropout)
        self.enc_layers    = nn.ModuleList([
            _make_bert_layer(enc_hidden_dim, enc_num_heads, enc_num_kv_heads, enc_ffn_dim, dropout)
            for _ in range(enc_num_layers)
        ])
        self.enc_norm = RMSNorm(enc_hidden_dim)

        # Projection if encoder and decoder dimensions differ
        self.enc_to_dec = (
            nn.Linear(enc_hidden_dim, dec_hidden_dim, bias=False)
            if enc_hidden_dim != dec_hidden_dim else nn.Identity()
        )

        # ── Decoder (GPT backbone + cross-attention) ─────────────────────────
        self.dec_embed     = nn.Embedding(vocab_size, dec_hidden_dim, padding_idx=pad_id)
        self.dec_embed_drop = nn.Dropout(dropout)
        self.dec_layers    = nn.ModuleList([
            TranslationDecoderLayer(
                dec_hidden_dim, dec_num_heads, dec_num_kv_heads,
                dec_num_kv_heads_cross, dec_ffn_dim, dropout
            )
            for _ in range(dec_num_layers)
        ])
        self.dec_norm = RMSNorm(dec_hidden_dim)

        # Translation output head
        self.lm_head = nn.Linear(dec_hidden_dim, vocab_size, bias=False)

        # RoPE for encoder and decoder
        enc_head_dim = enc_hidden_dim // enc_num_heads
        dec_head_dim = dec_hidden_dim // dec_num_heads
        self.register_buffer('enc_freqs_cis',
                             precompute_freqs_cis(enc_head_dim, max_seq_len), persistent=False)
        self.register_buffer('dec_freqs_cis',
                             precompute_freqs_cis(dec_head_dim, max_seq_len), persistent=False)

        self._init_new_weights()

    def _init_new_weights(self):
        """Initialize weights for newly-added modules (cross-attention, head)."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def load_pretrained(
        self,
        bert_checkpoint: str,
        gpt_checkpoint : str,
        device         : torch.device = torch.device('cpu'),
    ):
        """
        Load pretrained weights from BERT and GPT checkpoints.

        BERT checkpoint → encoder embed + enc_layers + enc_norm
        GPT checkpoint  → decoder embed + dec_layers (self-attn + FFN) + dec_norm

        Cross-attention layers and lm_head remain randomly initialized.
        """
        bert_sd = torch.load(bert_checkpoint, map_location=device, weights_only=True)
        gpt_sd  = torch.load(gpt_checkpoint,  map_location=device, weights_only=True)

        # ── Load encoder from BERT ─────────────────────────────────────────────
        # BERT state dict key layout:
        #   embed.weight          → enc_embed.weight
        #   layers.{i}.*          → enc_layers.{i}.*
        #   final_norm.weight     → enc_norm.weight
        #   mlm_norm.*, mlm_head.*  (skip — MLM-only heads)
        enc_state = {}
        for k, v in bert_sd.items():
            if k.startswith('embed.'):
                enc_state['enc_' + k] = v                       # enc_embed.*
            elif k.startswith('layers.'):
                enc_state['enc_' + k] = v                       # enc_layers.{i}.*
            elif k.startswith('final_norm.'):
                enc_state['enc_norm.' + k.split('.', 1)[1]] = v # enc_norm.*
            # skip: mlm_norm, mlm_head, embed_drop, freqs_cis

        missing_enc, _ = self.load_state_dict(enc_state, strict=False)
        print(f"[Encoder] Loaded {len(enc_state)} tensors from BERT checkpoint")
        enc_missing = [k for k in missing_enc if k.startswith('enc_')]
        if enc_missing:
            print(f"[Encoder] WARNING — missing keys: {enc_missing[:5]}")

        # ── Load decoder backbone from GPT ────────────────────────────────────
        # GPT state dict key layout:
        #   embed.weight          → dec_embed.weight
        #   layers.{i}.norm1.*   → dec_layers.{i}.norm1.*       (self-attn pre-norm)
        #   layers.{i}.attn.*    → dec_layers.{i}.self_attn.*   ← NAME CHANGE
        #   layers.{i}.norm2.*   → dec_layers.{i}.norm2.*       (FFN pre-norm)
        #   layers.{i}.ffn.*     → dec_layers.{i}.ffn.*
        #   layers.{i}.drop.*    (skip — dropout, no weights)
        #   final_norm.weight     → dec_norm.*
        #   lm_norm.*, lm_head.* (skip — CLM-only heads)
        dec_state = {}
        for k, v in gpt_sd.items():
            if k.startswith('embed.'):
                dec_state['dec_' + k] = v                        # dec_embed.*
            elif k.startswith('layers.'):
                # GPT uses 'attn' as module name; TranslationDecoderLayer uses 'self_attn'
                # Replace: layers.{i}.attn.* → dec_layers.{i}.self_attn.*
                new_k = 'dec_' + k.replace('.attn.', '.self_attn.')
                dec_state[new_k] = v
            elif k.startswith('final_norm.'):
                dec_state['dec_norm.' + k.split('.', 1)[1]] = v  # dec_norm.*
            # skip: lm_norm, lm_head, embed_drop, freqs_cis

        missing_dec, _ = self.load_state_dict(dec_state, strict=False)
        print(f"[Decoder] Loaded {len(dec_state)} tensors from GPT checkpoint")

        # Cross-attention + norm_cross are expected to be missing (new, random init)
        n_cross = sum(1 for k in missing_dec if 'cross_attn' in k or 'norm_cross' in k)
        other_missing = [k for k in missing_dec
                         if 'cross_attn' not in k and 'norm_cross' not in k
                         and k.startswith('dec_')]
        print(f"[Decoder] Cross-attention params (random init, expected): {n_cross}")
        if other_missing:
            print(f"[Decoder] WARNING — unexpected missing: {other_missing[:5]}")

    def encode(self, src_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode source sentence. Returns (encoder_output, padding_mask)."""
        B, S = src_ids.shape
        pad_mask = (src_ids == self.pad_id)  # (B, S)
        x = self.enc_embed_drop(self.enc_embed(src_ids))
        freqs = self.enc_freqs_cis[:S].to(x.device)
        for layer in self.enc_layers:
            x = layer(x, freqs, padding_mask=pad_mask)
        x = self.enc_norm(x)
        return self.enc_to_dec(x), pad_mask

    def decode_step(
        self,
        tgt_ids  : torch.Tensor,   # (B, T) generated so far
        enc_out  : torch.Tensor,   # (B, S, C) encoder output
        enc_mask : torch.Tensor,   # (B, S) encoder padding mask
    ) -> torch.Tensor:
        """One decoder forward pass (used in beam search)."""
        B, T = tgt_ids.shape
        x = self.dec_embed_drop(self.dec_embed(tgt_ids))
        freqs = self.dec_freqs_cis[:T].to(x.device)
        for layer in self.dec_layers:
            x = layer(x, freqs, enc_out, enc_mask)
        x = self.dec_norm(x)
        return self.lm_head(x)  # (B, T, vocab_size)

    def forward(
        self,
        src_ids : torch.Tensor,   # (B, S)
        tgt_ids : torch.Tensor,   # (B, T)
    ) -> torch.Tensor:
        """
        Full forward pass for teacher-forcing training.
        Returns logits: (B, T, vocab_size)
        """
        enc_out, enc_mask = self.encode(src_ids)
        return self.decode_step(tgt_ids, enc_out, enc_mask)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        enc   = sum(p.numel() for name, p in self.named_parameters() if name.startswith('enc_'))
        dec   = sum(p.numel() for name, p in self.named_parameters() if name.startswith('dec_'))
        cross = sum(p.numel() for name, p in self.named_parameters() if 'cross_attn' in name or 'norm_cross' in name)
        return {'total': total, 'encoder': enc, 'decoder': dec, 'cross_attn_new': cross}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bert_layer(hidden_dim, num_heads, num_kv_heads, ffn_dim, dropout):
    """Create a BERT encoder layer (imported to avoid circular imports)."""
    from bert_model import BERTEncoderLayer
    return BERTEncoderLayer(hidden_dim, num_heads, num_kv_heads, ffn_dim, dropout)
