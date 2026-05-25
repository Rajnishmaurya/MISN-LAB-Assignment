"""
Masked Language Modeling (MLM) Dataset for BERT pretraining.

Masking strategy (identical to original BERT paper):
  - Select 15% of non-special tokens as candidates
  - Of those:
      80% replace with [MASK] token
      10% replace with a random vocabulary token
      10% keep unchanged
  - Only the selected positions contribute to the MLM loss

Special token IDs (must match tokenizer):
  PAD  = 0
  BOS  = 1
  EOS  = 2
  UNK  = 3
  MASK = 4
"""

import random
import torch
from torch.utils.data import Dataset
import sentencepiece as spm


PAD_ID  = 0
BOS_ID  = 1
EOS_ID  = 2
UNK_ID  = 3
MASK_ID = 4

SPECIAL_IDS = {PAD_ID, BOS_ID, EOS_ID, UNK_ID, MASK_ID}


class MLMDataset(Dataset):
    """
    Dataset for BERT masked language model pretraining.

    Each sample:
      input_ids  : (max_len,)  token IDs with some replaced by MASK/random
      labels     : (max_len,)  original token IDs at masked positions, -100 elsewhere
      attn_mask  : (max_len,)  1 = real token, 0 = padding

    Args:
        text_file  : path to plain text file, one sentence per line
        spm_model  : path to SentencePiece model file
        max_len    : maximum sequence length (including BOS/EOS)
        mask_prob  : fraction of eligible tokens to mask (default 0.15)
        vocab_size : total vocabulary size (needed for random token replacement)
    """

    def __init__(
        self,
        text_file  : str,
        spm_model  : str,
        max_len    : int   = 512,
        mask_prob  : float = 0.15,
        vocab_size : int   = 16000,
    ):
        self.sp         = spm.SentencePieceProcessor(model_file=spm_model)
        self.max_len    = max_len
        self.mask_prob  = mask_prob
        self.vocab_size = vocab_size

        with open(text_file, encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> dict:
        tokens = self.sp.encode(self.lines[idx])

        # Add BOS/EOS and truncate to max_len
        tokens = [BOS_ID] + tokens[:self.max_len - 2] + [EOS_ID]

        # Mask tokens
        input_ids, labels = self._mask(tokens)

        # Pad to max_len
        seq_len   = len(input_ids)
        pad_len   = self.max_len - seq_len
        attn_mask = [1] * seq_len + [0] * pad_len
        input_ids = input_ids  + [PAD_ID] * pad_len
        labels    = labels     + [-100]   * pad_len

        return {
            'input_ids' : torch.tensor(input_ids,  dtype=torch.long),
            'labels'    : torch.tensor(labels,     dtype=torch.long),
            'attn_mask' : torch.tensor(attn_mask,  dtype=torch.long),
        }

    def _mask(self, tokens: list[int]) -> tuple[list[int], list[int]]:
        """Apply BERT masking to a token sequence."""
        input_ids = tokens.copy()
        labels    = [-100] * len(tokens)  # -100 = ignore in CrossEntropyLoss

        # Eligible positions: non-special tokens
        eligible = [i for i, t in enumerate(tokens) if t not in SPECIAL_IDS]

        # Sample 15%
        n_mask  = max(1, int(len(eligible) * self.mask_prob))
        masked  = random.sample(eligible, min(n_mask, len(eligible)))

        for i in masked:
            labels[i] = tokens[i]  # store original for loss computation

            r = random.random()
            if r < 0.80:
                input_ids[i] = MASK_ID
            elif r < 0.90:
                # Random token (exclude special IDs)
                input_ids[i] = random.randint(len(SPECIAL_IDS), self.vocab_size - 1)
            # else: keep original (10%)

        return input_ids, labels
