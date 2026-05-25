"""
Causal Language Modeling (CLM) Dataset for GPT pretraining.

Each sample is a sequence of tokens where the model learns to predict the
next token at every position:
    input : [BOS, t1, t2, ..., t_{n-1}]
    labels: [t1,  t2, t3, ..., t_n,    EOS]

Or equivalently, given input_ids, labels = input_ids shifted left by 1.
This is the standard GPT/CLM objective.
"""

import torch
from torch.utils.data import Dataset
import sentencepiece as spm


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


class CLMDataset(Dataset):
    """
    Dataset for GPT causal language model pretraining.

    Each sample:
      input_ids : (max_len,)  [BOS, t1, t2, ..., t_{n-1}, PAD, ...]
      labels    : (max_len,)  [t1, t2, ..., t_n, EOS, -100, ...]
                              -100 at padding positions (ignored in loss)

    Args:
        text_file : path to plain text file, one sentence per line
        spm_model : path to SentencePiece model file
        max_len   : maximum sequence length (including BOS/EOS)
    """

    def __init__(self, text_file: str, spm_model: str, max_len: int = 512):
        self.sp      = spm.SentencePieceProcessor(model_file=spm_model)
        self.max_len = max_len

        with open(text_file, encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> dict:
        tokens = self.sp.encode(self.lines[idx])

        # Truncate so that BOS + tokens + EOS fits in max_len
        tokens = tokens[:self.max_len - 2]
        full   = [BOS_ID] + tokens + [EOS_ID]  # length ≤ max_len

        # input: full[:-1], labels: full[1:]
        inp = full[:-1]  # length = len(full) - 1
        lbl = full[1:]

        # Pad to max_len - 1
        pad_len = (self.max_len - 1) - len(inp)
        inp = inp + [PAD_ID] * pad_len
        lbl = lbl + [-100]   * pad_len   # -100 = ignore padding in loss

        return {
            'input_ids': torch.tensor(inp, dtype=torch.long),
            'labels'   : torch.tensor(lbl, dtype=torch.long),
        }
