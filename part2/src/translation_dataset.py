"""
Translation Dataset for Part 2 fine-tuning.

Uses the same bidirectional data format as Part 1:
  Source: "<hi2mr> <hindi sentence>"  or  "<mr2hi> <marathi sentence>"
  Target: corresponding translation

Reuses the pre-trained SentencePiece model (spm.model) from Part 1 data pipeline.
The tokenizer was trained on the full bilingual corpus with direction tokens.
"""

import torch
from torch.utils.data import Dataset
import sentencepiece as spm

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


class TranslationDataset(Dataset):
    """
    Each sample:
      src : (max_len,)  source token IDs  [BOS, direction_token, tokens..., EOS, PAD...]
      tgt : (max_len,)  target token IDs  [BOS, tokens..., EOS, PAD...]

    Args:
        src_file  : path to source file (one sentence per line, with direction token)
        tgt_file  : path to target file
        spm_model : path to SentencePiece .model file
        max_len   : maximum sequence length (padded/truncated to this length)
    """

    def __init__(self, src_file: str, tgt_file: str, spm_model: str, max_len: int = 128):
        self.sp      = spm.SentencePieceProcessor(model_file=spm_model)
        self.max_len = max_len

        with open(src_file, encoding='utf-8') as f:
            self.src_lines = f.read().splitlines()
        with open(tgt_file, encoding='utf-8') as f:
            self.tgt_lines = f.read().splitlines()

        assert len(self.src_lines) == len(self.tgt_lines), \
            f"Source ({len(self.src_lines)}) and target ({len(self.tgt_lines)}) lengths differ"

    def __len__(self) -> int:
        return len(self.src_lines)

    def _encode(self, text: str) -> list[int]:
        ids = [BOS_ID] + self.sp.encode(text) + [EOS_ID]
        ids = ids[:self.max_len]
        ids += [PAD_ID] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src = torch.tensor(self._encode(self.src_lines[idx]), dtype=torch.long)
        tgt = torch.tensor(self._encode(self.tgt_lines[idx]), dtype=torch.long)
        return src, tgt
