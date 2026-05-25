import torch
from torch.utils.data import Dataset
import sentencepiece as spm

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, spm_model, max_len=128):
        self.src_lines = open(src_file, encoding='utf-8').read().splitlines()
        self.tgt_lines = open(tgt_file, encoding='utf-8').read().splitlines()
        self.sp = spm.SentencePieceProcessor(model_file=spm_model)
        self.max_len = max_len

    def __len__(self):
        return len(self.src_lines)

    def encode(self, text):
        ids = [1] + self.sp.encode(text) + [2]
        ids = ids[:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, idx):
        src = torch.tensor(self.encode(self.src_lines[idx]))
        tgt = torch.tensor(self.encode(self.tgt_lines[idx]))
        return src, tgt
