"""
Train the SentencePiece tokenizer for Part 2.

This is a SEPARATE tokenizer from Part 1 because Part 2 BERT pretraining
requires a [MASK] special token.  The [MASK] token is added as a
user-defined symbol, which SentencePiece assigns to ID 4 — immediately
after the built-in specials (PAD=0, BOS=1, EOS=2, UNK=3).

Special token IDs after training:
  PAD   = 0   (pad_id)
  BOS   = 1   (bos_id)
  EOS   = 2   (eos_id)
  UNK   = 3   (unk_id)
  [MASK]= 4   (first user_defined_symbol)

The vocab_size is kept at 16000 so embeddings are interchangeable with
Part 1 (the [MASK] token replaces one low-frequency BPE piece).

Input : data/pretrain_corpus.txt  (built by prepare_pretrain_data.py)
Output: spm_part2.model, spm_part2.vocab
"""

import sentencepiece as spm
from pathlib import Path

corpus = "data/pretrain_corpus.txt"
assert Path(corpus).exists(), \
    f"Corpus not found: {corpus}\nRun prepare_pretrain_data.py first."

print(f"Training SentencePiece tokenizer (Part 2) on: {corpus}")

spm.SentencePieceTrainer.train(
    input             = corpus,
    model_prefix      = "spm_part2",
    vocab_size        = 16000,
    character_coverage= 1.0,
    model_type        = 'bpe',
    bos_id            = 1,
    eos_id            = 2,
    pad_id            = 0,
    unk_id            = 3,
    user_defined_symbols = ['[MASK]'],  # → ID 4 (first after specials)
)

# Verify the MASK token is at ID 4
sp = spm.SentencePieceProcessor(model_file="spm_part2.model")
assert sp.piece_to_id('[MASK]') == 4, \
    f"[MASK] got ID {sp.piece_to_id('[MASK]')} instead of 4!"

print("Tokenizer trained successfully: spm_part2.model")
print(f"  Vocab size : {sp.vocab_size()}")
print(f"  PAD  ID    : {sp.piece_to_id('<pad>')}")
print(f"  BOS  ID    : {sp.piece_to_id('<s>')}")
print(f"  EOS  ID    : {sp.piece_to_id('</s>')}")
print(f"  UNK  ID    : {sp.piece_to_id('<unk>')}")
print(f"  MASK ID    : {sp.piece_to_id('[MASK]')}")
