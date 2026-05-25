import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("spm.model")

print(sp.encode("<hi2mr> मेरा नाम राज है", out_type=str))
