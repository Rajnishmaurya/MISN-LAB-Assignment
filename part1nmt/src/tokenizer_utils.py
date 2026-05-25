import sentencepiece as spm

def train_sentencepiece(input_file, model_prefix, vocab_size=16000):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type='bpe',
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3
    )
