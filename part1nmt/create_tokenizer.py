from src.tokenizer_utils import train_sentencepiece

train_sentencepiece(
    input_file="final_combined.txt",
    model_prefix="spm",
    vocab_size=16000
)

print("Multilingual tokenizer trained successfully.")
