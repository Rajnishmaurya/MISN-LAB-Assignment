from transformers import AutoModel, AutoTokenizer

# Hindi BERT
hindi_model = AutoModel.from_pretrained(
    "l3cube-pune/hindi-bert-v2"
)

hindi_tokenizer = AutoTokenizer.from_pretrained(
    "l3cube-pune/hindi-bert-v2"
)

hindi_model.save_pretrained(
    "pretrained_models/hindi-bert-v2"
)

hindi_tokenizer.save_pretrained(
    "pretrained_models/hindi-bert-v2"
)

print("Hindi BERT downloaded.")


# Marathi BERT
marathi_model = AutoModel.from_pretrained(
    "l3cube-pune/marathi-bert-v2"
)

marathi_tokenizer = AutoTokenizer.from_pretrained(
    "l3cube-pune/marathi-bert-v2"
)

marathi_model.save_pretrained(
    "pretrained_models/marathi-bert-v2"
)

marathi_tokenizer.save_pretrained(
    "pretrained_models/marathi-bert-v2"
)

print("Marathi BERT downloaded.")