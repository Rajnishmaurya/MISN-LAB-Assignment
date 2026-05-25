from transformers import AutoModel, AutoTokenizer
import sentencepiece as spm_lib
import torch


@torch.no_grad()
def initialize_with_bert(embedding_layer, bert_model_path, spm_model_path=None):
    """
    Initialize embedding_layer weights from a pretrained BERT model.

    If spm_model_path is provided, maps each SPM BPE token to its closest
    BERT WordPiece token(s) so the embedding indices are semantically aligned.
    Without spm_model_path, falls back to a direct index copy (wrong mapping,
    only kept for debugging purposes).
    """
    bert = AutoModel.from_pretrained(bert_model_path)
    bert_emb = bert.embeddings.word_embeddings.weight  # [bert_vocab, bert_dim]
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)

    vocab_size = embedding_layer.weight.shape[0]   # SPM vocab size
    emb_dim = embedding_layer.weight.shape[1]       # our hidden dim

    if spm_model_path is None:
        # Fallback: direct index copy — indices don't match, use only if SPM not available
        for i in range(min(vocab_size, bert_emb.shape[0])):
            embedding_layer.weight[i] = bert_emb[i][:emb_dim]
        print("  Warning: BERT init used direct index copy — vocabularies not aligned.")
        return embedding_layer

    sp = spm_lib.SentencePieceProcessor(model_file=spm_model_path)

    # Special token IDs in our SPM model
    SPECIAL_IDS = {0, 1, 2, 3}  # PAD, BOS, EOS, UNK

    n_aligned = 0
    n_fallback = 0

    for i in range(vocab_size):
        if i in SPECIAL_IDS:
            continue

        piece = sp.id_to_piece(i)
        # SPM uses ▁ (U+2581) as a word-boundary prefix; strip it for BERT lookup
        text = piece.replace('▁', ' ').strip()
        if not text:
            continue

        bert_ids = bert_tokenizer.encode(text, add_special_tokens=False)

        if bert_ids:
            # Average over multiple BERT subwords if SPM piece spans several
            vecs = bert_emb[bert_ids]           # [n_bert_tokens, bert_dim]
            embedding_layer.weight[i] = vecs.mean(0)[:emb_dim]
            n_aligned += 1
        else:
            n_fallback += 1

    total = vocab_size - len(SPECIAL_IDS)
    print(f"  Aligned {n_aligned}/{total} tokens with BERT vocabulary "
          f"({n_fallback} tokens had no BERT match, kept random init).")

    return embedding_layer
