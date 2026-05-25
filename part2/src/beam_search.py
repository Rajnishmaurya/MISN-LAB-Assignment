"""
Beam search decoder for the TranslationModel.

Works with the encoder-decoder architecture: encodes source once,
then autoregressively expands beam hypotheses using the decoder.
"""

import torch

BOS_ID = 1
EOS_ID = 2


@torch.no_grad()
def beam_decode(
    model,
    src: torch.Tensor,       # (1, S) single source sentence
    beam_width: int = 5,
    max_len: int = 128,
    length_penalty: float = 0.6,
) -> list[int]:
    """
    Beam search decoding for TranslationModel.

    Args:
        model       : TranslationModel (in eval mode)
        src         : (1, S) source token IDs
        beam_width  : number of beams
        max_len     : maximum decoding steps
        length_penalty: Google NMT length penalty exponent (0 = no penalty)

    Returns:
        list[int] — best hypothesis token IDs (without BOS)
    """
    device = src.device
    model.eval()

    # Encode source once
    enc_out, enc_mask = model.encode(src)  # (1, S, C), (1, S)

    # Expand to beam_width copies
    enc_out  = enc_out.expand(beam_width, -1, -1)
    enc_mask = enc_mask.expand(beam_width, -1)

    # Each beam: (token_ids, cumulative_log_prob)
    sequences = [([BOS_ID], 0.0)]
    completed = []

    for _ in range(max_len):
        if not sequences:
            break

        # Build batch from current sequences
        max_t = max(len(s) for s, _ in sequences)
        pad = torch.full((len(sequences), max_t), 0, dtype=torch.long, device=device)
        for b, (seq, _) in enumerate(sequences):
            pad[b, :len(seq)] = torch.tensor(seq, device=device)

        # Expand enc to current beam count (may be < beam_width early on)
        cur_enc  = enc_out[:len(sequences)]
        cur_mask = enc_mask[:len(sequences)]

        logits = model.decode_step(pad, cur_enc, cur_mask)  # (B, T, V)
        # Take last position
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # (B, V)

        all_candidates = []
        for b, (seq, score) in enumerate(sequences):
            topk_scores, topk_ids = log_probs[b].topk(beam_width)
            for score_i, tok_i in zip(topk_scores.tolist(), topk_ids.tolist()):
                new_seq   = seq + [tok_i]
                new_score = score + score_i
                if tok_i == EOS_ID:
                    pen = len(new_seq) ** length_penalty
                    completed.append((new_seq, new_score / pen))
                else:
                    all_candidates.append((new_seq, new_score))

        if not all_candidates:
            break

        # Keep top beam_width candidates
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        sequences = all_candidates[:beam_width]

    if completed:
        best = max(completed, key=lambda x: x[1])
        return best[0][1:]  # strip BOS

    return sequences[0][0][1:]  # best incomplete, strip BOS
