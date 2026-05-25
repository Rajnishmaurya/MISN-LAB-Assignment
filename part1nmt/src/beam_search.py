import torch

EOS_ID = 2

@torch.no_grad()
def beam_decode(model, src, beam_width=5, max_len=128):

    model.eval()

    encoder_outputs, hidden, cell = model.encoder(src)

    encoder_projection = model.decoder.attention.project_encoder(encoder_outputs)

    # Each entry: (token_ids, cumulative_log_prob, hidden, cell)
    sequences = [([1], 0.0, hidden, cell)]
    completed = []

    for _ in range(max_len):

        if not sequences:
            break

        all_candidates = []

        for seq, score, h, c in sequences:

            x = torch.tensor([seq[-1]]).to(src.device)

            output, h_new, c_new, _ = model.decoder(
                x,
                h,
                c,
                encoder_outputs,
                encoder_projection
            )

            log_probs = torch.log_softmax(output, dim=-1)

            topk = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                token = topk.indices[0][i].item()
                token_score = topk.values[0][i].item()

                new_seq = seq + [token]
                new_score = score + token_score

                if token == EOS_ID:
                    length_penalty = len(new_seq) ** 0.6
                    completed.append((new_seq, new_score / length_penalty))
                else:
                    all_candidates.append((new_seq, new_score, h_new, c_new))

        if not all_candidates:
            break

        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]

    if completed:
        best = max(completed, key=lambda x: x[1])
        return best[0]

    return sequences[0][0]
