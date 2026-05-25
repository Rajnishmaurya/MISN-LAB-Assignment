import torch
from tqdm import tqdm

from metrics import compute_metrics

_eval_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)


@torch.no_grad()
def evaluate_model(model, dataloader, sp, device):
    """
    Greedy evaluation (teacher_forcing_ratio=0).

    Returns avg_loss, bleu (0-100), chrf++ (0-100),
    predictions (list[str]), references (list[str]).
    """
    model.eval()

    total_loss = 0.0
    predictions = []
    references  = []

    for src, tgt in tqdm(dataloader, desc="Evaluating", leave=False):

        src = src.to(device)
        tgt = tgt.to(device)

        output = model(src, tgt, teacher_forcing_ratio=0.0)

        output_dim = output.shape[-1]

        loss = _eval_criterion(
            output[:, 1:].reshape(-1, output_dim),
            tgt[:, 1:].reshape(-1)
        )
        total_loss += loss.item()

        pred_tokens = output.argmax(-1)

        for pred_seq, tgt_seq in zip(pred_tokens, tgt):
            predictions.append(sp.decode(pred_seq.cpu().numpy().tolist()))
            references.append(sp.decode(tgt_seq.cpu().numpy().tolist()))

    avg_loss = total_loss / len(dataloader)
    bleu, chrf = compute_metrics(predictions, references)

    return avg_loss, bleu, chrf, predictions, references
