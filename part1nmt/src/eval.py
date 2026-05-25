"""
Standalone evaluation script.

Usage:
    # Greedy decoding on test set
    python src/eval.py --config configs/bert.yaml --checkpoint checkpoints/best_bert_model.pt

    # Beam search on validation set
    python src/eval.py --config configs/random.yaml --checkpoint checkpoints/best_random_model.pt \
        --split valid --beam_width 5
"""

import os
import argparse
import yaml
import torch
import sentencepiece as spm

from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import TranslationDataset
from model import Encoder, Decoder, Seq2Seq
from metrics import compute_metrics
from beam_search import beam_decode


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--split', type=str, default='test', choices=['test', 'valid'])
parser.add_argument('--beam_width', type=int, default=1,
                    help='Beam width. 1 = greedy decoding.')
parser.add_argument('--max_examples', type=int, default=None,
                    help='Limit number of examples (for quick debugging).')
args = parser.parse_args()


with open(args.config) as f:
    config = yaml.safe_load(f)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
print(f"Evaluating split : {args.split}")
print(f"Checkpoint       : {args.checkpoint}")
print(f"Beam width       : {args.beam_width}")


sp = spm.SentencePieceProcessor()
sp.load(config['spm_model'])


if args.split == 'test':
    src_file = config.get('test_src', 'data/test.src')
    tgt_file = config.get('test_tgt', 'data/test.tgt')
else:
    src_file = config['valid_src']
    tgt_file = config['valid_tgt']


dataset = TranslationDataset(src_file, tgt_file, config['spm_model'])

if args.max_examples is not None:
    from torch.utils.data import Subset
    dataset = Subset(dataset, range(min(args.max_examples, len(dataset))))

loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)


encoder = Encoder(config['vocab_size'], config['emb_dim'], config['hidden_dim'],
                  num_layers=config.get('num_layers', 2),
                  dropout=config.get('dropout', 0.3))
decoder = Decoder(config['vocab_size'], config['emb_dim'], config['hidden_dim'],
                  num_layers=config.get('num_layers', 2),
                  dropout=config.get('dropout', 0.3))
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
model.eval()
print(f"Loaded checkpoint from {args.checkpoint}")


criterion = torch.nn.CrossEntropyLoss(ignore_index=0)


predictions = []
references = []
total_loss = 0.0

with torch.no_grad():
    for src, tgt in tqdm(loader, desc="Evaluating"):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        if args.beam_width > 1:
            # Beam search — decode one sentence at a time
            for i in range(src.shape[0]):
                single_src = src[i].unsqueeze(0)
                pred_ids = beam_decode(model, single_src, beam_width=args.beam_width)
                predictions.append(sp.decode(pred_ids))

            tgt_ref = tgt
        else:
            # Greedy decoding
            output = model(src, tgt, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            loss = criterion(
                output[:, 1:].reshape(-1, output_dim),
                tgt[:, 1:].reshape(-1)
            )
            total_loss += loss.item()

            pred_tokens = output.argmax(-1)
            for pred_seq in pred_tokens:
                predictions.append(sp.decode(pred_seq.cpu().numpy().tolist()))

            tgt_ref = tgt

        for tgt_seq in tgt_ref:
            references.append(sp.decode(tgt_seq.cpu().numpy().tolist()))


bleu, chrf = compute_metrics(predictions, references)

avg_loss = total_loss / len(loader) if args.beam_width == 1 else float('nan')

print(f"\n{'='*50}")
print(f"Split      : {args.split}")
print(f"Examples   : {len(predictions)}")
if args.beam_width == 1:
    print(f"Loss       : {avg_loss:.4f}")
print(f"BLEU (100) : {bleu:.2f}")
print(f"CHRF++ (100): {chrf:.2f}")
print(f"{'='*50}\n")


exp_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
decode_tag = f"beam{args.beam_width}" if args.beam_width > 1 else "greedy"
out_dir = f"outputs/eval/{exp_name}_{args.split}_{decode_tag}"
os.makedirs(out_dir, exist_ok=True)

# Save metrics
with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"Split      : {args.split}\n")
    f.write(f"Checkpoint : {args.checkpoint}\n")
    f.write(f"Beam width : {args.beam_width}\n")
    f.write(f"Examples   : {len(predictions)}\n")
    if args.beam_width == 1:
        f.write(f"Loss       : {avg_loss:.4f}\n")
    f.write(f"BLEU (100) : {bleu:.2f}\n")
    f.write(f"CHRF++ (100): {chrf:.2f}\n")

# Save predictions alongside references
with open(os.path.join(out_dir, "predictions.txt"), "w", encoding="utf-8") as f:
    for pred, ref in zip(predictions, references):
        f.write(f"REF : {ref}\n")
        f.write(f"PRED: {pred}\n")
        f.write("\n")

print(f"Results saved to {out_dir}/")
