"""
Standalone evaluation script for the translation model.

Supports greedy decoding and beam search.

Usage:
    # Greedy
    python src/eval_translation.py --config configs/translation.yaml \
        --checkpoint checkpoints/translation_best.pt --split test

    # Beam search
    python src/eval_translation.py --config configs/translation.yaml \
        --checkpoint checkpoints/translation_best.pt --split test --beam_width 5
"""

import os, sys, argparse
import yaml, torch
import sentencepiece as spm
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translation_model   import TranslationModel
from translation_dataset import TranslationDataset
from metrics             import compute_metrics
from beam_search         import beam_decode

parser = argparse.ArgumentParser()
parser.add_argument('--config',      required=True)
parser.add_argument('--checkpoint',  required=True)
parser.add_argument('--split',       default='test', choices=['test', 'valid'])
parser.add_argument('--beam_width',  type=int, default=1)
parser.add_argument('--max_examples',type=int, default=None)
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sp = spm.SentencePieceProcessor(model_file=config['spm_model'])

src_file = config['test_src']  if args.split == 'test' else config['valid_src']
tgt_file = config['test_tgt']  if args.split == 'test' else config['valid_tgt']

dataset = TranslationDataset(src_file, tgt_file, config['spm_model'],
                             config.get('max_seq_len', 128))
if args.max_examples:
    from torch.utils.data import Subset
    dataset = Subset(dataset, range(min(args.max_examples, len(dataset))))

loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

model = TranslationModel(
    vocab_size             = config['vocab_size'],
    enc_hidden_dim         = config.get('enc_hidden_dim', 768),
    enc_num_layers         = config.get('enc_num_layers', 12),
    enc_num_heads          = config.get('enc_num_heads', 12),
    enc_num_kv_heads       = config.get('enc_num_kv_heads',  4),
    enc_ffn_dim            = config.get('enc_ffn_dim', 3072),
    dec_hidden_dim         = config.get('dec_hidden_dim', 768),
    dec_num_layers         = config.get('dec_num_layers', 16),
    dec_num_heads          = config.get('dec_num_heads', 12),
    dec_num_kv_heads       = config.get('dec_num_kv_heads', 4),
    dec_num_kv_heads_cross = config.get('dec_num_kv_heads_cross', 4),
    dec_ffn_dim            = config.get('dec_ffn_dim', 3072),
    max_seq_len            = config.get('max_seq_len', 128),
    dropout                = 0.0,
    pad_id                 = config.get('pad_id', 0),
).to(DEVICE)
model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
model.eval()
print(f"Loaded checkpoint: {args.checkpoint}")

predictions, references = [], []
total_loss = 0.0
crit = torch.nn.CrossEntropyLoss(ignore_index=0)

with torch.no_grad():
    for src, tgt in tqdm(loader, desc='Evaluating'):
        src = src.to(DEVICE); tgt = tgt.to(DEVICE)

        if args.beam_width > 1:
            for i in range(src.shape[0]):
                pred_ids = beam_decode(model, src[i:i+1], beam_width=args.beam_width)
                predictions.append(sp.decode(pred_ids))
        else:
            logits = model(src, tgt)
            loss   = crit(logits[:, 1:].reshape(-1, config['vocab_size']), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            for tok in logits.argmax(-1):
                predictions.append(sp.decode(tok.cpu().numpy().tolist()))

        for tok in tgt:
            references.append(sp.decode(tok.cpu().numpy().tolist()))

bleu, chrf = compute_metrics(predictions, references)
avg_loss   = total_loss / len(loader) if args.beam_width == 1 else float('nan')

print(f"\n{'='*55}")
print(f"  Split      : {args.split}")
print(f"  Beam width : {args.beam_width}")
print(f"  Examples   : {len(predictions)}")
if args.beam_width == 1:
    print(f"  Loss       : {avg_loss:.4f}")
print(f"  BLEU (100) : {bleu:.2f}")
print(f"  CHRF++(100): {chrf:.2f}")
print(f"{'='*55}")

ckpt_name  = os.path.splitext(os.path.basename(args.checkpoint))[0]
decode_tag = f"beam{args.beam_width}" if args.beam_width > 1 else "greedy"
out_dir    = f"outputs/eval/{ckpt_name}_{args.split}_{decode_tag}"
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
    f.write(f"Split      : {args.split}\nCheckpoint : {args.checkpoint}\n")
    f.write(f"Beam width : {args.beam_width}\nExamples   : {len(predictions)}\n")
    if args.beam_width == 1: f.write(f"Loss       : {avg_loss:.4f}\n")
    f.write(f"BLEU (100) : {bleu:.2f}\nCHRF++(100): {chrf:.2f}\n")

with open(os.path.join(out_dir, "predictions.txt"), "w", encoding="utf-8") as f:
    for p, r in zip(predictions, references):
        f.write(f"REF : {r}\nPRED: {p}\n\n")

print(f"Results saved to {out_dir}/")
