"""
Comparative analysis: BERT-initialized vs Random-initialized embeddings.

Loads both best checkpoints, runs greedy decoding on the same test (or valid)
examples, and produces a side-by-side comparison report.

Usage:
    python src/compare.py \
        --bert_config   configs/bert.yaml \
        --bert_ckpt     checkpoints/best_bert_model.pt \
        --random_config configs/random.yaml \
        --random_ckpt   checkpoints/best_random_model.pt \
        --split test \
        --n_examples 200
"""

import argparse
import os
import yaml
import torch
import sentencepiece as spm
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from dataset import TranslationDataset
from model import Encoder, Decoder, Seq2Seq
from metrics import compute_metrics


def load_model(config, ckpt_path, device):
    encoder = Encoder(config['vocab_size'], config['emb_dim'], config['hidden_dim'],
                      num_layers=config.get('num_layers', 2),
                      dropout=config.get('dropout', 0.3))
    decoder = Decoder(config['vocab_size'], config['emb_dim'], config['hidden_dim'],
                      num_layers=config.get('num_layers', 2),
                      dropout=config.get('dropout', 0.3))
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def decode_all(model, loader, sp, device):
    predictions = []
    references = []
    for src, tgt in tqdm(loader, desc="Decoding", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)
        output = model(src, tgt, teacher_forcing_ratio=0.0)
        pred_tokens = output.argmax(-1)
        for pred_seq, tgt_seq in zip(pred_tokens, tgt):
            predictions.append(sp.decode(pred_seq.cpu().numpy().tolist()))
            references.append(sp.decode(tgt_seq.cpu().numpy().tolist()))
    return predictions, references


def bar_comparison(labels, values, title, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=['#4C72B0', '#DD8452'], width=0.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}", ha='center', va='bottom', fontsize=11)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.2 + 1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


parser = argparse.ArgumentParser()
parser.add_argument('--bert_config',   type=str, required=True)
parser.add_argument('--bert_ckpt',     type=str, required=True)
parser.add_argument('--random_config', type=str, required=True)
parser.add_argument('--random_ckpt',   type=str, required=True)
parser.add_argument('--split',         type=str, default='test', choices=['test', 'valid'])
parser.add_argument('--n_examples',    type=int, default=500,
                    help='Number of examples for comparison (use all if 0).')
args = parser.parse_args()


with open(args.bert_config) as f:
    bert_cfg = yaml.safe_load(f)
with open(args.random_config) as f:
    rand_cfg = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sp = spm.SentencePieceProcessor()
sp.load(bert_cfg['spm_model'])


if args.split == 'test':
    src_file = bert_cfg.get('test_src', 'data/test.src')
    tgt_file = bert_cfg.get('test_tgt', 'data/test.tgt')
else:
    src_file = bert_cfg['valid_src']
    tgt_file = bert_cfg['valid_tgt']

full_dataset = TranslationDataset(src_file, tgt_file, bert_cfg['spm_model'])
n = len(full_dataset) if args.n_examples == 0 else min(args.n_examples, len(full_dataset))
dataset = Subset(full_dataset, range(n))

# Both models use the same data — batch size from random config (smaller model, larger batch)
loader = DataLoader(dataset, batch_size=rand_cfg['batch_size'], shuffle=False, num_workers=0)

print(f"Loading BERT model from {args.bert_ckpt} ...")
bert_model = load_model(bert_cfg, args.bert_ckpt, DEVICE)

print(f"Loading Random model from {args.random_ckpt} ...")
rand_model = load_model(rand_cfg, args.random_ckpt, DEVICE)

print(f"Decoding {n} examples from split='{args.split}' ...")
bert_preds, references = decode_all(bert_model, loader, sp, DEVICE)
rand_preds, _          = decode_all(rand_model, loader, sp, DEVICE)


bert_bleu, bert_chrf   = compute_metrics(bert_preds, references)
rand_bleu, rand_chrf   = compute_metrics(rand_preds, references)


out_dir = "outputs/comparison"
os.makedirs(out_dir, exist_ok=True)


bar_comparison(
    ['BERT Embeddings', 'Random Embeddings'],
    [bert_bleu, rand_bleu],
    f'BLEU (100) Comparison — {args.split}',
    'BLEU (100)',
    os.path.join(out_dir, f'bleu_comparison_{args.split}.png')
)

bar_comparison(
    ['BERT Embeddings', 'Random Embeddings'],
    [bert_chrf, rand_chrf],
    f'CHRF++ (100) Comparison — {args.split}',
    'CHRF++ (100)',
    os.path.join(out_dir, f'chrf_comparison_{args.split}.png')
)


print(f"\n{'='*60}")
print(f"{'Metric':<20} {'BERT':>12} {'Random':>12} {'Δ':>8}")
print(f"{'-'*60}")
print(f"{'BLEU (100)':<20} {bert_bleu:>12.2f} {rand_bleu:>12.2f} {bert_bleu-rand_bleu:>+8.2f}")
print(f"{'CHRF++ (100)':<20} {bert_chrf:>12.2f} {rand_chrf:>12.2f} {bert_chrf-rand_chrf:>+8.2f}")
print(f"{'='*60}\n")


with open(os.path.join(out_dir, f'metrics_{args.split}.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Split     : {args.split}\n")
    f.write(f"Examples  : {n}\n\n")
    f.write(f"{'Metric':<20} {'BERT':>12} {'Random':>12} {'Delta':>8}\n")
    f.write(f"{'-'*55}\n")
    f.write(f"{'BLEU (100)':<20} {bert_bleu:>12.2f} {rand_bleu:>12.2f} {bert_bleu-rand_bleu:>+8.2f}\n")
    f.write(f"{'CHRF++ (100)':<20} {bert_chrf:>12.2f} {rand_chrf:>12.2f} {bert_chrf-rand_chrf:>+8.2f}\n")


with open(os.path.join(out_dir, f'side_by_side_{args.split}.txt'), 'w', encoding='utf-8') as f:
    for i, (ref, bp, rp) in enumerate(zip(references, bert_preds, rand_preds)):
        f.write(f"[{i+1}]\n")
        f.write(f"REF   : {ref}\n")
        f.write(f"BERT  : {bp}\n")
        f.write(f"RANDOM: {rp}\n\n")

print(f"Comparison saved to {out_dir}/")
