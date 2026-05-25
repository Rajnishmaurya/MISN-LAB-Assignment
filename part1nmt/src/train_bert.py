import os
import math
import time
import json
import random
import argparse

import yaml
import torch
import numpy as np
import pandas as pd
import sentencepiece as spm

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

from dataset import TranslationDataset
from model import Encoder, Decoder, Seq2Seq
from evaluate import evaluate_model
from plotting import plot_metric, plot_lr
from bert_embeddings import initialize_with_bert
from metrics import compute_metrics


# ── Args & config ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = config.get('seed', 42)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Device & AMP dtype ────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_dtype_str = config.get('amp_dtype', 'bfloat16')
AMP_DTYPE  = torch.bfloat16 if _dtype_str == 'bfloat16' else torch.float16
USE_SCALER = (AMP_DTYPE == torch.float16)   # GradScaler only needed for fp16

# TF32 gives free speedup on A100 with negligible precision loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

print("=" * 60)
print("  BERT Embeddings — NMT Training")
print("=" * 60)
print(f"  Device    : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  AMP dtype : {_dtype_str}")
print(f"  Seed      : {SEED}")
print("=" * 60)

# ── Output dirs ───────────────────────────────────────────────────────────────
os.makedirs("checkpoints",   exist_ok=True)
os.makedirs("outputs/bert",  exist_ok=True)
os.makedirs("logs",          exist_ok=True)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
sp = spm.SentencePieceProcessor()
sp.load(config['spm_model'])

# ── Datasets & loaders ────────────────────────────────────────────────────────
num_workers = config.get('num_workers', 4)

train_dataset = TranslationDataset(config['train_src'], config['train_tgt'], config['spm_model'])
valid_dataset = TranslationDataset(config['valid_src'], config['valid_tgt'], config['spm_model'])
test_dataset  = TranslationDataset(config['test_src'],  config['test_tgt'],  config['spm_model'])

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                          shuffle=True,  num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'],
                          shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'],
                          shuffle=False, num_workers=num_workers, pin_memory=True)

print(f"\n  Train : {len(train_dataset):,} pairs  ({len(train_loader):,} batches)")
print(f"  Valid : {len(valid_dataset):,} pairs  ({len(valid_loader):,} batches)")
print(f"  Test  : {len(test_dataset):,}  pairs  ({len(test_loader):,}  batches)\n")

# ── Model ─────────────────────────────────────────────────────────────────────
_dropout = config.get('dropout', 0.3)
encoder = Encoder(config['vocab_size'], config['emb_dim'], config['hidden_dim'],
                  num_layers=config.get('num_layers', 2), dropout=_dropout)
decoder = Decoder(config['vocab_size'], config['emb_dim'], config['hidden_dim'],
                  num_layers=config.get('num_layers', 2), dropout=_dropout)

hindi_bert_path   = config.get('hindi_bert_path',   'pretrained_models/hindi-bert-v2')
marathi_bert_path = config.get('marathi_bert_path',  'pretrained_models/marathi-bert-v2')

print("  Initializing encoder embeddings with Hindi BERT ...")
initialize_with_bert(encoder.embedding, hindi_bert_path,   spm_model_path=config['spm_model'])

print("  Initializing decoder embeddings with Marathi BERT ...")
initialize_with_bert(decoder.embedding, marathi_bert_path, spm_model_path=config['spm_model'])

model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Total params     : {total_params:,}")
print(f"  Trainable params : {trainable_params:,}\n")

# ── Loss, optimizer, scheduler ────────────────────────────────────────────────
criterion = torch.nn.CrossEntropyLoss(
    ignore_index=0,
    label_smoothing=config.get('label_smoothing', 0.0)
)

optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-2)

accum_steps   = config.get('accumulation_steps', 1)
total_steps   = math.ceil(len(train_loader) / accum_steps) * config['epochs']
warmup_steps  = math.ceil(len(train_loader) / accum_steps) * config.get('warmup_epochs', 1)
min_lr_ratio  = config.get('min_lr', 1e-6) / config['lr']

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = LambdaLR(optimizer, lr_lambda)
scaler    = GradScaler('cuda') if USE_SCALER else None

# ── Training state ────────────────────────────────────────────────────────────
train_losses, val_losses         = [], []
train_bleu_scores, val_bleu_scores = [], []
train_chrf_scores, val_chrf_scores = [], []
lr_history                       = []
epoch_times                      = []
best_val_loss                    = float('inf')
best_epoch                       = 0
global_step                      = 0
training_start                   = time.time()

print("=" * 60)
print("  Starting training")
print("=" * 60)

# ── Epoch loop ────────────────────────────────────────────────────────────────
for epoch in range(config['epochs']):
    epoch_start = time.time()
    model.train()

    total_loss  = 0.0
    train_preds = []
    train_refs  = []

    optimizer.zero_grad()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

    for step, (src, tgt) in enumerate(progress_bar):
        src = src.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)

        with autocast('cuda', dtype=AMP_DTYPE):
            output        = model(src, tgt)
            output_dim    = output.shape[-1]
            output_flat   = output[:, 1:].reshape(-1, output_dim)
            tgt_flat      = tgt[:, 1:].reshape(-1)
            loss          = criterion(output_flat, tgt_flat) / accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        total_loss += loss.item() * accum_steps

        # Collect first 1000 training samples for BLEU estimate
        if len(train_preds) < 1000:
            pred_tokens = output.argmax(-1).detach()
            for pred_seq, tgt_seq in zip(pred_tokens, tgt):
                train_preds.append(sp.decode(pred_seq.cpu().numpy().tolist()))
                train_refs.append(sp.decode(tgt_seq.cpu().numpy().tolist()))

        progress_bar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")

    # ── Epoch metrics ──────────────────────────────────────────────────────
    train_loss         = total_loss / len(train_loader)
    train_bleu, train_chrf = compute_metrics(train_preds, train_refs)

    val_loss, val_bleu, val_chrf, val_preds, val_refs = evaluate_model(
        model, valid_loader, sp, DEVICE
    )

    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    current_lr = scheduler.get_last_lr()[0]
    lr_history.append(current_lr)

    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    torch.cuda.reset_peak_memory_stats()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_bleu_scores.append(train_bleu)
    val_bleu_scores.append(val_bleu)
    train_chrf_scores.append(train_chrf)
    val_chrf_scores.append(val_chrf)

    print(f"\n  Epoch {epoch+1}/{config['epochs']}  |  time: {epoch_time/60:.1f} min  |  "
          f"GPU mem: {gpu_mem_gb:.2f} GB  |  LR: {current_lr:.2e}")
    print(f"    Train  — Loss: {train_loss:.4f}  BLEU: {train_bleu:.2f}  CHRF++: {train_chrf:.2f}")
    print(f"    Valid  — Loss: {val_loss:.4f}    BLEU: {val_bleu:.2f}    CHRF++: {val_chrf:.2f}")

    # ── Checkpoint ────────────────────────────────────────────────────────
    torch.save(model.state_dict(), f"checkpoints/bert_epoch_{epoch+1}.pt")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch + 1
        torch.save(model.state_dict(), "checkpoints/best_bert_model.pt")
        # Save best-epoch validation predictions (with references)
        with open("outputs/bert/best_val_predictions.txt", "w", encoding="utf-8") as f:
            for pred, ref in zip(val_preds[:200], val_refs[:200]):
                f.write(f"REF : {ref}\n")
                f.write(f"PRED: {pred}\n\n")
        print(f"    ✓  Best model saved (epoch {best_epoch})")

# ── Post-training: save all metrics ──────────────────────────────────────────
total_time_h = (time.time() - training_start) / 3600

metrics_df = pd.DataFrame({
    "epoch":      list(range(1, config['epochs'] + 1)),
    "lr":         lr_history,
    "train_loss": train_losses,
    "val_loss":   val_losses,
    "train_bleu": train_bleu_scores,
    "val_bleu":   val_bleu_scores,
    "train_chrf": train_chrf_scores,
    "val_chrf":   val_chrf_scores,
    "epoch_time_s": epoch_times,
})
metrics_df.to_csv("outputs/bert/metrics.csv", index=False)

# ── Plots ─────────────────────────────────────────────────────────────────────
plot_metric(train_losses,      val_losses,      "Loss",        "outputs/bert/loss_plot.png",
            title="Train vs Validation Loss")
plot_metric(train_bleu_scores, val_bleu_scores, "BLEU (100)",  "outputs/bert/bleu_plot.png",
            title="Train vs Validation BLEU (100)")
plot_metric(train_chrf_scores, val_chrf_scores, "CHRF++ (100)","outputs/bert/chrf_plot.png",
            title="Train vs Validation CHRF++ (100)")
plot_lr(lr_history, "outputs/bert/lr_plot.png")

# ── Test evaluation with best model ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  Running test evaluation with best model ...")
print("=" * 60)

model.load_state_dict(torch.load("checkpoints/best_bert_model.pt", map_location=DEVICE))
test_loss, test_bleu, test_chrf, test_preds, test_refs = evaluate_model(
    model, test_loader, sp, DEVICE
)

print(f"\n  TEST RESULTS (best model from epoch {best_epoch})")
print(f"  Loss     : {test_loss:.4f}")
print(f"  BLEU (100): {test_bleu:.2f}")
print(f"  CHRF++ (100): {test_chrf:.2f}")

with open("outputs/bert/test_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Best epoch   : {best_epoch}\n")
    f.write(f"Test Loss    : {test_loss:.4f}\n")
    f.write(f"Test BLEU (100) : {test_bleu:.2f}\n")
    f.write(f"Test CHRF++ (100): {test_chrf:.2f}\n")

with open("outputs/bert/test_predictions.txt", "w", encoding="utf-8") as f:
    for pred, ref in zip(test_preds[:200], test_refs[:200]):
        f.write(f"REF : {ref}\n")
        f.write(f"PRED: {pred}\n\n")

# ── JSON training summary (for report) ───────────────────────────────────────
summary = {
    "experiment":  "bert_embeddings",
    "config":       config,
    "model": {
        "total_params":     total_params,
        "trainable_params": trainable_params,
    },
    "gpu": {
        "name":  torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                   if torch.cuda.is_available() else 0,
    },
    "training": {
        "total_time_hours":  round(total_time_h, 2),
        "avg_epoch_time_min": round(sum(epoch_times) / len(epoch_times) / 60, 2),
        "best_epoch":         best_epoch,
        "best_val_loss":      round(best_val_loss, 4),
        "best_val_bleu":      round(val_bleu_scores[best_epoch - 1], 2),
        "best_val_chrf":      round(val_chrf_scores[best_epoch - 1], 2),
    },
    "test": {
        "loss":  round(test_loss, 4),
        "bleu":  round(test_bleu, 2),
        "chrf":  round(test_chrf, 2),
    },
    "per_epoch": metrics_df.round(4).to_dict(orient="records"),
}

with open("outputs/bert/training_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print(f"  BERT training complete  |  total time: {total_time_h:.2f} h")
print(f"  All results saved to outputs/bert/")
print("=" * 60)
