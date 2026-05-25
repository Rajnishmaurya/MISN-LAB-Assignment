"""
Fine-tune the encoder-decoder translation model.

Loads pretrained BERT (encoder) and GPT (decoder backbone) weights,
adds randomly initialized cross-attention layers, then fine-tunes on
the Hindi-Marathi translation task.

Usage:
    python src/finetune_translation.py --config configs/translation.yaml
"""

import os, sys, math, time, json, random, argparse
import yaml, torch, numpy as np, pandas as pd
import sentencepiece as spm
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translation_model   import TranslationModel
from translation_dataset import TranslationDataset
from metrics             import compute_metrics
from plotting            import plot_metric, plot_lr

# ── Args & config ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.safe_load(f)

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = config.get('seed', 42)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Device / AMP ─────────────────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_dtype_str = config.get('amp_dtype', 'bfloat16')
AMP_DTYPE  = torch.bfloat16 if _dtype_str == 'bfloat16' else torch.float16
USE_SCALER = (AMP_DTYPE == torch.float16)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

print("=" * 60)
print("  Translation Fine-tuning")
print("=" * 60)
print(f"  Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Output dirs ───────────────────────────────────────────────────────────────
os.makedirs('checkpoints',          exist_ok=True)
os.makedirs('outputs/translation',  exist_ok=True)
os.makedirs('logs',                 exist_ok=True)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
sp = spm.SentencePieceProcessor(model_file=config['spm_model'])

# ── Datasets ─────────────────────────────────────────────────────────────────
max_len     = config.get('max_seq_len', 128)
num_workers = config.get('num_workers', 4)

train_ds = TranslationDataset(config['train_src'], config['train_tgt'], config['spm_model'], max_len)
valid_ds = TranslationDataset(config['valid_src'], config['valid_tgt'], config['spm_model'], max_len)
test_ds  = TranslationDataset(config['test_src'],  config['test_tgt'],  config['spm_model'], max_len)

train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                          shuffle=True,  num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=config['batch_size'],
                          shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'],
                          shuffle=False, num_workers=num_workers, pin_memory=True)

print(f"\n  Train: {len(train_ds):,} | Valid: {len(valid_ds):,} | Test: {len(test_ds):,}\n")

# ── Model ─────────────────────────────────────────────────────────────────────
model = TranslationModel(
    vocab_size              = config['vocab_size'],
    enc_hidden_dim          = config.get('enc_hidden_dim',       768),
    enc_num_layers          = config.get('enc_num_layers',        12),
    enc_num_heads           = config.get('enc_num_heads',         12),
    enc_num_kv_heads        = config.get('enc_num_kv_heads',       4),
    enc_ffn_dim             = config.get('enc_ffn_dim',         3072),
    dec_hidden_dim          = config.get('dec_hidden_dim',       768),
    dec_num_layers          = config.get('dec_num_layers',        16),
    dec_num_heads           = config.get('dec_num_heads',         12),
    dec_num_kv_heads        = config.get('dec_num_kv_heads',       4),
    dec_num_kv_heads_cross  = config.get('dec_num_kv_heads_cross', 4),
    dec_ffn_dim             = config.get('dec_ffn_dim',         3072),
    max_seq_len             = max_len,
    dropout                 = config.get('dropout',              0.1),
    pad_id                  = config.get('pad_id',                 0),
).to(DEVICE)

# Load pretrained weights
model.load_pretrained(
    bert_checkpoint = config['bert_checkpoint'],
    gpt_checkpoint  = config['gpt_checkpoint'],
    device          = DEVICE,
)

param_counts = model.count_parameters()
print(f"  Total params     : {param_counts['total']:,}")
print(f"  Encoder params   : {param_counts['encoder']:,}")
print(f"  Decoder params   : {param_counts['decoder']:,}")
print(f"  Cross-attn (new) : {param_counts['cross_attn_new']:,}\n")

# ── Loss / Optimizer / Scheduler ─────────────────────────────────────────────
criterion   = torch.nn.CrossEntropyLoss(
    ignore_index=0,
    label_smoothing=config.get('label_smoothing', 0.1)
)
_eval_crit  = torch.nn.CrossEntropyLoss(ignore_index=0)

optimizer   = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
accum_steps = config.get('accumulation_steps', 1)
total_steps = math.ceil(len(train_loader) / accum_steps) * config['epochs']
warmup_steps= int(total_steps * config.get('warmup_ratio', 0.05))
min_lr_ratio= config.get('min_lr', 1e-6) / config['lr']

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = LambdaLR(optimizer, lr_lambda)
scaler    = GradScaler('cuda') if USE_SCALER else None

# ── Training helpers ──────────────────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    total_loss, preds, refs = 0.0, [], []
    with torch.no_grad():
        for src, tgt in tqdm(loader, desc='Evaluating', leave=False):
            src = src.to(DEVICE); tgt = tgt.to(DEVICE)
            with autocast('cuda', dtype=AMP_DTYPE):
                logits = model(src, tgt)                            # (B, T, V)
            loss = _eval_crit(
                logits[:, 1:].reshape(-1, config['vocab_size']),
                tgt[:, 1:].reshape(-1)
            )
            total_loss += loss.item()
            for pred_tok, tgt_tok in zip(logits.argmax(-1), tgt):
                preds.append(sp.decode(pred_tok.cpu().numpy().tolist()))
                refs.append( sp.decode(tgt_tok.cpu().numpy().tolist()))
    bleu, chrf = compute_metrics(preds, refs)
    return total_loss / len(loader), bleu, chrf, preds, refs

# ── Training state ────────────────────────────────────────────────────────────
train_losses, val_losses           = [], []
train_bleu, val_bleu               = [], []
train_chrf, val_chrf               = [], []
lr_history, epoch_times            = [], []
best_val_loss = float('inf')
best_epoch    = 0
global_step   = 0
training_start = time.time()

print("=" * 60)
print("  Starting translation fine-tuning")
print("=" * 60)

for epoch in range(config['epochs']):
    epoch_start = time.time()
    model.train()
    total_loss, t_preds, t_refs = 0.0, [], []
    optimizer.zero_grad()
    bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

    for step, (src, tgt) in enumerate(bar):
        src = src.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)

        with autocast('cuda', dtype=AMP_DTYPE):
            logits = model(src, tgt)
            loss   = criterion(
                logits[:, 1:].reshape(-1, config['vocab_size']),
                tgt[:, 1:].reshape(-1)
            ) / accum_steps

        if scaler: scaler.scale(loss).backward()
        else:      loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            if scaler: scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler: scaler.step(optimizer); scaler.update()
            else:      optimizer.step()
            scheduler.step(); optimizer.zero_grad(); global_step += 1

        total_loss += loss.item() * accum_steps

        if len(t_preds) < 500:
            for p, r in zip(logits.argmax(-1).detach(), tgt):
                t_preds.append(sp.decode(p.cpu().numpy().tolist()))
                t_refs.append( sp.decode(r.cpu().numpy().tolist()))

        bar.set_postfix(loss=f"{loss.item()*accum_steps:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}")

    tr_bleu, tr_chrf = compute_metrics(t_preds, t_refs)
    vl_loss, vl_bleu, vl_chrf, v_preds, v_refs = evaluate(valid_loader)
    tr_loss    = total_loss / len(train_loader)
    epoch_time = time.time() - epoch_start
    current_lr = scheduler.get_last_lr()[0]
    gpu_mem    = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    train_losses.append(tr_loss); val_losses.append(vl_loss)
    train_bleu.append(tr_bleu);   val_bleu.append(vl_bleu)
    train_chrf.append(tr_chrf);   val_chrf.append(vl_chrf)
    lr_history.append(current_lr); epoch_times.append(epoch_time)

    print(f"\n  Epoch {epoch+1}/{config['epochs']} | {epoch_time/60:.1f}min | GPU {gpu_mem:.2f}GB | LR {current_lr:.2e}")
    print(f"    Train  Loss: {tr_loss:.4f}  BLEU: {tr_bleu:.2f}  CHRF++: {tr_chrf:.2f}")
    print(f"    Valid  Loss: {vl_loss:.4f}  BLEU: {vl_bleu:.2f}  CHRF++: {vl_chrf:.2f}")

    torch.save(model.state_dict(), f"checkpoints/translation_epoch{epoch+1}.pt")
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss; best_epoch = epoch + 1
        torch.save(model.state_dict(), "checkpoints/translation_best.pt")
        with open("outputs/translation/best_val_predictions.txt", "w", encoding="utf-8") as f:
            for p, r in zip(v_preds[:200], v_refs[:200]):
                f.write(f"REF : {r}\nPRED: {p}\n\n")
        print(f"    ✓  Best model saved (epoch {best_epoch})")

# ── Post-training ─────────────────────────────────────────────────────────────
total_h = (time.time() - training_start) / 3600
metrics_df = pd.DataFrame({
    'epoch': range(1, config['epochs']+1),
    'train_loss': train_losses, 'val_loss': val_losses,
    'train_bleu': train_bleu,   'val_bleu': val_bleu,
    'train_chrf': train_chrf,   'val_chrf': val_chrf,
    'lr': lr_history, 'epoch_time_s': epoch_times,
})
metrics_df.to_csv('outputs/translation/metrics.csv', index=False)

plot_metric(train_losses, val_losses, 'Loss',       'outputs/translation/loss_plot.png',  'Train vs Val Loss')
plot_metric(train_bleu,   val_bleu,   'BLEU (100)', 'outputs/translation/bleu_plot.png',  'Train vs Val BLEU')
plot_metric(train_chrf,   val_chrf,   'CHRF++(100)','outputs/translation/chrf_plot.png',  'Train vs Val CHRF++')
plot_lr(lr_history, 'outputs/translation/lr_plot.png')

# Test evaluation
print("\n" + "=" * 60)
print("  Test evaluation with best model ...")
model.load_state_dict(torch.load("checkpoints/translation_best.pt", map_location=DEVICE))
test_loss, test_bleu, test_chrf, test_preds, test_refs = evaluate(test_loader)
print(f"  TEST  Loss: {test_loss:.4f}  BLEU: {test_bleu:.2f}  CHRF++: {test_chrf:.2f}")

with open("outputs/translation/test_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Best epoch       : {best_epoch}\n")
    f.write(f"Test Loss        : {test_loss:.4f}\n")
    f.write(f"Test BLEU (100)  : {test_bleu:.2f}\n")
    f.write(f"Test CHRF++ (100): {test_chrf:.2f}\n")

with open("outputs/translation/test_predictions.txt", "w", encoding="utf-8") as f:
    for p, r in zip(test_preds[:200], test_refs[:200]):
        f.write(f"REF : {r}\nPRED: {p}\n\n")

summary = {
    'experiment': 'part2_translation',
    'config': config,
    'params': param_counts,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    'training': {
        'total_hours': round(total_h, 2),
        'best_epoch': best_epoch,
        'best_val_loss': round(best_val_loss, 4),
        'best_val_bleu': round(val_bleu[best_epoch-1], 2),
        'best_val_chrf': round(val_chrf[best_epoch-1], 2),
    },
    'test': {'loss': round(test_loss,4), 'bleu': round(test_bleu,2), 'chrf': round(test_chrf,2)},
    'per_epoch': metrics_df.round(4).to_dict(orient='records'),
}
with open("outputs/translation/training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Fine-tuning complete | {total_h:.2f} h")
print(f"  All results in outputs/translation/")
print("=" * 60)
