"""
GPT-2 style model pretraining with Causal Language Modeling (CLM).

Usage:
    python src/pretrain_gpt.py --config configs/gpt_pretrain.yaml
"""

import os, sys, math, time, json, random, argparse
import yaml, torch, numpy as np, pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt_model  import GPTModel
from clm_dataset import CLMDataset
from plotting    import plot_metric, plot_single, plot_lr

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
print("  GPT Pretraining — CLM")
print("=" * 60)
print(f"  Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"  AMP    : {_dtype_str}")

# ── Output dirs ───────────────────────────────────────────────────────────────
os.makedirs('checkpoints',          exist_ok=True)
os.makedirs('outputs/gpt_pretrain', exist_ok=True)
os.makedirs('logs',                 exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
full_dataset = CLMDataset(
    text_file = config['pretrain_text'],
    spm_model = config['spm_model'],
    max_len   = config.get('max_seq_len', 512),
)

val_size   = max(1000, int(0.02 * len(full_dataset)))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                generator=torch.Generator().manual_seed(SEED))

num_workers  = config.get('num_workers', 4)
train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                          shuffle=True,  num_workers=num_workers, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                          shuffle=False, num_workers=num_workers, pin_memory=True)

print(f"\n  Train : {len(train_ds):,} samples  ({len(train_loader):,} batches)")
print(f"  Val   : {len(val_ds):,}  samples  ({len(val_loader):,}  batches)\n")

# ── Model ─────────────────────────────────────────────────────────────────────
model = GPTModel(
    vocab_size   = config['vocab_size'],
    hidden_dim   = config.get('hidden_dim',   768),
    num_layers   = config.get('num_layers',    16),
    num_heads    = config.get('num_heads',     12),
    num_kv_heads = config.get('num_kv_heads',   4),
    ffn_dim      = config.get('ffn_dim',      3072),
    max_seq_len  = config.get('max_seq_len',   512),
    dropout      = config.get('dropout',       0.1),
    pad_id       = config.get('pad_id',          0),
).to(DEVICE)

total = model.count_parameters()
print(f"  GPT params : {total:,}  (~{total/1e6:.1f}M)")

# ── Optimizer / Scheduler ────────────────────────────────────────────────────
criterion    = torch.nn.CrossEntropyLoss(ignore_index=-100)
optimizer    = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                  betas=(0.9, 0.95), weight_decay=0.1)
accum_steps  = config.get('accumulation_steps', 1)
total_steps  = math.ceil(len(train_loader) / accum_steps) * config['epochs']
warmup_steps = int(total_steps * config.get('warmup_ratio', 0.05))
min_lr_ratio = config.get('min_lr', 1e-6) / config['lr']

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = LambdaLR(optimizer, lr_lambda)
scaler    = GradScaler('cuda') if USE_SCALER else None

# ── Training state ────────────────────────────────────────────────────────────
train_losses, val_losses = [], []
lr_history, epoch_times  = [], []
best_val_loss = float('inf')
best_epoch    = 0
global_step   = 0
training_start = time.time()

print("=" * 60)
print("  Starting CLM pretraining")
print("=" * 60)

# ── Epoch loop ────────────────────────────────────────────────────────────────
for epoch in range(config['epochs']):
    epoch_start = time.time()
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

    for step, batch in enumerate(bar):
        input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
        labels    = batch['labels'].to(DEVICE, non_blocking=True)

        with autocast('cuda', dtype=AMP_DTYPE):
            logits = model(input_ids)                                   # (B, T, V)
            loss   = criterion(logits.view(-1, config['vocab_size']), labels.view(-1))
            loss   = loss / accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            if scaler: scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        total_loss += loss.item() * accum_steps
        bar.set_postfix(loss=f"{loss.item()*accum_steps:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}")

    # ── Validation ───────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(DEVICE)
            labels    = batch['labels'].to(DEVICE)
            with autocast('cuda', dtype=AMP_DTYPE):
                logits = model(input_ids)
                loss   = criterion(logits.view(-1, config['vocab_size']), labels.view(-1))
            val_loss += loss.item()
    val_loss /= len(val_loader)

    train_loss = total_loss / len(train_loader)
    epoch_time = time.time() - epoch_start
    current_lr = scheduler.get_last_lr()[0]
    gpu_mem    = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    lr_history.append(current_lr)
    epoch_times.append(epoch_time)

    train_ppl = math.exp(min(train_loss, 20))
    val_ppl   = math.exp(min(val_loss,   20))

    print(f"\n  Epoch {epoch+1}/{config['epochs']} | {epoch_time/60:.1f}min | "
          f"GPU {gpu_mem:.2f}GB | LR {current_lr:.2e}")
    print(f"    Train Loss: {train_loss:.4f}  PPL: {train_ppl:.2f}")
    print(f"    Val   Loss: {val_loss:.4f}  PPL: {val_ppl:.2f}")

    torch.save(model.state_dict(), f"checkpoints/gpt_pretrain_epoch{epoch+1}.pt")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch + 1
        torch.save(model.state_dict(), "checkpoints/gpt_pretrained_best.pt")
        print(f"    ✓  Best GPT model saved (epoch {best_epoch})")

# ── Post-training ─────────────────────────────────────────────────────────────
total_h = (time.time() - training_start) / 3600
metrics_df = pd.DataFrame({
    'epoch': range(1, config['epochs'] + 1),
    'train_loss': train_losses, 'val_loss': val_losses,
    'lr': lr_history, 'epoch_time_s': epoch_times,
    'train_ppl': [math.exp(min(l, 20)) for l in train_losses],
    'val_ppl':   [math.exp(min(l, 20)) for l in val_losses],
})
metrics_df.to_csv('outputs/gpt_pretrain/metrics.csv', index=False)

plot_metric(train_losses, val_losses, 'Loss (CLM)',
            'outputs/gpt_pretrain/loss_plot.png', title='GPT CLM Train vs Val Loss')
plot_single([math.exp(min(l,20)) for l in train_losses],
            'Perplexity', 'outputs/gpt_pretrain/train_ppl_plot.png', 'Train Perplexity')
plot_single([math.exp(min(l,20)) for l in val_losses],
            'Perplexity', 'outputs/gpt_pretrain/val_ppl_plot.png',   'Val Perplexity')
plot_lr(lr_history, 'outputs/gpt_pretrain/lr_plot.png')

summary = {
    'model': 'gpt_pretrain',
    'config': config,
    'params': total,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    'training': {
        'total_hours': round(total_h, 2),
        'best_epoch': best_epoch,
        'best_val_loss': round(best_val_loss, 4),
        'best_val_ppl': round(math.exp(min(best_val_loss, 20)), 2),
    },
    'per_epoch': metrics_df.round(4).to_dict(orient='records'),
}
with open('outputs/gpt_pretrain/training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print(f"  GPT pretraining complete  |  {total_h:.2f} h")
print(f"  Best epoch: {best_epoch}  |  Best val loss: {best_val_loss:.4f}")
print(f"  Checkpoint: checkpoints/gpt_pretrained_best.pt")
print("=" * 60)
