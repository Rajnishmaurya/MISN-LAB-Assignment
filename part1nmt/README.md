# Part 1 — Hindi ↔ Marathi Neural Machine Translation

LSTM + Bahdanau Attention Seq2Seq with two embedding strategies:
- **Random**: randomly initialized, fully trainable embeddings
- **BERT**: embeddings pre-initialized from `l3cube-pune/hindi-bert-v2` (encoder) and `l3cube-pune/marathi-bert-v2` (decoder)

Both models support **bidirectional** translation (Hindi→Marathi and Marathi→Hindi) using direction tokens `<hi2mr>` and `<mr2hi>` prepended to every source sentence.

---

## Quick Reproduce (Single Script)

To reproduce the entire pipeline from scratch in one command:

```bash
bash run_all.sh
```

> See `run_all.sh` for the complete step-by-step pipeline. Each step can also be run individually as described below.

---

## Directory Structure

```
part1nmt/
├── configs/
│   ├── bert.yaml                    # Hyperparameters for BERT-init experiment
│   └── random.yaml                  # Hyperparameters for random-init experiment
├── src/
│   ├── model.py                     # Encoder, Decoder, Seq2Seq model classes
│   ├── attention.py                 # Bahdanau (additive) attention mechanism
│   ├── dataset.py                   # TranslationDataset with SentencePiece tokenization
│   ├── bert_embeddings.py           # Vocabulary-aligned BERT → SPM embedding initialization
│   ├── evaluate.py                  # evaluate_model() used inside training loop
│   ├── metrics.py                   # BLEU(100) and CHRF++(100) via sacrebleu
│   ├── beam_search.py               # Beam search decoder with length penalty (α=0.6)
│   ├── plotting.py                  # Loss / BLEU / CHRF++ / LR curve plots
│   ├── tokenizer_utils.py           # SentencePiece BPE training wrapper
│   ├── train_random.py              # Full training script — random embeddings
│   ├── train_bert.py                # Full training script — BERT embeddings
│   ├── eval.py                      # Standalone evaluation (greedy or beam search)
│   └── compare.py                   # Side-by-side BERT vs Random comparison + bar charts
├── data/                            # Created by data pipeline (see below)
│   ├── train.hi / train.mr          # Raw Hindi / Marathi training sentences
│   ├── valid.hi / valid.mr          # Validation split (10% of train)
│   ├── test.hi  / test.mr           # Raw test sentences
│   ├── train.src / train.tgt        # Bidirectional training pairs
│   ├── valid.src / valid.tgt        # Bidirectional validation pairs
│   └── test.src  / test.tgt         # Bidirectional test pairs
├── pretrained_models/               # Created by download_models.py
│   ├── hindi-bert-v2/               # l3cube-pune/hindi-bert-v2 (cached locally)
│   └── marathi-bert-v2/             # l3cube-pune/marathi-bert-v2 (cached locally)
├── checkpoints/
│   ├── best_random_model.pt         # Best random model (by validation loss)
│   └── best_bert_model.pt           # Best BERT model (by validation loss)
├── outputs/
│   ├── random/                      # All metrics, plots, predictions — random model
│   ├── bert/                        # All metrics, plots, predictions — BERT model
│   ├── eval/                        # Standalone eval.py results (beam search)
│   └── comparison/                  # compare.py bar charts + side-by-side text
├── logs/                            # SLURM stdout / stderr logs
├── create_split.py                  # Step 1: create validation split from train data
├── create_biderectional_data.py     # Step 2: build bidirectional .src/.tgt files
├── create_tokenizer.py              # Step 3: train shared SentencePiece BPE tokenizer
├── download_models.py               # Download BERT models and cache locally
├── verify_tokenizer.py              # Verify tokenizer special token IDs
├── train_random.slurm               # SLURM job script — random embedding training
├── train_bert.slurm                 # SLURM job script — BERT embedding training
├── eval.slurm                       # SLURM job script — beam search eval + comparison
├── run_all.sh                       # Single script to reproduce entire pipeline
└── requirements.txt                 # Python dependencies
```

---

## Environment Setup

```bash
# Create and activate conda environment
conda create -n nmt python=3.10 -y
conda activate nmt

# Install PyTorch with CUDA (adjust CUDA version for your system)
# For CUDA 11.8:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Install all remaining dependencies
pip install -r requirements.txt
```

> Run `nvidia-smi` to check your driver's maximum supported CUDA version.

---

## Data Pipeline

### Raw data format

Place the downloaded dataset files in `data/`:

```
data/
├── train.hi    # Hindi training sentences (one per line)
├── train.mr    # Marathi training sentences (aligned with train.hi)
├── test.hi     # Hindi test sentences
└── test.mr     # Marathi test sentences
```

### Step 1 — Create validation split

```bash
python create_split.py
```

- Reads `data/train.hi` and `data/train.mr`
- Splits 90% → training, 10% → validation (random seed=42)
- Writes: `data/train_new.hi`, `data/train_new.mr`, `data/valid.hi`, `data/valid.mr`

```bash
# Rename train_new files to train
mv data/train_new.hi data/train.hi
mv data/train_new.mr data/train.mr
```

### Step 2 — Build bidirectional dataset

```bash
python create_biderectional_data.py
```

Doubles every split by creating both translation directions with direction tokens:

| Source | Target |
|--------|--------|
| `<hi2mr> <hindi sentence>` | `<marathi sentence>` |
| `<mr2hi> <marathi sentence>` | `<hindi sentence>` |

- Reads: `data/train.hi/.mr`, `data/valid.hi/.mr`, `data/test.hi/.mr`
- Writes: `data/train.src`, `data/train.tgt`, `data/valid.src`, `data/valid.tgt`, `data/test.src`, `data/test.tgt`

### Step 3 — Train SentencePiece tokenizer

```bash
# Combine all source and target text for tokenizer training
cat data/train.src data/train.tgt > final_combined.txt

# Train shared BPE tokenizer
python create_tokenizer.py
```

- Model type: BPE (Byte Pair Encoding)
- Vocab size: 16,000
- Character coverage: 1.0 (covers full Unicode — required for Devanagari script)
- Special token IDs: `PAD=0`, `BOS=1`, `EOS=2`, `UNK=3`
- Outputs: `spm.model`, `spm.vocab`

---

## Download Pretrained BERT Models

Required **only** for the BERT embedding experiment. Run once before submitting SLURM jobs (compute nodes may not have internet access).

```bash
python download_models.py
```

Downloads and caches locally:
- `pretrained_models/hindi-bert-v2/` ← `l3cube-pune/hindi-bert-v2` from HuggingFace
- `pretrained_models/marathi-bert-v2/` ← `l3cube-pune/marathi-bert-v2` from HuggingFace

---

## Model Architecture

### Encoder
| Component | Detail |
|-----------|--------|
| Type | 2-layer LSTM |
| Input | Embedding lookup → LSTM |
| Hidden dim | 512 |
| Embedding dim | 256 (Random) / 768 (BERT) |
| Dropout | 0.3 (between layers) |
| Output | All hidden states + final (hidden, cell) |

### Decoder
| Component | Detail |
|-----------|--------|
| Type | 2-layer LSTM with Bahdanau attention |
| Input | concat(embedding, context vector) → LSTM |
| LSTM input dim | emb_dim + hidden_dim |
| Hidden dim | 512 |
| Output | Linear projection → vocab logits |

### Bahdanau Attention (additive)

The attention score between decoder state `s` and encoder output `h_i` is:

```
e_i  = V · tanh(W1 · s  +  W2 · h_i)
α_i  = softmax(e_i)            # over all source positions
ctx  = Σ α_i · h_i             # weighted sum = context vector
```

**Optimization**: `W2 · h_i` is precomputed once per batch (not at every decoder step), reducing redundant matrix multiplications.

### Seq2Seq with Teacher Forcing
- **Training**: `teacher_forcing_ratio=0.5` — 50% ground-truth, 50% model prediction
- **Evaluation**: `teacher_forcing_ratio=0.0` — fully autoregressive (honest evaluation)
- **BLEU during training**: also uses `teacher_forcing_ratio=0.0` for unbiased metric

### Beam Search
- Beam width: 2 (used for all reported results; configurable via `--beam_width`)
- Length penalty: `score / len(sequence)^0.6` — prevents short sequence bias
- BOS token: ID=1, EOS token: ID=2

---

## Hyperparameters

| Parameter | Random | BERT |
|-----------|--------|------|
| Embedding dim | 256 | 768 (matches BERT hidden size) |
| Hidden dim | 512 | 512 |
| LSTM layers | 2 | 2 |
| Dropout | 0.3 | 0.3 |
| Vocab size | 16,000 | 16,000 |
| Max seq len | 128 | 128 |
| Optimizer | AdamW | AdamW |
| Weight decay | 1e-2 | 1e-2 |
| Learning rate | 1e-4 | 1e-4 |
| Min LR | 1e-6 | 1e-6 |
| LR schedule | Linear warmup (2 epochs) + cosine decay | Same |
| Label smoothing | 0.1 | 0.1 |
| Gradient clipping | 1.0 (norm) | 1.0 (norm) |
| Epochs | 10 | 10 |
| Batch size (A100) | 256 | 128 |
| AMP dtype | bfloat16 | bfloat16 |
| Seed | 42 | 42 |

---

## Training

### On a SLURM cluster (recommended for full training)

```bash
# Edit partition name and conda env name in .slurm files first
# Check available partitions: sinfo

# Train both models (independent — can run in parallel)
sbatch train_random.slurm   # saves to checkpoints/best_random_model.pt
sbatch train_bert.slurm     # saves to checkpoints/best_bert_model.pt

# Monitor jobs
squeue -u $USER
tail -f logs/random_*.out
tail -f logs/bert_*.out
```

### Locally (for debugging/small runs)

```bash
cd part1nmt
python src/train_random.py --config configs/random.yaml
python src/train_bert.py   --config configs/bert.yaml
```

### What gets saved during training

For each model, the following are saved automatically:

```
outputs/{random,bert}/
├── metrics.csv              # Per-epoch: loss, BLEU, CHRF++, LR, epoch time
├── loss_plot.png            # Train vs validation loss curve
├── bleu_plot.png            # Train vs validation BLEU(100) curve
├── chrf_plot.png            # Train vs validation CHRF++(100) curve
├── lr_plot.png              # Learning rate schedule curve
├── best_val_predictions.txt # 200 REF/PRED pairs from best epoch
├── test_results.txt         # Final test BLEU and CHRF++ (greedy)
├── test_predictions.txt     # 200 REF/PRED pairs from test set
└── training_summary.json    # Full JSON: config, GPU info, timing, all metrics
```

---

## Evaluation

### Beam search evaluation (standalone)

```bash
# Random model — beam search on test set
python src/eval.py \
    --config     configs/random.yaml \
    --checkpoint checkpoints/best_random_model.pt \
    --split      test \
    --beam_width 2

# BERT model — beam search on test set
python src/eval.py \
    --config     configs/bert.yaml \
    --checkpoint checkpoints/best_bert_model.pt \
    --split      test \
    --beam_width 2
```

### Side-by-side comparative analysis

```bash
python src/compare.py \
    --bert_config   configs/bert.yaml \
    --bert_ckpt     checkpoints/best_bert_model.pt \
    --random_config configs/random.yaml \
    --random_ckpt   checkpoints/best_random_model.pt \
    --split         test \
    --n_examples    0
```

Produces:
- `outputs/comparison/bleu_comparison_test.png` — bar chart
- `outputs/comparison/chrf_comparison_test.png` — bar chart
- `outputs/comparison/metrics_test.txt` — quantitative table
- `outputs/comparison/side_by_side_test.txt` — per-example REF / BERT / RANDOM

### All evaluation in one SLURM job

```bash
sbatch eval.slurm
```

---

## Results

### Greedy decoding (test set, 20,780 examples)

| Model | BLEU (100) | CHRF++ (100) | Best Epoch |
|-------|-----------|-------------|------------|
| Random embeddings | 1.66 | 8.47 | 9 |
| BERT embeddings | 0.97 | 7.96 | 10 |

### Beam search (beam=2, test set, 20,780 examples)

| Model | BLEU (100) | CHRF++ (100) |
|-------|-----------|-------------|
| Random embeddings | 2.00 | 9.34 |
| BERT embeddings | 1.98 | 9.46 |

> Scores are low due to the limited number of training epochs (10) and the challenging nature of Hindi↔Marathi translation with a small LSTM model. Beam search consistently improves over greedy decoding.

---

## Outputs Reference

| Path | Contents |
|------|----------|
| `outputs/random/metrics.csv` | Per-epoch train/val loss, BLEU, CHRF++, LR, epoch time |
| `outputs/random/loss_plot.png` | Train vs validation loss curve |
| `outputs/random/bleu_plot.png` | Train vs validation BLEU(100) curve |
| `outputs/random/chrf_plot.png` | Train vs validation CHRF++(100) curve |
| `outputs/random/lr_plot.png` | Learning rate schedule |
| `outputs/random/test_results.txt` | Final test BLEU and CHRF++ scores |
| `outputs/random/test_predictions.txt` | 200 REF/PRED example pairs |
| `outputs/random/training_summary.json` | Full JSON summary of training run |
| `outputs/bert/` | Same structure as `outputs/random/` for BERT model |
| `outputs/eval/` | Per-run metrics and predictions from `eval.py` |
| `outputs/comparison/bleu_comparison_test.png` | Bar chart: BERT vs Random BLEU |
| `outputs/comparison/chrf_comparison_test.png` | Bar chart: BERT vs Random CHRF++ |
| `outputs/comparison/metrics_test.txt` | Quantitative comparison table with delta |
| `outputs/comparison/side_by_side_test.txt` | Per-example REF / BERT / RANDOM translations |

---

## References

1. Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015
2. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
3. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019
