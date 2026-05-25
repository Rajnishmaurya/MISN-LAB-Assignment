# MISN Lab — AdiVaani Hiring Assessment
**IIT Delhi | Due: May 25, 2026**

Hindi ↔ Marathi Neural Machine Translation — two-part assignment exploring classical recurrent architectures and modern Transformer-based language model pretraining.

---

## Repository Structure

```
MISN-LAB-Assignment/
├── part1nmt/          # Part 1 — LSTM + Bahdanau Attention NMT
│   ├── README.md      # Detailed Part 1 documentation
│   ├── run_all.sh     # Single script to reproduce Part 1
│   ├── src/           # All source code
│   ├── configs/       # Hyperparameter configs (random.yaml, bert.yaml)
│   ├── data/          # Processed bidirectional dataset
│   ├── checkpoints/   # Trained model weights
│   ├── outputs/       # Plots, metrics, predictions
│   └── logs/          # SLURM training logs
│
├── part2/             # Part 2 — LM Pretraining + Encoder-Decoder Translation
│   ├── README.md      # Detailed Part 2 documentation
│   ├── run_all.sh     # Single script to reproduce Part 2
│   ├── src/           # All source code
│   ├── configs/       # Hyperparameter configs
│   ├── data/          # Pretraining corpus + translation data
│   ├── checkpoints/   # Trained model weights
│   ├── outputs/       # Plots, metrics, predictions
│   └── logs/          # SLURM training logs
│
├── Adivaani_Hiring_Assignment.pdf   # Original assignment PDF
└── README.md                        # This file
```

---

## Environment Setup

```bash
# Create conda environment
conda create -n nmt python=3.10 -y
conda activate nmt

# Install PyTorch (adjust CUDA version for your system)
# CUDA 11.8:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Install Part 1 dependencies
pip install -r part1nmt/requirements.txt

# Install Part 2 dependencies
pip install -r part2/requirements.txt
```

> Run `nvidia-smi` to check your driver's maximum supported CUDA version.

---

## Part 1 — Classical Neural Machine Translation

> **Detailed documentation:** [part1nmt/README.md](part1nmt/README.md)

### Objective
Implement a Seq2Seq NMT system using LSTM + Bahdanau attention. Compare two embedding strategies: randomly initialized vs BERT pre-initialized embeddings.

### Architecture

```
Source → Embedding → 2-layer Bi-LSTM Encoder
                              ↓
                    Bahdanau Attention (additive)
                              ↓
Target → Embedding → 2-layer LSTM Decoder → Linear → vocab logits
```

| Component | Detail |
|-----------|--------|
| Encoder | 2-layer LSTM, hidden=512 |
| Decoder | 2-layer LSTM + Bahdanau attention, hidden=512 |
| Attention | Additive: `e = V·tanh(W1·s + W2·h)` |
| Tokenizer | Shared SentencePiece BPE, vocab=16,000 |
| Direction tokens | `<hi2mr>` / `<mr2hi>` for bidirectional translation |
| Decoding | Greedy + Beam search (length penalty α=0.6) |

### Experiments

| Experiment | Embedding dim | Params | Batch | Epochs |
|-----------|--------------|--------|-------|--------|
| Random init | 256 | **25.3M** | 256 | 10 |
| BERT init (l3cube Hindi + Marathi) | 768 | **43.8M** | 128 | 10 |

### Results — Part 1

#### Greedy Decoding (test set, 20,780 examples)

| Model | Test Loss | BLEU (100) | CHRF++ (100) | Best Epoch | Training Time |
|-------|----------|-----------|-------------|------------|--------------|
| Random embeddings | 6.7903 | 1.66 | 8.47 | 9 | 7.40 h |
| BERT embeddings | 6.7516 | 0.97 | 7.96 | 10 | 7.31 h |

#### Beam Search (beam=2, test set)

| Model | BLEU (100) | CHRF++ (100) |
|-------|-----------|-------------|
| Random embeddings | 2.00 | 9.34 |
| BERT embeddings | 1.98 | 9.46 |

### Reproduce Part 1

```bash
cd part1nmt
conda activate nmt
bash run_all.sh
```

Or step by step:
```bash
# 1. Prepare data
python create_split.py
mv data/train_new.hi data/train.hi && mv data/train_new.mr data/train.mr
python create_biderectional_data.py
cat data/train.src data/train.tgt > final_combined.txt
python create_tokenizer.py

# 2. Download BERT models (for BERT experiment)
python download_models.py

# 3. Train
sbatch train_random.slurm   # ~7.4h on A100
sbatch train_bert.slurm     # ~7.3h on A100

# 4. Evaluate
sbatch eval.slurm
```

---

## Part 2 — Language Model Pretraining and Translation

> **Detailed documentation:** [part2/README.md](part2/README.md)

### Objective
Pretrain a BERT-like encoder and GPT-2 style decoder entirely from scratch on Hindi + Marathi monolingual text. Combine them into an encoder-decoder translation model by adding cross-attention to the GPT decoder and loading pretrained weights.

### Architectural Modifications (all from scratch)

| Component | Standard | Implemented |
|-----------|----------|-------------|
| Positional Embeddings | Sinusoidal / Learned | **RoPE** (Su et al. 2021) |
| Attention | Multi-Head Attention | **GQA** — num_heads=12, kv_heads=4 (Ainslie et al. 2023) |
| Normalization | LayerNorm | **RMSNorm** (Zhang & Sennrich 2019) |

### Models

#### BERT-like Encoder [`src/bert_model.py`]
```
Token Embedding → 12 × [RMSNorm → GQA Self-Attn (bidirectional, RoPE) → RMSNorm → FFN] → RMSNorm → MLM Head
```
- hidden=768, layers=12, heads=12, kv_heads=4, ffn=3072
- Parameters: **100,109,440 (~100M)**
- Pretraining: MLM — 15% masking (80% MASK, 10% random, 10% keep), no NSP

#### GPT-2 Style Decoder [`src/gpt_model.py`]
```
Token Embedding → 16 × [RMSNorm → GQA Causal Self-Attn (RoPE) → RMSNorm → FFN] → RMSNorm → LM Head
```
- hidden=768, layers=16, heads=12, kv_heads=4, ffn=3072
- Parameters: **125,281,408 (~125M)**
- Pretraining: CLM — next-token prediction

#### Encoder-Decoder Translation Model [`src/translation_model.py`]
```
Source → BERT Encoder → contextual K, V
Target → Decoder Embedding → 16 × [Self-Attn → Cross-Attn (K,V from encoder) → FFN] → LM Head
```

| Component | Params | Init |
|-----------|--------|------|
| Encoder | 87,804,672 | BERT checkpoint |
| Decoder self-attn + FFN | 112,976,640 | GPT checkpoint |
| Decoder cross-attention | 25,178,112 | **Random** |
| LM head | 12,288,000 | Random |
| **Total** | **238,247,424** | — |

### Tokenizer
- Separate from Part 1 — includes `[MASK]` token (ID=4) for MLM pretraining
- SentencePiece BPE, vocab=16,000, character_coverage=1.0
- Special IDs: PAD=0, BOS=1, EOS=2, UNK=3, **[MASK]=4**

### Results — Part 2

#### BERT Pretraining (5 epochs, A100 80GB, 4.87 h)

| Epoch | Train Loss | Val Loss | Val PPL |
|-------|-----------|---------|---------|
| 1 | 6.478 | 5.133 | 169.49 |
| 2 | 4.750 | 4.277 | 72.02 |
| 3 | 4.171 | 3.861 | 47.52 |
| 4 | 3.884 | 3.663 | 38.97 |
| **5 ✓** | **3.770** | **3.656** | **38.69** |

#### GPT Pretraining (5 epochs, A100 80GB, 6.47 h)

| Epoch | Train Loss | Val Loss | Val PPL |
|-------|-----------|---------|---------|
| 1 | 5.952 | 4.885 | 132.24 |
| 2 | 4.628 | 4.316 | 74.85 |
| 3 | 4.232 | 4.085 | 59.44 |
| 4 | 4.042 | 3.988 | 53.93 |
| **5 ✓** | **3.963** | **3.970** | **53.01** |

#### Translation Fine-tuning (5 epochs, A100 80GB, 2.36 h)

| Epoch | Train Loss | Val BLEU | Val CHRF++ |
|-------|-----------|---------|-----------|
| **1 ✓** | 2.210 | 27.85 | 60.40 |
| 2 | 1.314 | 25.82 | 61.33 |
| 3 | 1.307 | 31.33 | 68.58 |
| 4 | 1.304 | 25.04 | 63.29 |
| 5 | 1.303 | 25.70 | 63.24 |

> **✓ = best checkpoint** (selected by lowest validation loss = 0.0357 at epoch 1). Val BLEU fluctuates across epochs due to the small fine-tuning dataset and label smoothing; the saved model is used for all reported test scores.

#### Final Test Results (greedy, 20,780 examples)

| Test Loss | BLEU (100) | CHRF++ (100) |
|----------|-----------|-------------|
| 0.0383 | **25.46** | **57.71** |

### Reproduce Part 2

```bash
cd part2
conda activate nmt
bash run_all.sh
```

Or step by step on SLURM:
```bash
# 1. Prepare data and tokenizer
python prepare_pretrain_data.py
python create_tokenizer_part2.py

# 2. Pretrain BERT and GPT in parallel
sbatch pretrain_bert.slurm    # ~5h on A100
sbatch pretrain_gpt.slurm     # ~7h on A100

# 3. Fine-tune translation (after both pretrain jobs finish)
sbatch finetune_translation.slurm

# 4. Evaluate
sbatch eval_translation.slurm
```

---

## Overall Results Summary

| Part | Model | BLEU (100) | CHRF++ (100) |
|------|-------|-----------|-------------|
| Part 1 | LSTM + Random embeddings (greedy) | 1.66 | 8.47 |
| Part 1 | LSTM + BERT embeddings (greedy) | 0.97 | 7.96 |
| Part 1 | LSTM + Random embeddings (beam=2) | 2.00 | 9.34 |
| Part 1 | LSTM + BERT embeddings (beam=2) | 1.98 | 9.46 |
| **Part 2** | **BERT encoder + GPT decoder (greedy)** | **25.46** | **57.71** |

---

## Dataset

Hindi–Marathi parallel corpus downloaded from the assignment link.

| Split | Pairs (bidirectional) | Source |
|-------|-----------------------|--------|
| Train | ~435,000 | Assignment dataset |
| Valid | ~48,000 | 10% split from train |
| Test | ~20,780 | Assignment dataset |

**Bidirectional setup:** each original pair generates two examples using direction tokens `<hi2mr>` (Hindi→Marathi) and `<mr2hi>` (Marathi→Hindi), effectively doubling the dataset.

---

## Key Design Decisions

**Part 1:**
- Shared SentencePiece BPE tokenizer for both languages promotes subword overlap between Hindi and Marathi (both Devanagari script)
- Bidirectional training with direction tokens allows a single model to handle both translation directions
- BERT embeddings initialized via vocabulary alignment: each SPM BPE piece mapped to averaged BERT WordPiece vectors — not a direct index copy

**Part 2:**
- BERT encoder chosen for bidirectional context understanding of source sentence; GPT decoder chosen for autoregressive generation
- Cross-attention randomly initialized — no existing weights from either pretrained model can meaningfully initialize source-target alignment
- Lower fine-tuning LR (5e-5 vs 1e-4) to preserve pretrained representations
- Separate tokenizer with [MASK]=4 required for MLM masking pipeline

---

## References

1. Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015
2. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
3. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019
4. Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), OpenAI 2019
5. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021
6. Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023
7. Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019
