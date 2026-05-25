#!/bin/bash
# =============================================================================
#  run_all.sh — Full reproducible pipeline for Part 2
#
#  Runs every step sequentially: data prep → tokenizer → BERT pretrain →
#  GPT pretrain → translation fine-tune → evaluation.
#
#  Usage:
#      conda activate nmt && bash run_all.sh
#
#  Requirements:
#    - conda environment "nmt" activated
#    - Part 1 data must exist at ../part1nmt/data/ (for building pretrain corpus)
#      OR data/pretrain_corpus.txt already present (skip Step 1)
#    - GPU with at least 40GB VRAM for training steps
#
#  On SLURM cluster (recommended — runs jobs in background):
#    See individual .slurm files instead of this script.
#    BERT and GPT pretraining can run in parallel with:
#      sbatch pretrain_bert.slurm && sbatch pretrain_gpt.slurm
# =============================================================================

set -e   # exit immediately if any command fails

# ── Ensure script runs from part2/ directory ─────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Working directory: $SCRIPT_DIR"

echo "========================================================"
echo "  Part 2 — Full Reproducible Pipeline"
echo "  Start: $(date)"
echo "========================================================"

# ── Step 1: Build pretraining corpus ─────────────────────────────────────────
echo ""
echo ">>> STEP 1: Building pretraining corpus..."
if [ -f "data/pretrain_corpus.txt" ]; then
    echo "    data/pretrain_corpus.txt already exists — skipping."
    echo "    (Delete it and re-run to regenerate from scratch)"
else
    python prepare_pretrain_data.py
    echo "    Done. data/pretrain_corpus.txt created."
fi

# ── Step 2: Train SentencePiece tokenizer ────────────────────────────────────
echo ""
echo ">>> STEP 2: Training SentencePiece tokenizer (Part 2, with [MASK]=4)..."
if [ -f "spm_part2.model" ]; then
    echo "    spm_part2.model already exists — skipping."
else
    python create_tokenizer_part2.py
    echo "    Done. spm_part2.model created. [MASK] ID verified = 4."
fi

# ── Step 3: Create output directories ────────────────────────────────────────
echo ""
echo ">>> STEP 3: Creating output directories..."
mkdir -p checkpoints
mkdir -p outputs/bert_pretrain
mkdir -p outputs/gpt_pretrain
mkdir -p outputs/translation
mkdir -p outputs/eval
mkdir -p logs
echo "    Done."

# ── Step 4: Pretrain BERT (~100M, MLM) ───────────────────────────────────────
echo ""
echo "========================================================"
echo ">>> STEP 4: Pretraining BERT encoder (~100M params, MLM)..."
echo "    Config : configs/bert_pretrain.yaml"
echo "    Output : outputs/bert_pretrain/"
echo "    Note   : This step takes ~5h on A100. On SLURM use:"
echo "             sbatch pretrain_bert.slurm"
echo "========================================================"
python src/pretrain_bert.py --config configs/bert_pretrain.yaml
echo "    Done. Checkpoint: checkpoints/bert_pretrained_best.pt"

# ── Step 5: Pretrain GPT (~125M, CLM) ────────────────────────────────────────
echo ""
echo "========================================================"
echo ">>> STEP 5: Pretraining GPT decoder (~125M params, CLM)..."
echo "    Config : configs/gpt_pretrain.yaml"
echo "    Output : outputs/gpt_pretrain/"
echo "    Note   : This step takes ~6h on A100. On SLURM use:"
echo "             sbatch pretrain_gpt.slurm"
echo "========================================================"
python src/pretrain_gpt.py --config configs/gpt_pretrain.yaml
echo "    Done. Checkpoint: checkpoints/gpt_pretrained_best.pt"

# ── Step 6: Fine-tune translation model ──────────────────────────────────────
echo ""
echo "========================================================"
echo ">>> STEP 6: Fine-tuning encoder-decoder translation model..."
echo "    Config     : configs/translation.yaml"
echo "    BERT ckpt  : checkpoints/bert_pretrained_best.pt"
echo "    GPT ckpt   : checkpoints/gpt_pretrained_best.pt"
echo "    Output     : outputs/translation/"
echo "    Note       : On SLURM use: sbatch finetune_translation.slurm"
echo "========================================================"

# Verify pretrain checkpoints exist before fine-tuning
if [ ! -f "checkpoints/bert_pretrained_best.pt" ]; then
    echo "ERROR: checkpoints/bert_pretrained_best.pt not found!"
    echo "       Complete Step 4 (BERT pretraining) first."
    exit 1
fi
if [ ! -f "checkpoints/gpt_pretrained_best.pt" ]; then
    echo "ERROR: checkpoints/gpt_pretrained_best.pt not found!"
    echo "       Complete Step 5 (GPT pretraining) first."
    exit 1
fi

python src/finetune_translation.py --config configs/translation.yaml
echo "    Done. Checkpoint: checkpoints/translation_best.pt"

# ── Step 7: Evaluate on test set ─────────────────────────────────────────────
echo ""
echo ">>> STEP 7: Evaluating on test set (greedy decoding)..."
python src/eval_translation.py \
    --config     configs/translation.yaml \
    --checkpoint checkpoints/translation_best.pt \
    --split      test \
    --beam_width 1
echo "    Done. Results saved to outputs/eval/translation_best_test_greedy/"

# ── Final Summary ─────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  ALL STEPS COMPLETE"
echo "  End: $(date)"
echo ""
echo "  Key checkpoints:"
echo "    checkpoints/bert_pretrained_best.pt"
echo "    checkpoints/gpt_pretrained_best.pt"
echo "    checkpoints/translation_best.pt"
echo ""
echo "  Key outputs:"
echo "    outputs/bert_pretrain/loss_plot.png        (BERT MLM loss)"
echo "    outputs/bert_pretrain/val_ppl_plot.png     (BERT perplexity)"
echo "    outputs/gpt_pretrain/loss_plot.png         (GPT CLM loss)"
echo "    outputs/gpt_pretrain/val_ppl_plot.png      (GPT perplexity)"
echo "    outputs/translation/loss_plot.png          (translation loss)"
echo "    outputs/translation/bleu_plot.png          (BLEU curve)"
echo "    outputs/translation/chrf_plot.png          (CHRF++ curve)"
echo "    outputs/translation/test_results.txt       (final scores)"
echo "    outputs/eval/                              (beam search scores)"
echo "========================================================"
