#!/bin/bash
# =============================================================================
#  run_all.sh — Full reproducible pipeline for Part 1 NMT
#
#  Runs every step from raw data to final evaluation in one script.
#
#  Usage:
#      bash run_all.sh
#
#  Requirements:
#    - conda environment "nmt" already created and activated, OR
#      run this script inside the conda env:
#          conda activate nmt && bash run_all.sh
#    - Raw data files present: data/train.hi, data/train.mr,
#                               data/test.hi,  data/test.mr
#    - Internet access (for downloading BERT models in Step 4)
# =============================================================================

set -e   # exit immediately if any command fails

# ── Ensure script runs from part1nmt/ directory ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Working directory: $SCRIPT_DIR"

echo "========================================================"
echo "  Part 1 NMT — Full Reproducible Pipeline"
echo "  Start: $(date)"
echo "========================================================"

# ── Step 1: Create validation split ──────────────────────────────────────────
echo ""
echo ">>> STEP 1: Creating validation split (90/10 from train data)..."
python create_split.py

# Rename train_new → train (overwrite original)
mv data/train_new.hi data/train.hi
mv data/train_new.mr data/train.mr
echo "    Done. train/valid split created."

# ── Step 2: Build bidirectional dataset ──────────────────────────────────────
echo ""
echo ">>> STEP 2: Building bidirectional .src/.tgt files..."
python create_biderectional_data.py
echo "    Done. train/valid/test .src .tgt files created."

# ── Step 3: Train SentencePiece tokenizer ────────────────────────────────────
echo ""
echo ">>> STEP 3: Training SentencePiece BPE tokenizer (vocab=16000)..."
cat data/train.src data/train.tgt > final_combined.txt
python create_tokenizer.py
echo "    Done. spm.model and spm.vocab created."

# ── Step 4: Download pretrained BERT models ───────────────────────────────────
echo ""
echo ">>> STEP 4: Downloading Hindi and Marathi BERT models..."
python download_models.py
echo "    Done. Models saved to pretrained_models/."

# ── Step 5: Create output directories ────────────────────────────────────────
echo ""
echo ">>> STEP 5: Creating output directories..."
mkdir -p checkpoints
mkdir -p outputs/random outputs/bert outputs/eval outputs/comparison
mkdir -p logs
echo "    Done."

# ── Step 6: Train Random Embeddings model ────────────────────────────────────
echo ""
echo "========================================================"
echo ">>> STEP 6: Training Random Embeddings model..."
echo "    Config : configs/random.yaml"
echo "    Output : outputs/random/"
echo "========================================================"
python src/train_random.py --config configs/random.yaml
echo "    Done. Best model saved to checkpoints/best_random_model.pt"

# ── Step 7: Train BERT Embeddings model ──────────────────────────────────────
echo ""
echo "========================================================"
echo ">>> STEP 7: Training BERT Embeddings model..."
echo "    Config : configs/bert.yaml"
echo "    Output : outputs/bert/"
echo "========================================================"
python src/train_bert.py --config configs/bert.yaml
echo "    Done. Best model saved to checkpoints/best_bert_model.pt"

# ── Step 8: Beam search evaluation — Random model ────────────────────────────
echo ""
echo ">>> STEP 8: Beam search evaluation — Random model (beam=2)..."
python src/eval.py \
    --config     configs/random.yaml \
    --checkpoint checkpoints/best_random_model.pt \
    --split      test \
    --beam_width 2
echo "    Done. Results saved to outputs/eval/best_random_model_test_beam2/"

# ── Step 9: Beam search evaluation — BERT model ──────────────────────────────
echo ""
echo ">>> STEP 9: Beam search evaluation — BERT model (beam=2)..."
python src/eval.py \
    --config     configs/bert.yaml \
    --checkpoint checkpoints/best_bert_model.pt \
    --split      test \
    --beam_width 2
echo "    Done. Results saved to outputs/eval/best_bert_model_test_beam2/"

# ── Step 10: Comparative analysis ────────────────────────────────────────────
echo ""
echo ">>> STEP 10: Running comparative analysis (BERT vs Random)..."
python src/compare.py \
    --bert_config   configs/bert.yaml \
    --bert_ckpt     checkpoints/best_bert_model.pt \
    --random_config configs/random.yaml \
    --random_ckpt   checkpoints/best_random_model.pt \
    --split         test \
    --n_examples    0
echo "    Done. Bar charts and side-by-side saved to outputs/comparison/"

# ── Final Summary ─────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  ALL STEPS COMPLETE"
echo "  End: $(date)"
echo ""
echo "  Key outputs:"
echo "    checkpoints/best_random_model.pt"
echo "    checkpoints/best_bert_model.pt"
echo "    outputs/random/test_results.txt         (greedy scores)"
echo "    outputs/bert/test_results.txt           (greedy scores)"
echo "    outputs/eval/                           (beam search scores)"
echo "    outputs/comparison/metrics_test.txt     (BERT vs Random table)"
echo "    outputs/comparison/bleu_comparison_test.png"
echo "    outputs/comparison/chrf_comparison_test.png"
echo "    outputs/comparison/side_by_side_test.txt"
echo "========================================================"
