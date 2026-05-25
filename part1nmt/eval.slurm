#!/bin/bash
#SBATCH --job-name=nmt_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu          # change to your cluster's partition name

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate nmt


cd /home/da24m015/part1nmt

echo "========================================================"
echo "  JOB ID    : $SLURM_JOB_ID"
echo "  NODE      : $(hostname)"
echo "  START TIME: $(date)"
echo "========================================================"

# ── 1. Beam search evaluation on test set ─────────────────────────────────────
echo ""
echo "--- Beam search eval: RANDOM model (beam=5) ---"
python src/eval.py \
    --config      configs/random.yaml \
    --checkpoint  checkpoints/best_random_model.pt \
    --split       test \
    --beam_width  2

echo ""
echo "--- Beam search eval: BERT model (beam=5) ---"
python src/eval.py \
    --config      configs/bert.yaml \
    --checkpoint  checkpoints/best_bert_model.pt \
    --split       test \
    --beam_width  2

# ── 2. Comparative analysis ───────────────────────────────────────────────────
echo ""
echo "--- Comparative analysis: BERT vs Random ---"
python src/compare.py \
    --bert_config   configs/bert.yaml \
    --bert_ckpt     checkpoints/best_bert_model.pt \
    --random_config configs/random.yaml \
    --random_ckpt   checkpoints/best_random_model.pt \
    --split         test \
    --n_examples    0

# ── 3. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  All evaluation complete"
echo "  Results:"
echo "    outputs/bert/test_results.txt"
echo "    outputs/random/test_results.txt"
echo "    outputs/eval/       (beam search results)"
echo "    outputs/comparison/ (side-by-side + bar charts)"
echo "  END TIME: $(date)"
echo "========================================================"
