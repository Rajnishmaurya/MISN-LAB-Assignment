#!/bin/bash
#SBATCH --job-name=part2_eval_cpu
#SBATCH --output=logs/eval_cpu_%j.out
#SBATCH --error=logs/eval_cpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=cpu

source ~/.bashrc
conda activate nmt

# Disable GPU usage
export CUDA_VISIBLE_DEVICES=""

cd /home/da24m015/part2

echo "========================================================"
echo "  Part 2 - Beam Search Evaluation"
echo "========================================================"

echo ""
echo "---- Greedy eval (test set) ----"

python src/eval_translation.py \
    --config configs/translation.yaml \
    --checkpoint checkpoints/translation_best.pt \
    --split test \
    --beam_width 1

echo ""
echo "---- Beam search eval (beam=5, test set) ----"

python src/eval_translation.py \
    --config configs/translation.yaml \
    --checkpoint checkpoints/translation_best.pt \
    --split test \
    --beam_width 5

echo ""
echo "========================================================"
echo "  Evaluation complete"
echo "  Results: outputs/eval/"
echo "  END: $(date)"
echo "========================================================"
