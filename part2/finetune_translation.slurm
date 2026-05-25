#!/bin/bash
#SBATCH --job-name=part2_translation
#SBATCH --output=logs/translation_%j.out
#SBATCH --error=logs/translation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu          # change to your cluster's partition name

source ~/.bashrc
conda activate nmt

cd /home/da24m015/part2

echo "========================================================"
echo "  JOB ID    : $SLURM_JOB_ID"
echo "  NODE      : $(hostname)"
echo "  START     : $(date)"
echo "========================================================"

nvidia-smi

echo ""
echo "========================================================"
echo "  Starting translation fine-tuning (BERT enc + GPT dec)"
echo "========================================================"

python src/finetune_translation.py --config configs/translation.yaml

EXIT_CODE=$?

echo ""
echo "========================================================"
echo "  Fine-tuning done  |  exit: $EXIT_CODE"
echo "  Results: outputs/translation/"
echo "  END: $(date)"
echo "========================================================"

exit $EXIT_CODE
