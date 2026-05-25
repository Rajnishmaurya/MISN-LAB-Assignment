#!/bin/bash
#SBATCH --job-name=gpt_pretrain
#SBATCH --output=logs/gpt_pretrain_%j.out
#SBATCH --error=logs/gpt_pretrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu          # change to your cluster's partition name

source ~/.bashrc
conda activate nmt          # change to your conda env name

cd /home/da24m015/part2

echo "========================================================"
echo "  JOB ID    : $SLURM_JOB_ID"
echo "  NODE      : $(hostname)"
echo "  START     : $(date)"
echo "========================================================"

nvidia-smi

echo ""
echo "========================================================"
echo "  Starting GPT CLM pretraining"
echo "========================================================"

python src/pretrain_gpt.py --config configs/gpt_pretrain.yaml

EXIT_CODE=$?

echo ""
echo "========================================================"
echo "  GPT pretraining done  |  exit: $EXIT_CODE"
echo "  END: $(date)"
echo "========================================================"

exit $EXIT_CODE
