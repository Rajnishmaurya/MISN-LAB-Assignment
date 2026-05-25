#!/bin/bash
#SBATCH --job-name=bert_pretrain
#SBATCH --output=logs/bert_pretrain_%j.out
#SBATCH --error=logs/bert_pretrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu          # change to your cluster's partition name

source ~/.bashrc
conda activate nmt          # change to your conda env 

cd /home/da24m015/part2

echo "========================================================"
echo "  JOB ID    : $SLURM_JOB_ID"
echo "  NODE      : $(hostname)"
echo "  START     : $(date)"
echo "========================================================"

nvidia-smi

python -c "
import torch, sys
print('Python     :', sys.version.split()[0])
print('PyTorch    :', torch.__version__)
print('CUDA avail :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU        :', torch.cuda.get_device_name(0))
    print('VRAM       :', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')
"

echo ""
echo "========================================================"
echo "  Starting BERT MLM pretraining"
echo "========================================================"

python src/pretrain_bert.py --config configs/bert_pretrain.yaml

EXIT_CODE=$?

echo ""
echo "========================================================"
echo "  BERT pretraining done  |  exit: $EXIT_CODE"
echo "  END: $(date)"
echo "========================================================"

exit $EXIT_CODE
