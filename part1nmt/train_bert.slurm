#!/bin/bash
#SBATCH --job-name=bert_nmt
#SBATCH --output=logs/bert_%j.out
#SBATCH --error=logs/bert_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu          # change to your cluster's A100 partition name

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate nmt          # change to your conda env name
cd /home/da24m015/part1nmt

# ── System diagnostics (logged for report) ────────────────────────────────────
echo "========================================================"
echo "  JOB ID      : $SLURM_JOB_ID"
echo "  NODE        : $(hostname)"
echo "  START TIME  : $(date)"
echo "  WORK DIR    : $WORK_DIR"
echo "========================================================"

echo ""
echo "--- GPU Info ---"
nvidia-smi

echo ""
echo "--- Python / PyTorch ---"
python -c "
import torch, sys
print('Python      :', sys.version.split()[0])
print('PyTorch     :', torch.__version__)
print('CUDA avail  :', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU count   :', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name    :', torch.cuda.get_device_name(0))
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU VRAM    : {mem:.1f} GB')
"

echo ""
echo "========================================================"
echo "  Starting BERT NMT training"
echo "========================================================"

python src/train_bert.py --config configs/bert.yaml

EXIT_CODE=$?

echo ""
echo "========================================================"
echo "  Training finished  |  exit code: $EXIT_CODE"
echo "  END TIME: $(date)"
echo "========================================================"

exit $EXIT_CODE
