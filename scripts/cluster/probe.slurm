#!/usr/bin/env bash
#SBATCH --job-name=jepa-probe
#SBATCH --partition=npl-2024
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/u/scratch/EFCM/EFCMdvlr/jepa/logs/probe_%j.out
#SBATCH --error=/gpfs/u/scratch/EFCM/EFCMdvlr/jepa/logs/probe_%j.out
set -euo pipefail
cd /gpfs/u/scratch/EFCM/EFCMdvlr/jepa
mkdir -p logs
source .venv/bin/activate
export HF_HOME=/gpfs/u/scratch/EFCM/EFCMdvlr/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
python scripts/probe_supervised.py \
  --manifest /gpfs/u/scratch/EFCM/EFCMdvlr/label_pool/label_pool.jsonl \
  --labels /gpfs/u/scratch/EFCM/EFCMdvlr/label_pool/label_pool_evaluation_labels.jsonl \
  --folds 5
