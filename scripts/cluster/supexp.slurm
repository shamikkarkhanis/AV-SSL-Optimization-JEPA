#!/usr/bin/env bash
#SBATCH --job-name=jepa-supexp
#SBATCH --partition=npl-2024
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/gpfs/u/scratch/EFCM/EFCMdvlr/jepa/logs/supexp_%j.out
#SBATCH --error=/gpfs/u/scratch/EFCM/EFCMdvlr/jepa/logs/supexp_%j.out
set -uo pipefail
cd /gpfs/u/scratch/EFCM/EFCMdvlr/jepa
source .venv/bin/activate
export HF_HOME=/gpfs/u/scratch/EFCM/EFCMdvlr/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
python scripts/run_supervised_experiment.py \
  --label-pool-manifest /gpfs/u/scratch/EFCM/EFCMdvlr/label_pool/label_pool.jsonl \
  --label-pool-labels /gpfs/u/scratch/EFCM/EFCMdvlr/label_pool/label_pool_evaluation_labels.jsonl \
  --benchmark-labels /gpfs/u/barn/EFCM/shared/data/manifests/nuscenes_trainval01_evaluation_labels.jsonl \
  --repeats 10
