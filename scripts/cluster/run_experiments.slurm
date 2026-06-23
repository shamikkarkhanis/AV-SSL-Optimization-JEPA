#!/usr/bin/env bash
#SBATCH --job-name=jepa-ftc
#SBATCH --array=0-24
#SBATCH --partition=npl-2024
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/jepa_%A_%a.out
#SBATCH --error=logs/jepa_%A_%a.err
set -euo pipefail
cd /gpfs/u/scratch/EFCM/EFCMdvlr/jepa
mkdir -p logs
source .venv/bin/activate
export HF_HOME=/gpfs/u/scratch/EFCM/EFCMdvlr/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
BATCH_ROOT=experiments/main/${SLURM_ARRAY_JOB_ID:-local}
python scripts/run_main_experiments.py \
    --base-config configs/benchmark_cluster.yaml \
    --batch-root "$BATCH_ROOT" \
    --manifest-dir /gpfs/u/scratch/EFCM/EFCMdvlr/manifests \
    --seeds 1,2,3,4,5 --epochs 25 \
    --index "${SLURM_ARRAY_TASK_ID}"
