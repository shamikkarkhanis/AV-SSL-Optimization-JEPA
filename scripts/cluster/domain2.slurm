#!/usr/bin/env bash
#SBATCH --job-name=jepa-domain2
#SBATCH --partition=npl-2024
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/u/scratch/EFCM/EFCMdvlr/jepa/logs/domain2_%j.out
#SBATCH --error=/gpfs/u/scratch/EFCM/EFCMdvlr/jepa/logs/domain2_%j.out
set -uo pipefail
cd /gpfs/u/scratch/EFCM/EFCMdvlr/jepa
source .venv/bin/activate
export HF_HOME=/gpfs/u/scratch/EFCM/EFCMdvlr/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1
python scripts/domain_confound_test2.py
