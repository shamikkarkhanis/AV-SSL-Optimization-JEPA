#!/usr/bin/env bash
#SBATCH --job-name=jepa-ftc
#SBATCH --array=0-24            # 25 plan items (20 training + 5 baselines); see --dry-run
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00         # generous; sweeps_32 @ ~25 epochs is the long pole
#SBATCH --output=logs/jepa_%A_%a.out
#SBATCH --error=logs/jepa_%A_%a.err
# ---- EDIT to match your CCI allocation ----
###SBATCH --partition=el8
###SBATCH --account=YOUR_ACCOUNT
#
# One array task == one plan item == one GPU run. All tasks share a batch root
# keyed by the array job id; each writes results_<index>.jsonl (no append races).
#
# Submit:  sbatch scripts/cluster/run_experiments.slurm
# Then:    python scripts/analyze_main_experiments.py --results experiments/main/<JOBID>
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
mkdir -p logs

# --- environment (EDIT for your cluster) ---
# module load python cuda            # if your cluster uses modules
source .venv/bin/activate            # or your conda env

# --- offline: cluster nodes have no internet, so HF must read a pre-staged cache ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$REPO/.hf_cache}"   # must already contain the V-JEPA2 weights

BATCH_ROOT="experiments/main/${SLURM_ARRAY_JOB_ID:-local}"
MANIFEST_DIR="${MANIFEST_DIR:-/gpfs/u/home/EFCM/EFCMdvlr/barn-shared/data/manifests}"

python scripts/run_main_experiments.py \
    --base-config configs/benchmark.yaml \
    --batch-root "$BATCH_ROOT" \
    --manifest-dir "$MANIFEST_DIR" \
    --seeds 1,2,3,4,5 \
    --epochs 25 \
    --index "${SLURM_ARRAY_TASK_ID}"
