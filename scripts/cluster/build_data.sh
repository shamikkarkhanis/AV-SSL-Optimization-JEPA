#!/usr/bin/env bash
# Build the four training manifests + the fixed evaluation benchmark on the cluster.
#
# Run ONCE from a node that can read the frame data (login or interactive). Edit
# the *_ROOT variables to match your account, then `bash scripts/cluster/build_data.sh`.
#
# Training manifests are built with --absolute so they are self-contained (no
# dataset.data_root needed). The benchmark is merged from the committed labeled
# manifests; each source needs a --root so its relative frame paths become absolute.
set -euo pipefail

# ---- EDIT THESE -----------------------------------------------------------
BARN="/gpfs/u/home/EFCM/EFCMdvlr/barn-shared"     # your shared scratch
SAMPLES_ROOT="$BARN/data/raw/v1.0-trainval/samples"   # contains CAM_FRONT/*.jpg
SWEEPS_ROOT="$BARN/data/raw/v1.0-trainval/sweeps"     # contains CAM_FRONT/*.jpg
TRAINVAL_ROOT="$BARN/data/raw/v1.0-trainval"          # nuScenes root for benchmark
WAYMO_ROOT="$BARN/data/raw/waymo_test"                # Waymo frames root
BDD_ROOT="$BARN/data/raw/bdd100k_test"                # BDD frames root
OUT="$BARN/data/manifests"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-python}"   # set PY=./.venv/bin/python if not using a module
# ---------------------------------------------------------------------------

mkdir -p "$OUT"
cd "$REPO"

echo "== Building 4 training manifests (absolute paths) =="
$PY scripts/build_manifest.py --dataroot "$SAMPLES_ROOT" --window 16 --stride 16 \
    --absolute --output "$OUT/nuscenes_samples_16.jsonl"
$PY scripts/build_manifest.py --dataroot "$SAMPLES_ROOT" --window 32 --stride 32 \
    --absolute --output "$OUT/nuscenes_samples_32.jsonl"
$PY scripts/build_manifest.py --dataroot "$SWEEPS_ROOT" --window 16 --stride 16 \
    --absolute --output "$OUT/nuscenes_sweeps_16.jsonl"
$PY scripts/build_manifest.py --dataroot "$SWEEPS_ROOT" --window 32 --stride 32 \
    --absolute --output "$OUT/nuscenes_sweeps_32.jsonl"

echo "== Merging the fixed evaluation benchmark =="
# Uses the committed labeled manifests + label files in data/manifests.
$PY scripts/merge_benchmark.py \
    --manifest data/manifests/nuscenes_trainval.jsonl "$TRAINVAL_ROOT" \
    --manifest data/manifests/waymo_test.jsonl "$WAYMO_ROOT" \
    --manifest data/manifests/bdd100k_test.jsonl "$BDD_ROOT" \
    --labels data/manifests/nuscenes_trainval_evaluation_labels.jsonl \
    --labels data/manifests/waymo_test_evaluation_labels.jsonl \
    --labels data/manifests/bdd100k_test_evaluation_labels.jsonl \
    --output-manifest "$OUT/benchmark_test.jsonl" \
    --output-labels "$OUT/benchmark_test_evaluation_labels.jsonl" \
    --require-label

echo
echo "Done. Now point configs/benchmark.yaml (or --manifest-dir / paths) at:"
echo "  training manifests: $OUT/nuscenes_{samples,sweeps}_{16,32}.jsonl"
echo "  benchmark:          $OUT/benchmark_test.jsonl"
echo "  benchmark labels:   $OUT/benchmark_test_evaluation_labels.jsonl"
