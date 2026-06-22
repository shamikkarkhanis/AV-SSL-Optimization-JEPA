#!/usr/bin/env bash
# Pre-stage the V-JEPA2 encoder weights so the no-internet compute nodes can load
# them offline. Run this ONCE somewhere with internet, then make sure HF_HOME
# points at the resulting cache on the cluster.
#
# Option 1: run on a CCI LOGIN node if it has outbound internet.
# Option 2: run on your laptop, then rsync the cache to the cluster:
#     rsync -av ~/.cache/huggingface/ EFCMdvlr@blp01.cci.rpi.edu:~/AV-SSL-Optimization-JEPA/.hf_cache/
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_HOME="${HF_HOME:-$REPO/.hf_cache}"
PY="${PY:-python}"

echo "Downloading facebook/vjepa2-vitl-fpc64-256 into $HF_HOME ..."
$PY - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("facebook/vjepa2-vitl-fpc64-256")
print("Cached at:", path)
PY

echo "Verifying offline load..."
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY - <<'PY'
from transformers import AutoModel
m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
print("Offline load OK:", sum(p.numel() for p in m.parameters()), "params")
PY

echo "Done. On the cluster set:  export HF_HOME=$HF_HOME"
