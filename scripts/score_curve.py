"""Score + evaluate every (or every Nth) epoch checkpoint to get AP-vs-epoch.

Addresses the "train to convergence / show training curves" reviewer concern.
A run trained with checkpoint_every=1 keeps training/checkpoints/checkpoint_epoch_*.pt;
this scores each on the fixed benchmark and pairs it with the epoch's validation
loss (from training/summary.json) so you can plot AP and val-loss vs epoch.

Usage:
    python scripts/score_curve.py --config configs/benchmark.yaml \
        --run-dir experiments/main/run1/runs/samples_32__seed1 --every 2
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jepa.config import deep_merge, load_and_resolve_config
from jepa.pipeline import evaluate_stage, score_stage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("score_curve")

_EPOCH_RE = re.compile(r"checkpoint_epoch_(\d+)\.pt$")


def _epoch_checkpoints(run_dir: Path) -> List[tuple[int, Path]]:
    ckpt_dir = run_dir / "training" / "checkpoints"
    found = []
    for path in sorted(ckpt_dir.glob("checkpoint_epoch_*.pt")):
        m = _EPOCH_RE.search(path.name)
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found)


def _val_loss_by_epoch(run_dir: Path) -> Dict[int, float]:
    summary_path = run_dir / "training" / "summary.json"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text())
    return {int(e["epoch"]): float(e["val_loss"]) for e in summary.get("epochs", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an AP-vs-epoch curve for one run.")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--every", type=int, default=1, help="Score every Nth epoch checkpoint.")
    parser.add_argument("--set", action="append", default=[], help="Inline JSON config overrides.")
    args = parser.parse_args()

    config = load_and_resolve_config(args.config)
    for override_json in args.set:
        config = deep_merge(config, json.loads(override_json))

    run_dir = Path(args.run_dir)
    checkpoints = _epoch_checkpoints(run_dir)
    if not checkpoints:
        raise SystemExit(f"No epoch checkpoints found under {run_dir}/training/checkpoints")
    val_losses = _val_loss_by_epoch(run_dir)

    curve_root = run_dir / "curve"
    curve_root.mkdir(parents=True, exist_ok=True)
    points: List[Dict[str, Optional[float]]] = []

    for epoch, ckpt in checkpoints:
        if (epoch - 1) % args.every != 0 and epoch != checkpoints[-1][0]:
            continue
        epoch_dir = curve_root / f"epoch_{epoch:03d}"
        score_stage(config, epoch_dir, checkpoint_path=ckpt)
        evaluation = evaluate_stage(config, epoch_dir)
        ranking = evaluation.get("ranking_metrics", {})
        point = {
            "epoch": epoch,
            "val_loss": val_losses.get(epoch),
            "average_precision": ranking.get("average_precision"),
            "ndcg": ranking.get("ndcg"),
            "precision_at_10": ranking.get("precision_at_k", {}).get("10"),
        }
        points.append(point)
        logger.info("epoch %d: AP=%s val_loss=%s", epoch, point["average_precision"], point["val_loss"])

    curve_path = run_dir / "training_curve.json"
    with open(curve_path, "w") as f:
        json.dump({"run_dir": str(run_dir), "points": points}, f, indent=2)
    logger.info("Wrote %d curve points to %s", len(points), curve_path)


if __name__ == "__main__":
    main()
