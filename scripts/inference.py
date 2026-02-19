"""JEPA Inference/Scoring Script (CPU-friendly)."""

import argparse
import json
import logging
import sys
import time
import resource
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from jepa.data import TubeletDataset, MaskTubelet
from jepa.models import JEPAModel
from jepa.evaluation import (
    compute_novelty_score,
    compute_energy_joules,
    derive_summary_output_path,
    distribution_stats,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")


def _get_peak_rss_mb() -> float:
    """Return process peak RSS in MB."""
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KB
    if sys.platform == "darwin":
        return rss_kb / (1024.0 * 1024.0)
    return rss_kb / 1024.0


def _extract_meta_sample(metas: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Extract per-sample metadata from a batched metadata dict."""
    meta_sample: Dict[str, Any] = {}

    for key, value in metas.items():
        if isinstance(value, list):
            meta_sample[key] = value[idx]
        elif isinstance(value, torch.Tensor):
            val = value[idx]
            meta_sample[key] = val.item() if val.ndim == 0 else val.tolist()

    return meta_sample


def score_clips(
    checkpoint_path: str,
    manifest_path: str,
    output_path: str,
    data_root: Optional[str] = None,
    batch_size: int = 1,
    device: str = "cpu",
    power_watts: float = 75.0,
    summary_output: Optional[str] = None,
    warmup_batches: int = 1,
    no_cost_metrics: bool = False,
):
    """Run inference scoring with optional cost telemetry."""
    device_obj = torch.device(device)
    logger.info(f"Using device: {device_obj}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    config = checkpoint["config"]

    # Resolve data_root: CLI arg > config > None
    resolved_data_root = data_root or config.get("data", {}).get("data_root")
    if resolved_data_root:
        logger.info(f"Using data_root: {resolved_data_root}")

    # Initialize model
    model = JEPAModel(
        encoder_name=config["model"]["encoder_name"],
        predictor_hidden=config["model"]["predictor"]["hidden_dim"],
        predictor_dropout=config["model"]["predictor"]["dropout"],
        freeze_encoder=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    model.eval()

    # Prepare data
    mask_transform = MaskTubelet(
        mask_ratio=config["data"]["mask_ratio"],
        patch_size=config["data"]["patch_size"],
        seed=42,
    )

    dataset = TubeletDataset(
        manifest_path=manifest_path,
        data_root=resolved_data_root,
        tubelet_size=config["data"]["tubelet_size"],
        transform=mask_transform,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Scoring {len(dataset)} tubelets...")

    if summary_output is None:
        summary_output = derive_summary_output_path(output_path)

    runtime_ms_values = []
    energy_joules_values = []
    batch_mem_delta_values = []
    baseline_peak_rss_mb = _get_peak_rss_mb()
    prev_peak_rss_mb = baseline_peak_rss_mb

    with open(output_path, "w") as f:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader)):
                masked = batch["masked_frames"].to(device_obj)
                clean = batch["clean_frames"].to(device_obj)
                mask_frac = batch["mask_frac"].to(device_obj)

                start = time.perf_counter()
                clean_emb, pred_emb = model(clean, masked, mask_frac)
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                clean_np = clean_emb.cpu().numpy()
                pred_np = pred_emb.cpu().numpy()
                scores = compute_novelty_score(pred_np, clean_np)

                batch_size_curr = len(scores)
                runtime_per_sample_ms = elapsed_ms / max(batch_size_curr, 1)
                energy_per_sample_j = compute_energy_joules(
                    runtime_per_sample_ms, power_watts
                )

                current_peak_rss_mb = _get_peak_rss_mb()
                batch_peak_delta_mb = max(0.0, current_peak_rss_mb - prev_peak_rss_mb)
                prev_peak_rss_mb = current_peak_rss_mb
                memory_overhead_per_sample_mb = batch_peak_delta_mb / max(batch_size_curr, 1)

                if not no_cost_metrics and batch_idx >= warmup_batches:
                    runtime_ms_values.extend([runtime_per_sample_ms] * batch_size_curr)
                    energy_joules_values.extend([energy_per_sample_j] * batch_size_curr)
                    batch_mem_delta_values.append(batch_peak_delta_mb)

                metas = batch["meta"]
                for i in range(batch_size_curr):
                    meta_sample = _extract_meta_sample(metas, i)
                    result: Dict[str, Any] = {
                        "score": float(scores[i]),
                        "scene": meta_sample.get("scene", "unknown"),
                        "camera": meta_sample.get("camera", "unknown"),
                        "tubelet_idx": int(meta_sample.get("tubelet_idx", 0)),
                    }
                    if not no_cost_metrics:
                        result.update(
                            {
                                "runtime_ms": float(runtime_per_sample_ms),
                                "energy_joules": float(energy_per_sample_j),
                                "memory_overhead_mb": float(memory_overhead_per_sample_mb),
                            }
                        )

                    f.write(json.dumps(result) + "\n")

    logger.info(f"Scoring complete. Results saved to {output_path}")

    if no_cost_metrics:
        logger.info("Cost telemetry disabled; skipping summary output.")
        return

    final_peak_rss_mb = _get_peak_rss_mb()
    peak_rss_delta_mb = max(0.0, final_peak_rss_mb - baseline_peak_rss_mb)

    summary = {
        "num_samples": len(dataset),
        "batch_size": batch_size,
        "device": str(device_obj),
        "power_watts": float(power_watts),
        "checkpoint": checkpoint_path,
        "manifest": manifest_path,
        "output_jsonl": output_path,
        "runtime_ms": distribution_stats(runtime_ms_values),
        "energy_joules": distribution_stats(energy_joules_values),
        "memory_overhead_mb": {
            "peak_rss_delta": float(peak_rss_delta_mb),
            "mean_per_batch_delta": float(
                sum(batch_mem_delta_values) / max(len(batch_mem_delta_values), 1)
            ),
        },
    }

    with open(summary_output, "w") as sf:
        json.dump(summary, sf, indent=2)

    logger.info(f"Cost summary saved to {summary_output}")


def main():
    parser = argparse.ArgumentParser(description="Score Clips with JEPA")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--manifest", required=True, help="Path to clip manifest")
    parser.add_argument("--output", default="scores.jsonl", help="Output JSONL path")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root directory for data files (overrides config)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument(
        "--power-watts",
        type=float,
        default=75.0,
        help="Average device power in watts for energy estimation",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Path for run summary JSON (default: derived from --output)",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Number of initial batches to exclude from summary stats",
    )
    parser.add_argument(
        "--no-cost-metrics",
        action="store_true",
        help="Disable runtime/energy/memory telemetry output",
    )

    args = parser.parse_args()

    score_clips(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_path=args.output,
        data_root=args.data_root,
        batch_size=args.batch_size,
        device=args.device,
        power_watts=args.power_watts,
        summary_output=args.summary_output,
        warmup_batches=max(0, args.warmup_batches),
        no_cost_metrics=args.no_cost_metrics,
    )


if __name__ == "__main__":
    main()
