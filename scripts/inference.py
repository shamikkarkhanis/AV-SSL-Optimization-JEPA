"""JEPA Inference/Scoring Script (CPU-friendly)."""

import argparse
import json
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    import psutil
except ImportError:
    psutil = None
import resource

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


def _append_run_index(summary: Dict[str, Any], summary_output: str) -> None:
    """Append lightweight run metadata for chronological experiment tracking."""
    summary_path = Path(summary_output)
    if "experiments/inference_runs" not in summary_path.as_posix():
        return

    index_path = Path("experiments") / "inference_runs" / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary_json": str(summary_path),
        "output_jsonl": summary.get("output_jsonl"),
        "num_samples": summary.get("num_samples"),
        "device": summary.get("device"),
        "runtime_ms_mean": summary.get("runtime_ms", {}).get("mean"),
        "energy_joules_total": summary.get("energy_joules", {}).get("total"),
    }
    with open(index_path, "a") as index_file:
        index_file.write(json.dumps(index_entry) + "\n")


def _peak_ru_maxrss_mb() -> float:
    """Return process ru_maxrss in MB (peak RSS from OS accounting)."""
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KB
    if sys.platform == "darwin":
        return rss_kb / (1024.0 * 1024.0)
    return rss_kb / 1024.0


def _get_current_rss_mb(process: Optional["psutil.Process"]) -> float:
    """Return current process RSS in MB, with ru_maxrss fallback."""
    if process is not None:
        return process.memory_info().rss / (1024.0 * 1024.0)
    return _peak_ru_maxrss_mb()


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
    num_workers: Optional[int] = None,
    amp: str = "auto",
    cpu_threads: Optional[int] = None,
):
    """Run inference scoring with optional cost telemetry."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device_obj = torch.device(device)
    logger.info(f"Using device: {device_obj}")

    if cpu_threads is not None and cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        logger.info(f"Using CPU threads: {cpu_threads}")

    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        if device_obj.type == "mps":
            num_workers = 0
        elif sys.platform == "darwin" and device_obj.type == "cpu":
            # Avoid torch_shm_manager failures in macOS sandboxed environments.
            num_workers = 0
        else:
            num_workers = min(max(cpu_count - 1, 0), 8)

    pin_memory = device_obj.type == "cuda"
    non_blocking = device_obj.type == "cuda"
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

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

    loader = DataLoader(dataset, **loader_kwargs)

    logger.info(f"Scoring {len(dataset)} tubelets with num_workers={num_workers}...")

    if summary_output is None:
        summary_output = derive_summary_output_path(output_path)
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)

    amp_dtype: Optional[torch.dtype] = None
    amp_device_type: Optional[str] = None
    if amp != "none":
        if device_obj.type == "cuda":
            amp_device_type = "cuda"
            amp_dtype = torch.float16
            if amp in ("auto", "bf16") and torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            elif amp == "fp16":
                amp_dtype = torch.float16
        elif device_obj.type == "cpu" and amp == "bf16":
            amp_device_type = "cpu"
            amp_dtype = torch.bfloat16

    process = psutil.Process() if psutil is not None else None
    if process is None:
        logger.warning(
            "psutil is not installed; memory overhead falls back to ru_maxrss (coarse)."
        )

    runtime_ms_values = []
    energy_joules_values = []
    batch_mem_positive_delta_values = []
    batch_mem_signed_delta_values = []
    baseline_current_rss_mb = _get_current_rss_mb(process)
    max_current_rss_mb = baseline_current_rss_mb

    with open(output_path, "w") as f:
        with torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(loader)):
                rss_before_batch_mb = _get_current_rss_mb(process)

                masked = batch["masked_frames"].to(device_obj, non_blocking=non_blocking)
                clean = batch["clean_frames"].to(device_obj, non_blocking=non_blocking)
                mask_frac = batch["mask_frac"].to(device_obj, non_blocking=non_blocking)

                start = time.perf_counter()
                amp_context = (
                    torch.autocast(device_type=amp_device_type, dtype=amp_dtype)
                    if amp_device_type is not None and amp_dtype is not None
                    else nullcontext()
                )
                with amp_context:
                    clean_emb, pred_emb = model(clean, masked, mask_frac)
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                clean_np = clean_emb.float().cpu().numpy()
                pred_np = pred_emb.float().cpu().numpy()
                scores = compute_novelty_score(pred_np, clean_np)

                batch_size_curr = len(scores)
                runtime_per_sample_ms = elapsed_ms / max(batch_size_curr, 1)
                energy_per_sample_j = compute_energy_joules(
                    runtime_per_sample_ms, power_watts
                )

                rss_after_batch_mb = _get_current_rss_mb(process)
                max_current_rss_mb = max(max_current_rss_mb, rss_after_batch_mb)
                batch_signed_delta_mb = rss_after_batch_mb - rss_before_batch_mb
                batch_positive_delta_mb = max(0.0, batch_signed_delta_mb)
                memory_overhead_per_sample_mb = batch_positive_delta_mb / max(
                    batch_size_curr, 1
                )

                if not no_cost_metrics and batch_idx >= warmup_batches:
                    runtime_ms_values.extend([runtime_per_sample_ms] * batch_size_curr)
                    energy_joules_values.extend([energy_per_sample_j] * batch_size_curr)
                    batch_mem_positive_delta_values.append(batch_positive_delta_mb)
                    batch_mem_signed_delta_values.append(batch_signed_delta_mb)

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

    peak_rss_delta_mb = max(0.0, max_current_rss_mb - baseline_current_rss_mb)

    summary = {
        "num_samples": len(dataset),
        "batch_size": batch_size,
        "device": str(device_obj),
        "power_watts": float(power_watts),
        "checkpoint": checkpoint_path,
        "manifest": manifest_path,
        "output_jsonl": output_path,
        "summary_json": summary_output,
        "runtime_ms": distribution_stats(runtime_ms_values),
        "energy_joules": distribution_stats(energy_joules_values),
        "memory_overhead_mb": {
            "peak_rss_delta": float(peak_rss_delta_mb),
            "mean_per_batch_delta": float(
                sum(batch_mem_positive_delta_values)
                / max(len(batch_mem_positive_delta_values), 1)
            ),
            "mean_per_batch_signed_delta": float(
                sum(batch_mem_signed_delta_values)
                / max(len(batch_mem_signed_delta_values), 1)
            ),
        },
        "memory_metric_source": "psutil_rss" if process is not None else "ru_maxrss",
    }

    with open(summary_output, "w") as sf:
        json.dump(summary, sf, indent=2)

    _append_run_index(summary, summary_output)
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
    parser.add_argument(
        "--device",
        default="auto",
        help="Device (auto/cpu/cuda/mps). auto prefers cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: auto-tuned per device)",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Override torch CPU thread count",
    )
    parser.add_argument(
        "--amp",
        default="auto",
        choices=["auto", "none", "fp16", "bf16"],
        help="Automatic mixed precision mode",
    )
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
        num_workers=args.num_workers,
        amp=args.amp,
        cpu_threads=args.cpu_threads,
    )


if __name__ == "__main__":
    main()
