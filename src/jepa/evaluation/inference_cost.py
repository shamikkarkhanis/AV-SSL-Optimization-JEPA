"""Inference cost utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


def compute_energy_joules(runtime_ms: float, power_watts: float) -> float:
    """Estimate energy in joules from runtime and average power."""
    return (runtime_ms / 1000.0) * power_watts


def derive_summary_output_path(
    output_path: str,
    experiments_root: str = "experiments",
    task_name: str = "inference_runs",
) -> str:
    """Build a timestamped summary path under experiments for longitudinal tracking."""
    output = Path(output_path)
    run_label = output.stem if output.stem else "run"
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_label)

    now = datetime.now()
    day_bucket = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H%M%S")

    run_dir = Path(experiments_root) / task_name / day_bucket / f"{timestamp}_{safe_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir / "summary.json")


def distribution_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of numeric values."""
    if not values:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
            "total": 0.0,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "total": float(arr.sum()),
    }
