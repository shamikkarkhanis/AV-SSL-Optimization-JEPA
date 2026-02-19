"""Inference cost utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


def compute_energy_joules(runtime_ms: float, power_watts: float) -> float:
    """Estimate energy in joules from runtime and average power."""
    return (runtime_ms / 1000.0) * power_watts


def derive_summary_output_path(output_path: str) -> str:
    """Derive default summary path from a JSONL output path."""
    output = Path(output_path)
    if output.suffix == ".jsonl":
        return str(output.with_suffix(".summary.json"))
    return f"{output_path}.summary.json"


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
