"""Tests for inference cost utility functions."""

from jepa.evaluation.inference_cost import (
    compute_energy_joules,
    derive_summary_output_path,
    distribution_stats,
)


def test_compute_energy_joules():
    runtime_ms = 200.0
    power_watts = 75.0
    assert compute_energy_joules(runtime_ms, power_watts) == 15.0


def test_derive_summary_output_path_jsonl():
    assert derive_summary_output_path("scores.jsonl") == "scores.summary.json"


def test_derive_summary_output_path_non_jsonl():
    assert derive_summary_output_path("scores.out") == "scores.out.summary.json"


def test_distribution_stats_values():
    values = [10.0, 20.0, 30.0, 40.0]
    stats = distribution_stats(values)

    assert stats["mean"] == 25.0
    assert stats["total"] == 100.0
    assert stats["min"] == 10.0
    assert stats["max"] == 40.0
    assert stats["p50"] == 25.0


def test_distribution_stats_empty():
    stats = distribution_stats([])
    assert stats == {
        "mean": 0.0,
        "p50": 0.0,
        "p95": 0.0,
        "min": 0.0,
        "max": 0.0,
        "total": 0.0,
    }
