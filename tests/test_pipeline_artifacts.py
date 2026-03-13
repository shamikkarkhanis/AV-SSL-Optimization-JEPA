"""Tests for run artifact layout and summary structure."""

import json
from pathlib import Path

from jepa.pipeline import write_root_summary


def test_write_root_summary_creates_expected_files(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "training").mkdir(parents=True)
    (run_dir / "scoring").mkdir(parents=True)
    (run_dir / "evaluation").mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text("seed: 1\n", encoding="utf-8")

    training = {
        "checkpoint_path": "best_model.pt",
        "peak_memory_mb": 1.0,
        "device_info": {},
        "effective_batch_size": 2,
        "total_train_time_seconds": 0.1,
        "epochs": [],
    }
    scoring = {
        "scores_path": "scores.jsonl",
        "clips_per_second": 10.0,
        "latency_ms": {"mean": 1.0, "p50": 1.0, "p95": 1.0},
        "memory_peak_mb": 2.0,
        "estimated_energy_joules": 3.0,
    }
    evaluation = {
        "ranking_metrics": {"average_precision": 1.0},
        "model_health": {"mean_cosine_similarity": 0.5},
    }

    summary = write_root_summary(run_dir, training, scoring, evaluation)
    assert (run_dir / "summary.json").exists()
    parsed = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert parsed["artifacts"]["config_resolved"].endswith("config_resolved.yaml")
    assert "training" in summary
    assert "scoring" in summary
    assert "evaluation" in summary
