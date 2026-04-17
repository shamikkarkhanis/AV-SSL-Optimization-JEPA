"""Tests for manifest normalization and ranking evaluation."""

import json
from pathlib import Path

import pytest

from jepa.data import derive_clip_id, load_clip_manifest, load_evaluation_labels
from jepa.evaluation import compute_ranking_metrics, join_scores_and_labels


def test_manifest_loader_derives_stable_clip_id_and_split(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"
    record = {
        "frames": [f"frame_{idx}.jpg" for idx in range(16)],
        "scene": "scene-1",
        "camera": "front",
    }
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    records = load_clip_manifest(manifest_path)
    assert len(records) == 1
    assert records[0]["clip_id"] == derive_clip_id(record)
    assert records[0]["split"] == "unspecified"


def test_label_join_and_ranking_metrics():
    scores = [
        {"clip_id": "a", "review_value_score": 0.9, "mean_cosine_similarity": 0.2},
        {"clip_id": "b", "review_value_score": 0.8, "mean_cosine_similarity": 0.3},
        {"clip_id": "c", "review_value_score": 0.1, "mean_cosine_similarity": 0.9},
    ]
    labels = [
        {"clip_id": "a", "binary_label": 1},
        {"clip_id": "b", "binary_label": 1},
        {"clip_id": "c", "binary_label": 0},
    ]
    joined = join_scores_and_labels(scores, labels)
    metrics = compute_ranking_metrics(joined, k_values=[1, 2, 3])

    assert len(joined) == 3
    assert metrics["average_precision"] > 0.9
    assert metrics["precision_at_k"]["1"] == 1.0
    assert metrics["recall_at_k"]["2"] == 1.0


def test_load_evaluation_labels_jsonl(tmp_path: Path):
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        json.dumps({"clip_id": "clip-1", "binary_label": 1}) + "\n",
        encoding="utf-8",
    )
    labels = load_evaluation_labels(labels_path)
    assert labels[0]["clip_id"] == "clip-1"
    assert labels[0]["binary_label"] == 1


def test_join_scores_and_labels_rejects_duplicate_labels():
    scores = [{"clip_id": "a", "review_value_score": 0.9}]
    labels = [
        {"clip_id": "a", "binary_label": 1},
        {"clip_id": "a", "binary_label": 0},
    ]

    with pytest.raises(ValueError, match="Duplicate label rows"):
        join_scores_and_labels(scores, labels)


def test_join_scores_and_labels_rejects_duplicate_scores():
    scores = [
        {"clip_id": "a", "review_value_score": 0.9},
        {"clip_id": "a", "review_value_score": 0.8},
    ]
    labels = [{"clip_id": "a", "binary_label": 1}]

    with pytest.raises(ValueError, match="Duplicate score rows"):
        join_scores_and_labels(scores, labels)


def test_compute_ranking_metrics_rejects_empty_join():
    with pytest.raises(ValueError, match="zero joined"):
        compute_ranking_metrics([])
