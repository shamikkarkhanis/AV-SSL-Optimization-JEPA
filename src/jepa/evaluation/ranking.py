"""Ranking metrics for clip review-value evaluation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.metrics import auc, average_precision_score, ndcg_score, precision_recall_curve


def _label_to_binary(label: str | None, include_medium_as_positive: bool = True) -> int:
    if label == "high_value":
        return 1
    if include_medium_as_positive and label == "medium_value":
        return 1
    return 0


def join_scores_and_labels(
    scores: Iterable[Dict[str, Any]],
    labels: Iterable[Dict[str, Any]],
    score_key: str = "review_value_score",
    label_key: str = "adjudicated_label",
) -> List[Dict[str, Any]]:
    score_map = {str(row["clip_id"]): row for row in scores}
    joined: List[Dict[str, Any]] = []
    for label in labels:
        clip_id = str(label["clip_id"])
        if clip_id not in score_map:
            continue
        joined.append({**score_map[clip_id], **label, "score": float(score_map[clip_id][score_key]), "label": label.get(label_key) or label.get("review_value")})
    return joined


def compute_precision_recall_at_k(
    binary_labels: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> Dict[str, float]:
    k = min(k, len(scores))
    if k == 0:
        return {"precision": 0.0, "recall": 0.0}
    order = np.argsort(-scores)
    topk = binary_labels[order][:k]
    total_positive = int(binary_labels.sum())
    hits = int(topk.sum())
    precision = hits / float(k)
    recall = hits / float(total_positive) if total_positive else 0.0
    return {"precision": precision, "recall": recall}


def compute_ranking_metrics(
    joined_rows: List[Dict[str, Any]],
    k_values: List[int] | None = None,
    include_medium_as_positive: bool = True,
    graded_mapping: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    k_values = k_values or [10, 50, 100]
    graded_mapping = graded_mapping or {
        "low_value": 0.0,
        "medium_value": 1.0,
        "high_value": 2.0,
    }
    if not joined_rows:
        return {
            "num_joined_clips": 0,
            "average_precision": 0.0,
            "pr_auc": 0.0,
            "ndcg": 0.0,
            "precision_at_k": {},
            "recall_at_k": {},
        }

    scores = np.asarray([row["score"] for row in joined_rows], dtype=np.float64)
    binary = np.asarray(
        [_label_to_binary(row.get("label"), include_medium_as_positive) for row in joined_rows],
        dtype=np.int32,
    )
    graded = np.asarray(
        [graded_mapping.get(str(row.get("label")), 0.0) for row in joined_rows],
        dtype=np.float64,
    )

    if binary.sum() == 0:
        average_precision = 0.0
        pr_auc = 0.0
    else:
        average_precision = float(average_precision_score(binary, scores))
        precision, recall, _ = precision_recall_curve(binary, scores)
        pr_auc = float(auc(recall, precision))

    ndcg = float(ndcg_score(graded.reshape(1, -1), scores.reshape(1, -1))) if np.any(graded > 0) else 0.0

    precision_at_k = {}
    recall_at_k = {}
    for k in k_values:
        metrics = compute_precision_recall_at_k(binary, scores, k)
        precision_at_k[str(k)] = metrics["precision"]
        recall_at_k[str(k)] = metrics["recall"]

    return {
        "num_joined_clips": len(joined_rows),
        "positive_rate": float(binary.mean()),
        "average_precision": average_precision,
        "pr_auc": pr_auc,
        "ndcg": ndcg,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
    }
