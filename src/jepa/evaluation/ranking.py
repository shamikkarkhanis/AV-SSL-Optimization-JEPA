"""Ranking metrics for clip review-value evaluation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.metrics import auc, average_precision_score, ndcg_score, precision_recall_curve


def _label_to_binary(label: Any, include_medium_as_positive: bool = True) -> int:
    if isinstance(label, (bool, np.bool_)):
        return int(label)
    if isinstance(label, (int, float, np.integer, np.floating)):
        return int(label > 0)
    if label == "high_value":
        return 1
    if include_medium_as_positive and label == "medium_value":
        return 1
    return 0


def join_scores_and_labels(
    scores: Iterable[Dict[str, Any]],
    labels: Iterable[Dict[str, Any]],
    score_key: str = "review_value_score",
    label_key: str = "binary_label",
) -> List[Dict[str, Any]]:
    score_map: Dict[str, Dict[str, Any]] = {}
    duplicate_score_ids = set()
    for row in scores:
        clip_id = str(row["clip_id"])
        if clip_id in score_map:
            duplicate_score_ids.add(clip_id)
        score_map[clip_id] = row
    if duplicate_score_ids:
        duplicate_ids = ", ".join(sorted(duplicate_score_ids))
        raise ValueError(f"Duplicate score rows found for clip_id(s): {duplicate_ids}")

    joined: List[Dict[str, Any]] = []
    seen_label_ids = set()
    duplicate_label_ids = set()
    for label in labels:
        clip_id = str(label["clip_id"])
        if clip_id in seen_label_ids:
            duplicate_label_ids.add(clip_id)
            continue
        seen_label_ids.add(clip_id)
        if clip_id not in score_map:
            continue
        primary_label = label.get(label_key)
        if primary_label is None:
            primary_label = label.get("binary_label")
        if primary_label is None:
            primary_label = label.get("adjudicated_label")
        if primary_label is None:
            primary_label = label.get("review_value")
        joined.append(
            {
                **score_map[clip_id],
                **label,
                "score": float(score_map[clip_id][score_key]),
                "label": primary_label,
            }
        )
    if duplicate_label_ids:
        duplicate_ids = ", ".join(sorted(duplicate_label_ids))
        raise ValueError(f"Duplicate label rows found for clip_id(s): {duplicate_ids}")
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
        raise ValueError("Cannot compute ranking metrics with zero joined score/label rows.")

    scores = np.asarray([row["score"] for row in joined_rows], dtype=np.float64)
    binary = np.asarray(
        [_label_to_binary(row.get("label"), include_medium_as_positive) for row in joined_rows],
        dtype=np.int32,
    )
    graded_values = []
    for row in joined_rows:
        label = row.get("label")
        if isinstance(label, (bool, np.bool_)):
            graded_values.append(float(int(label)))
        elif isinstance(label, (int, float, np.integer, np.floating)):
            graded_values.append(float(label))
        else:
            graded_values.append(float(graded_mapping.get(str(label), 0.0)))
    graded = np.asarray(graded_values, dtype=np.float64)

    if binary.sum() == 0:
        average_precision = 0.0
        pr_auc = 0.0
    else:
        average_precision = float(average_precision_score(binary, scores))
        precision, recall, _ = precision_recall_curve(binary, scores)
        pr_auc = float(auc(recall, precision))

    ndcg = (
        float(ndcg_score(graded.reshape(1, -1), scores.reshape(1, -1)))
        if np.any(graded > 0) and len(joined_rows) > 1
        else 0.0
    )

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
