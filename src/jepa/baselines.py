"""No-training baselines for the 'is the JEPA predictor actually helping?' question.

Each baseline produces a ``review_value_score`` per clip in the *same* format as
``pipeline.score_stage`` so that ``pipeline.evaluate_stage`` consumes the output
unchanged. All baselines reuse the frozen V-JEPA encoder; none of them require a
trained predictor checkpoint.

Methods
-------
masked_gap (A)
    Prediction-free floor. novelty = 1 - cos(encoder(masked), encoder(clean)).
    Measures how much masking perturbs the raw encoder embedding. No predictor,
    no training of any kind. The trivial zero-cost reference point.
embedding_density (B)
    Standard embedding-space novelty / anomaly detection. novelty = 1 - mean
    cosine similarity to a clip's k nearest neighbours within the evaluation pool
    (leave-one-out). No masking, no prediction. This is the conventional reading
    of "raw pretrained V-JEPA embedding novelty".
untrained_predictor (C)
    Sanity check. The full scoring pipeline but with the predictor left at its
    random initialisation (no training loop). Isolates whether training the head
    matters versus the architecture alone.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from jepa.config import resolve_config
from jepa.evaluation import distribution_stats

logger = logging.getLogger(__name__)

BASELINE_METHODS = ("masked_gap", "embedding_density", "untrained_predictor")


# --------------------------------------------------------------------------- #
# Pure scoring math (numpy only -- unit-testable without the encoder or data)  #
# --------------------------------------------------------------------------- #
def _l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), eps)
    return embeddings / norms


def masked_clean_novelty(clean: np.ndarray, masked: np.ndarray) -> np.ndarray:
    """Baseline A: row-wise ``1 - cos(masked, clean)`` (one value per tubelet)."""
    clean = np.asarray(clean, dtype=np.float64)
    masked = np.asarray(masked, dtype=np.float64)
    cos = np.sum(_l2_normalize(masked) * _l2_normalize(clean), axis=1)
    return 1.0 - cos


def embedding_density_novelty(clip_embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    """Baseline B: leave-one-out kNN novelty in cosine space (one value per clip).

    For each clip the novelty is ``1 - mean(cosine to its k nearest neighbours)``.
    Isolated clips (far from the crowd) score high; clips in a dense cluster of
    similar clips score low. Fully determined by the encoder and the evaluation
    pool, so it is independent of any training run.
    """
    embeddings = _l2_normalize(np.asarray(clip_embeddings, dtype=np.float64))
    n = embeddings.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)  # exclude self (leave-one-out)
    effective_k = min(k, n - 1)
    # top-k neighbour similarities per row
    neighbour_sims = np.sort(similarity, axis=1)[:, -effective_k:]
    return 1.0 - neighbour_sims.mean(axis=1)


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #
def score_baseline_stage(
    config: Dict[str, Any],
    run_dir: Path,
    method: str,
    k: int = 5,
) -> Dict[str, Any]:
    """Run a no-training baseline over the scoring split and write scores.jsonl.

    Output layout matches ``pipeline.score_stage`` (``scoring/scores.jsonl`` and
    ``scoring/summary.json``) so the standard ``evaluate_stage`` can be run next.
    """
    # Imported lazily so the pure functions above stay importable without torch
    # heavy pipeline dependencies during unit tests.
    from jepa.pipeline import (
        _amp_context,
        _build_loader,
        _build_model,
        _build_stage_dataset,
        _ensure_run_layout,
        _peak_memory_mb,
        _scores_artifact_path,
        resolve_device,
    )

    if method not in BASELINE_METHODS:
        raise ValueError(
            f"Unknown baseline method {method!r}. Expected one of {BASELINE_METHODS}."
        )

    config = resolve_config(config)
    _ensure_run_layout(run_dir, config)

    runtime_cfg = config["runtime"]
    score_cfg = config["score"]
    device = resolve_device(runtime_cfg.get("device", "auto"))
    if runtime_cfg.get("cpu_threads"):
        torch.set_num_threads(int(runtime_cfg["cpu_threads"]))

    # No checkpoint is loaded. For untrained_predictor the predictor stays at its
    # random init; for masked_gap / embedding_density the predictor is unused.
    _set_baseline_seed(int(config.get("seed", 42)))
    model = _build_model(config).to(device)
    model.eval()

    dataset = _build_stage_dataset(config, "score")
    loader = _build_loader(
        dataset,
        batch_size=int(runtime_cfg["batch_size_overrides"]["score"]),
        shuffle=False,
        num_workers=int(runtime_cfg.get("num_workers", 0)),
        device=device,
    )

    # Per-clip accumulators.
    clip_tubelet_novelty: Dict[str, List[float]] = defaultdict(list)
    clip_tubelet_cosine: Dict[str, List[float]] = defaultdict(list)
    clip_clean_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    clip_meta: Dict[str, Dict[str, Any]] = {}
    batch_latencies_ms: List[float] = []
    warmup_batches = max(int(score_cfg.get("warmup_batches", 0)), 0)
    timed_start: Optional[float] = None
    timed_clip_count = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            clip_ids = batch["meta"]["clip_id"]
            masked = batch["masked_frames"].to(device)
            clean = batch["clean_frames"].to(device)
            mask_frac = batch["mask_frac"].to(device)

            is_warmup = batch_idx < warmup_batches
            if not is_warmup and timed_start is None:
                timed_start = time.perf_counter()

            start = time.perf_counter()
            with _amp_context(runtime_cfg, device):
                clean_emb = model.encoder.get_embedding(clean)
                if method in ("masked_gap", "untrained_predictor"):
                    masked_emb = model.encoder.get_embedding(masked)
                    if method == "untrained_predictor":
                        pred_emb = model.predictor(masked_emb, mask_frac)
                    else:
                        pred_emb = masked_emb  # identity: no predictor at all
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if not is_warmup:
                per_sample_latency = elapsed_ms / max(len(clip_ids), 1)
                batch_latencies_ms.extend([per_sample_latency] * len(clip_ids))
                timed_clip_count += len(clip_ids)

            clean_np = clean_emb.float().cpu().numpy()
            if method in ("masked_gap", "untrained_predictor"):
                pred_np = pred_emb.float().cpu().numpy()
                novelty = masked_clean_novelty(clean_np, pred_np)
                cosine = 1.0 - novelty
            else:  # embedding_density: per-clip, computed after the full pass
                novelty = None
                cosine = None

            for idx, clip_id in enumerate(clip_ids):
                clip_id = str(clip_id)
                clip_clean_embeddings[clip_id].append(clean_np[idx])
                if novelty is not None:
                    clip_tubelet_novelty[clip_id].append(float(novelty[idx]))
                    clip_tubelet_cosine[clip_id].append(float(cosine[idx]))
                if clip_id not in clip_meta:
                    clip_meta[clip_id] = {
                        "clip_id": clip_id,
                        "scene_id": batch["meta"]["scene_id"][idx],
                        "camera": batch["meta"]["camera"][idx],
                        "split": batch["meta"]["split"][idx],
                    }

    # Per-clip mean clean embedding (used by embedding_density and reported health).
    ordered_clip_ids = list(clip_clean_embeddings.keys())
    clip_mean_embeddings = {
        clip_id: np.mean(np.stack(embs, axis=0), axis=0)
        for clip_id, embs in clip_clean_embeddings.items()
    }

    if method == "embedding_density":
        matrix = np.stack([clip_mean_embeddings[cid] for cid in ordered_clip_ids], axis=0)
        clip_density_novelty = embedding_density_novelty(matrix, k=k)
        density_by_clip = dict(zip(ordered_clip_ids, clip_density_novelty))

    score_rows = []
    for clip_id in ordered_clip_ids:
        if method == "embedding_density":
            clip_score = float(density_by_clip[clip_id])
            tubelet_values = [clip_score]
            mean_cos = float("nan")
        else:
            tubelet_values = clip_tubelet_novelty[clip_id]
            clip_score = float(np.mean(tubelet_values))
            mean_cos = float(np.mean(clip_tubelet_cosine[clip_id]))
        row = {
            **clip_meta[clip_id],
            "review_value_score": clip_score,
            "tubelet_score_mean": float(np.mean(tubelet_values)),
            "tubelet_score_std": float(np.std(tubelet_values)),
            "tubelet_count": len(tubelet_values),
            "mean_cosine_similarity": mean_cos,
            "baseline_method": method,
        }
        score_rows.append(row)

    score_rows.sort(key=lambda row: row["review_value_score"], reverse=True)
    scores_path = _scores_artifact_path(run_dir)
    with open(scores_path, "w", encoding="utf-8") as handle:
        for row in score_rows:
            handle.write(json.dumps(row) + "\n")

    total_seconds = 0.0 if timed_start is None else time.perf_counter() - timed_start
    summary = {
        "baseline_method": method,
        "k_neighbours": k if method == "embedding_density" else None,
        "checkpoint_path": None,
        "scores_path": str(scores_path),
        "num_clips": len(score_rows),
        "runtime_profile": runtime_cfg.get("profile"),
        "device": str(device),
        "warmup_batches_skipped": warmup_batches,
        "timed_clips": timed_clip_count,
        "timed_window_seconds": total_seconds,
        "clips_per_second": timed_clip_count / total_seconds if total_seconds > 0 else 0.0,
        "latency_ms": distribution_stats(batch_latencies_ms),
        "memory_peak_mb": _peak_memory_mb(),
        "score_distribution": distribution_stats(
            [row["review_value_score"] for row in score_rows]
        ),
    }
    with open(run_dir / "scoring" / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _set_baseline_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
