"""Stage runners for train, score, evaluate, and run_experiment."""

from __future__ import annotations

import json
import logging
import platform
import random
import resource
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from jepa.config import resolve_config
from jepa.data import (
    MaskTubelet,
    TubeletDataset,
    VideoProcessor,
    load_clip_manifest,
    load_evaluation_labels,
)
from jepa.evaluation import compute_energy_joules, distribution_stats
from jepa.evaluation.ranking import compute_ranking_metrics, join_scores_and_labels
from jepa.models import JEPAModel
from jepa.training import Trainer

logger = logging.getLogger(__name__)


def _peak_memory_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system().lower() == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _ensure_run_layout(run_dir: Path, resolved_config: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("training", "scoring", "evaluation"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)


def _dataset_transform(config: Dict[str, Any]) -> MaskTubelet:
    dataset_cfg = config["dataset"]
    return MaskTubelet(
        mask_ratio=float(dataset_cfg["mask_ratio"]),
        patch_size=int(dataset_cfg["patch_size"]),
        seed=int(config.get("seed", 42)),
        frame_processor=VideoProcessor(size=int(dataset_cfg.get("image_size", 224))),
    )


def _build_loader(
    dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _build_stage_dataset(config: Dict[str, Any], stage: str) -> TubeletDataset:
    dataset_cfg = config["dataset"]
    manifest_key = {
        "train": "training_manifest",
        "validation": "validation_manifest",
        "score": "scoring_manifest",
        "evaluation": "evaluation_manifest",
    }[stage]
    split_key = {
        "train": "training_split",
        "validation": "validation_split",
        "score": "scoring_split",
        "evaluation": "evaluation_split",
    }[stage]
    return TubeletDataset(
        manifest_path=dataset_cfg[manifest_key],
        data_root=dataset_cfg.get("data_root"),
        tubelet_size=int(dataset_cfg["tubelet_size"]),
        transform=_dataset_transform(config),
        split=dataset_cfg.get(split_key),
        frames_per_clip=int(dataset_cfg.get("frames_per_clip", 16)),
    )


def _build_model(config: Dict[str, Any]) -> JEPAModel:
    model_cfg = config["model"]
    predictor_cfg = model_cfg.get("predictor", {})
    return JEPAModel(
        encoder_name=model_cfg.get("pretrained_name_or_path")
        or model_cfg["encoder_name"],
        predictor_hidden=int(
            predictor_cfg.get("hidden_dim", model_cfg.get("embedding_dim", 1024))
        ),
        predictor_dropout=float(predictor_cfg.get("dropout", 0.1)),
        freeze_encoder=model_cfg.get("encoder_mode", "frozen") == "frozen",
        encoder_init_mode=model_cfg.get("init_mode", "pretrained"),
    )


def _optimizer_params(model: JEPAModel, config: Dict[str, Any]):
    if config["model"].get("encoder_mode", "frozen") == "finetune":
        return model.parameters()
    return model.predictor.parameters()


def _amp_context(runtime_cfg: Dict[str, Any], device: torch.device):
    amp = runtime_cfg.get("amp", "none")
    if amp == "none":
        return nullcontext()
    if device.type == "cuda":
        dtype = torch.float16
        if amp in ("auto", "bf16") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device.type == "cpu" and amp == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def _effective_batch_size(config: Dict[str, Any], stage: str) -> int:
    runtime_cfg = config["runtime"]
    return int(runtime_cfg["batch_size_overrides"][stage]) * int(
        runtime_cfg.get("grad_accum_steps", 1)
    )


def _checkpoint_artifact_path(run_dir: Path) -> Path:
    return run_dir / "training" / "checkpoints" / "best_model.pt"


def _scores_artifact_path(run_dir: Path) -> Path:
    return run_dir / "scoring" / "scores.jsonl"


def train_stage(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    config = resolve_config(config)
    _set_seed(int(config.get("seed", 42)))
    _ensure_run_layout(run_dir, config)

    runtime_cfg = config["runtime"]
    train_cfg = config["train"]
    device = resolve_device(runtime_cfg.get("device", "auto"))
    if runtime_cfg.get("cpu_threads"):
        torch.set_num_threads(int(runtime_cfg["cpu_threads"]))

    dataset = _build_stage_dataset(config, "train")
    loader = _build_loader(
        dataset,
        batch_size=int(runtime_cfg["batch_size_overrides"]["train"]),
        shuffle=True,
        num_workers=int(runtime_cfg.get("num_workers", 0)),
        device=device,
    )
    val_dataset = _build_stage_dataset(config, "validation")
    val_loader = _build_loader(
        val_dataset,
        batch_size=int(runtime_cfg["batch_size_overrides"]["evaluation"]),
        shuffle=False,
        num_workers=int(runtime_cfg.get("num_workers", 0)),
        device=device,
    )

    model = _build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        _optimizer_params(model, config),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    start_epoch = 1
    best_val_loss = float("inf")
    resume_checkpoint = config["model"].get("resume_checkpoint")
    if config["model"].get("init_mode") == "resume":
        if not resume_checkpoint:
            raise ValueError(
                "model.resume_checkpoint is required for init_mode=resume."
            )
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("val_loss", best_val_loss))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=run_dir / "training" / "checkpoints",
        amp_context_factory=lambda: _amp_context(runtime_cfg, device),
        grad_accum_steps=int(runtime_cfg.get("grad_accum_steps", 1)),
        normalize_embeddings=bool(train_cfg.get("normalize_embeddings", True)),
        encoder_mode=config["model"].get("encoder_mode", "frozen"),
    )

    epoch_summaries: List[Dict[str, Any]] = []
    overall_start = time.perf_counter()
    epochs = int(train_cfg.get("epochs", 1))
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = trainer.train_epoch(loader, epoch)
        val_loss = trainer.validate_epoch(val_loader, epoch)
        duration = time.perf_counter() - epoch_start
        samples_per_sec = len(dataset) / duration if duration > 0 else 0.0
        is_best = val_loss <= best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        trainer.save_checkpoint(epoch, val_loss, is_best=is_best, config=config)
        epoch_summaries.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "wall_clock_seconds": duration,
                "samples_per_second": samples_per_sec,
            }
        )

    total_train_time = time.perf_counter() - overall_start
    checkpoint_path = _checkpoint_artifact_path(run_dir)
    target_metric = train_cfg.get("target_metric")
    convergence_epoch = None
    if target_metric is not None:
        for row in epoch_summaries:
            if row["val_loss"] <= float(target_metric):
                convergence_epoch = row["epoch"]
                break

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "runtime_profile": runtime_cfg.get("profile"),
        "effective_batch_size": _effective_batch_size(config, "train"),
        "peak_memory_mb": _peak_memory_mb(),
        "total_train_time_seconds": total_train_time,
        "epochs": epoch_summaries,
        "final_val_loss": epoch_summaries[-1]["val_loss"] if epoch_summaries else None,
        "best_val_loss": best_val_loss if epoch_summaries else None,
        "convergence_to_target_epoch": convergence_epoch,
        "device_info": {
            "torch_device": str(device),
            "cpu_threads": runtime_cfg.get("cpu_threads"),
            "cuda_available": torch.cuda.is_available(),
        },
    }
    with open(run_dir / "training" / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def score_stage(
    config: Dict[str, Any], run_dir: Path
) -> Dict[str, Any]:
    config = resolve_config(config)
    _ensure_run_layout(run_dir, config)

    runtime_cfg = config["runtime"]
    device = resolve_device(runtime_cfg.get("device", "auto"))
    if runtime_cfg.get("cpu_threads"):
        torch.set_num_threads(int(runtime_cfg["cpu_threads"]))

    checkpoint_path = _checkpoint_artifact_path(run_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = _build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = _build_stage_dataset(config, "score")
    loader = _build_loader(
        dataset,
        batch_size=int(runtime_cfg["batch_size_overrides"]["score"]),
        shuffle=False,
        num_workers=int(runtime_cfg.get("num_workers", 0)),
        device=device,
    )

    clip_scores: Dict[str, List[float]] = defaultdict(list)
    clip_cosine: Dict[str, List[float]] = defaultdict(list)
    clip_meta: Dict[str, Dict[str, Any]] = {}
    batch_latencies_ms: List[float] = []
    score_cfg = config["score"]
    warmup_batches = max(int(score_cfg.get("warmup_batches", 0)), 0)
    timed_start: Optional[float] = None
    timed_clip_count = 0

    aggregation = score_cfg.get("aggregation", "mean")
    if aggregation not in {"mean", "max"}:
        raise ValueError(
            f"Unsupported score.aggregation={aggregation!r}. Expected 'mean' or 'max'."
        )

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
                clean_emb, pred_emb = model(clean, masked, mask_frac)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if not is_warmup:
                per_sample_latency = elapsed_ms / max(len(clip_ids), 1)
                batch_latencies_ms.extend([per_sample_latency] * len(clip_ids))
                timed_clip_count += len(clip_ids)

            clean_np = clean_emb.float().cpu().numpy()
            pred_np = pred_emb.float().cpu().numpy()
            novelty = 1.0 - np.sum(
                (
                    pred_np
                    / np.maximum(np.linalg.norm(pred_np, axis=1, keepdims=True), 1e-12)
                )
                * (
                    clean_np
                    / np.maximum(np.linalg.norm(clean_np, axis=1, keepdims=True), 1e-12)
                ),
                axis=1,
            )

            pred_norm = pred_np / np.maximum(
                np.linalg.norm(pred_np, axis=1, keepdims=True), 1e-12
            )
            clean_norm = clean_np / np.maximum(
                np.linalg.norm(clean_np, axis=1, keepdims=True), 1e-12
            )
            cosine = np.sum(pred_norm * clean_norm, axis=1)

            for idx, clip_id in enumerate(clip_ids):
                clip_id = str(clip_id)
                clip_scores[clip_id].append(float(novelty[idx]))
                clip_cosine[clip_id].append(float(cosine[idx]))
                if clip_id not in clip_meta:
                    clip_meta[clip_id] = {
                        "clip_id": clip_id,
                        "scene_id": batch["meta"]["scene_id"][idx],
                        "camera": batch["meta"]["camera"][idx],
                        "split": batch["meta"]["split"][idx],
                    }

    score_rows = []
    for clip_id, values in clip_scores.items():
        if aggregation == "max":
            clip_score = max(values)
        else:
            clip_score = float(np.mean(values))
        row = {
            **clip_meta[clip_id],
            "review_value_score": clip_score,
            "tubelet_score_mean": float(np.mean(values)),
            "tubelet_score_std": float(np.std(values)),
            "tubelet_count": len(values),
            "mean_cosine_similarity": float(np.mean(clip_cosine[clip_id])),
        }
        score_rows.append(row)

    score_rows.sort(key=lambda row: row["review_value_score"], reverse=True)
    scores_path = _scores_artifact_path(run_dir)
    with open(scores_path, "w", encoding="utf-8") as handle:
        for row in score_rows:
            handle.write(json.dumps(row) + "\n")

    total_seconds = 0.0 if timed_start is None else time.perf_counter() - timed_start
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "scores_path": str(scores_path),
        "num_clips": len(score_rows),
        "runtime_profile": runtime_cfg.get("profile"),
        "device": str(device),
        "warmup_batches_skipped": warmup_batches,
        "timed_clips": timed_clip_count,
        "timed_window_seconds": total_seconds,
        "clips_per_second": timed_clip_count / total_seconds
        if total_seconds > 0
        else 0.0,
        "latency_ms": distribution_stats(batch_latencies_ms),
        "memory_peak_mb": _peak_memory_mb(),
        "estimated_energy_joules": compute_energy_joules(
            total_seconds * 1000.0, float(score_cfg.get("power_watts", 75.0))
        ),
        "forward_pass_seconds": sum(batch_latencies_ms) / 1000.0,
        "score_distribution": distribution_stats(
            [row["review_value_score"] for row in score_rows]
        ),
        "mean_cosine_similarity": float(
            np.mean([row["mean_cosine_similarity"] for row in score_rows])
        )
        if score_rows
        else 0.0,
    }
    with open(run_dir / "scoring" / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _load_score_rows(scores_path: Path) -> List[Dict[str, Any]]:
    with open(scores_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _restrict_scores_to_evaluation_split(
    config: Dict[str, Any],
    score_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    dataset_cfg = config["dataset"]
    manifest_path = dataset_cfg.get("evaluation_manifest")
    split = dataset_cfg.get("evaluation_split")
    if not manifest_path:
        return score_rows

    manifest_records = load_clip_manifest(manifest_path, split=split)
    allowed_clip_ids = {str(record["clip_id"]) for record in manifest_records}
    filtered = [row for row in score_rows if str(row.get("clip_id")) in allowed_clip_ids]
    return filtered


def evaluate_stage(
    config: Dict[str, Any], run_dir: Path
) -> Dict[str, Any]:
    config = resolve_config(config)
    _ensure_run_layout(run_dir, config)
    evaluation_cfg = config["evaluation"]
    labels_path = config["dataset"].get("evaluation_labels")
    if not labels_path:
        raise ValueError("dataset.evaluation_labels is required for evaluation.")

    scores_path = _scores_artifact_path(run_dir)
    if scores_path.resolve() == Path(labels_path).resolve():
        raise ValueError("scores_path and dataset.evaluation_labels must point to different files.")
    score_rows = _restrict_scores_to_evaluation_split(
        config,
        _load_score_rows(scores_path),
    )
    if not score_rows:
        raise ValueError("No scored clips matched dataset.evaluation_manifest and dataset.evaluation_split.")
    labels = load_evaluation_labels(labels_path)
    joined = join_scores_and_labels(
        score_rows,
        labels,
        score_key="review_value_score",
        label_key=evaluation_cfg.get("primary_label_field", "binary_label"),
    )
    if not joined:
        raise ValueError(
            "No evaluation rows remained after joining scores with labels. "
            "Check clip_id alignment, evaluation split filtering, and label coverage."
        )
    ranking = compute_ranking_metrics(
        joined,
        k_values=list(evaluation_cfg.get("k_values", [10, 50, 100])),
        include_medium_as_positive=bool(
            evaluation_cfg.get("include_medium_as_positive", True)
        ),
        graded_mapping=dict(evaluation_cfg.get("graded_labels", {})),
    )
    mean_cosine_similarity = (
        float(np.mean([row.get("mean_cosine_similarity", 0.0) for row in joined]))
        if joined
        else 0.0
    )
    results = {
        "scores_path": str(scores_path),
        "labels_path": labels_path,
        "ranking_metrics": ranking,
        "model_health": {
            "mean_cosine_similarity": mean_cosine_similarity,
        },
    }
    with open(run_dir / "evaluation" / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def write_root_summary(
    run_dir: Path,
    training: Optional[Dict[str, Any]],
    scoring: Optional[Dict[str, Any]],
    evaluation: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    summary = {
        "run_dir": str(run_dir),
        "artifacts": {
            "config_resolved": str(run_dir / "config_resolved.yaml"),
            "training": str(run_dir / "training" / "summary.json"),
            "scoring": str(run_dir / "scoring" / "summary.json"),
            "evaluation": str(run_dir / "evaluation" / "summary.json"),
        },
        "training": training,
        "scoring": scoring,
        "evaluation": evaluation,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def run_experiment(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    training = train_stage(config, run_dir)
    scoring = score_stage(config, run_dir)
    evaluation = evaluate_stage(config, run_dir)
    return write_root_summary(run_dir, training, scoring, evaluation)
