"""Configuration loading and runtime profile resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _legacy_to_v1(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "dataset" in raw and "train" in raw and "score" in raw:
        return deepcopy(raw)

    data = raw.get("data", {})
    model = raw.get("model", {})
    training = raw.get("training", {})
    evaluation = raw.get("evaluation", {})
    inference = raw.get("inference", {})
    paper = raw.get("paper", {})

    return {
        "seed": raw.get("seed", 42),
        "dataset": {
            "data_root": data.get("data_root"),
            "frames_per_clip": 16,
            "tubelet_size": data.get("tubelet_size", 2),
            "image_size": data.get("image_size", 224),
            "patch_size": data.get("patch_size", 16),
            "mask_ratio": data.get("mask_ratio", 0.75),
            "training_manifest": data.get("manifest_path"),
            "training_split": "train",
            "validation_manifest": data.get("manifest_path"),
            "validation_split": "val",
            "scoring_manifest": data.get("manifest_path"),
            "scoring_split": "val",
            "evaluation_manifest": data.get("manifest_path"),
            "evaluation_split": "test",
            "evaluation_labels": None,
        },
        "model": {
            "encoder_name": model.get("encoder_name", "facebook/vjepa2-vitl-fpc64-256"),
            "embedding_dim": model.get("embedding_dim", 1024),
            "predictor": deepcopy(model.get("predictor", {})),
            "init_mode": "pretrained",
            "encoder_mode": "frozen" if model.get("freeze_encoder", True) else "finetune",
            "pretrained_name_or_path": model.get("encoder_name"),
            "resume_checkpoint": None,
        },
        "train": {
            "epochs": training.get("epochs", 1),
            "optimizer": training.get("optimizer", "adamw"),
            "lr": training.get("lr", 1e-4),
            "weight_decay": training.get("weight_decay", 0.01),
            "loss": training.get("loss", "l1"),
            "normalize_embeddings": training.get("normalize_embeddings", True),
            "checkpoint_every": training.get("checkpoint_every", 1),
            "target_metric": evaluation.get("target_similarity"),
        },
        "score": {
            "aggregation": "mean",
            "power_watts": inference.get("power_watts", 75.0),
            "warmup_batches": inference.get("warmup_batches", 1),
            "amp": inference.get("amp", "auto"),
        },
        "evaluation": {
            "k_values": [10, 50, 100],
            "primary_label_field": "binary_label",
            "positive_labels": ["high_value"],
            "include_medium_as_positive": True,
            "graded_labels": {"low_value": 0.0, "medium_value": 1.0, "high_value": 2.0},
            "health_metrics": ["mean_cosine_similarity", "validation_loss"],
        },
        "runtime": {
            "profile": "gpu",
            "device": training.get("device", inference.get("device", "auto")),
            "amp": inference.get("amp", "auto"),
            "num_workers": data.get("num_workers", 0),
            "cpu_threads": inference.get("cpu_threads"),
            "batch_size_overrides": {
                "train": data.get("batch_size", 8),
                "score": inference.get("batch_size", 1),
                "evaluation": evaluation.get("batch_size", 8),
            },
            "grad_accum_steps": 1,
            "profiles": {
                "cpu": {"device": "cpu", "amp": "none", "num_workers": 0},
                "gpu": {"device": "auto", "amp": inference.get("amp", "auto")},
                "ddp": {"device": "cuda", "amp": "auto"},
            },
        },
        "experiment": {
            "name": paper.get("default_hypothesis", "default"),
            "output_root": "experiments/runs",
            "description": "",
        },
    }


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _legacy_to_v1(raw)


def apply_runtime_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = deepcopy(config)
    runtime = resolved.setdefault("runtime", {})
    profile_name = runtime.get("profile", "gpu")
    profiles = runtime.get("profiles", {})
    profile_overlay = profiles.get(profile_name, {})
    runtime.update(deep_merge(runtime, profile_overlay))
    return resolved


def resolve_config(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = apply_runtime_profile(config)
    runtime = resolved.setdefault("runtime", {})
    dataset = resolved.setdefault("dataset", {})
    dataset.setdefault("validation_manifest", dataset.get("evaluation_manifest"))
    dataset.setdefault("validation_split", dataset.get("evaluation_split", "val"))
    dataset.setdefault("evaluation_manifest", dataset.get("scoring_manifest"))
    dataset.setdefault("evaluation_split", dataset.get("scoring_split", "val"))
    batch_overrides = runtime.setdefault("batch_size_overrides", {})
    batch_overrides.setdefault("train", 1)
    batch_overrides.setdefault("score", 1)
    batch_overrides.setdefault("evaluation", batch_overrides["score"])
    runtime.setdefault("grad_accum_steps", 1)
    runtime.setdefault("amp", "none")
    runtime.setdefault("num_workers", 0)
    return resolved


def load_and_resolve_config(path: str | Path, override: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config = load_config(path)
    if override:
        config = deep_merge(config, override)
    return resolve_config(config)
