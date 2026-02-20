"""End-to-end paper experiment runner.

Single entrypoint with one runtime flag: --hypothesis.
All other parameters are sourced from configs/default.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from jepa.data import TubeletDataset, MaskTubelet
from jepa.evaluation import compute_cosine_similarity
from jepa.models import JEPAModel
from jepa.training import Trainer
from inference import score_clips


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paper_run")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_device(device_value: str) -> torch.device:
    if device_value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_value)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_datasets(config: Dict[str, Any]) -> Tuple[Subset, Subset]:
    data_cfg = config["data"]
    transform = MaskTubelet(
        mask_ratio=data_cfg["mask_ratio"],
        patch_size=data_cfg["patch_size"],
        seed=config["seed"],
    )
    full_dataset = TubeletDataset(
        manifest_path=data_cfg["manifest_path"],
        data_root=data_cfg.get("data_root"),
        tubelet_size=data_cfg["tubelet_size"],
        transform=transform,
    )

    n_samples = len(full_dataset)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(config["seed"])
    rng.shuffle(indices)

    train_size = int(n_samples * float(data_cfg["train_split"]))
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()
    if len(val_indices) == 0:
        raise ValueError("Validation split is empty; lower train_split in config.")

    return Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)


def _build_loader(
    dataset: Subset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    pin_memory = device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _run_training(config: Dict[str, Any], run_root: Path) -> Path:
    train_cfg = config["training"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    device = _resolve_device(train_cfg["device"])
    logger.info("Training device: %s", device)

    train_subset, val_subset = _build_datasets(config)
    num_workers = int(data_cfg.get("num_workers", 0))
    train_loader = _build_loader(
        train_subset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=bool(data_cfg["shuffle_train"]),
        num_workers=num_workers,
        device=device,
    )
    val_loader = _build_loader(
        val_subset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )

    model = JEPAModel(
        encoder_name=model_cfg["encoder_name"],
        predictor_hidden=int(model_cfg["predictor"]["hidden_dim"]),
        predictor_dropout=float(model_cfg["predictor"]["dropout"]),
        freeze_encoder=bool(model_cfg["freeze_encoder"]),
    ).to(device)

    if train_cfg["optimizer"].lower() != "adamw":
        raise ValueError("Only AdamW optimizer is currently supported.")
    optimizer = torch.optim.AdamW(
        model.predictor.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    checkpoint_dir = run_root / "training" / "checkpoints"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    best_val = float("inf")
    best_path = checkpoint_dir / "best_model.pt"
    epochs = int(train_cfg["epochs"])

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate_epoch(val_loader, epoch)
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
        trainer.save_checkpoint(
            epoch=epoch,
            val_loss=val_loss,
            is_best=is_best,
            config=config,
        )
        logger.info(
            "Epoch %d/%d complete: train_loss=%.4f val_loss=%.4f",
            epoch,
            epochs,
            train_loss,
            val_loss,
        )

    if not best_path.exists():
        raise FileNotFoundError(f"Expected best checkpoint at {best_path}")
    return best_path


def _run_evaluation(
    config: Dict[str, Any],
    checkpoint_path: Path,
    run_root: Path,
) -> Path:
    eval_cfg = config["evaluation"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    infer_cfg = config["inference"]

    device = _resolve_device(infer_cfg["device"])
    logger.info("Evaluation device: %s", device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = JEPAModel(
        encoder_name=model_cfg["encoder_name"],
        predictor_hidden=int(model_cfg["predictor"]["hidden_dim"]),
        predictor_dropout=float(model_cfg["predictor"]["dropout"]),
        freeze_encoder=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = MaskTubelet(
        mask_ratio=data_cfg["mask_ratio"],
        patch_size=data_cfg["patch_size"],
        seed=config["seed"],
    )
    dataset = TubeletDataset(
        manifest_path=data_cfg["manifest_path"],
        data_root=data_cfg.get("data_root"),
        tubelet_size=data_cfg["tubelet_size"],
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(eval_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    all_pred = []
    all_target = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Eval"):
            masked = batch["masked_frames"].to(device, non_blocking=device.type == "cuda")
            clean = batch["clean_frames"].to(device, non_blocking=device.type == "cuda")
            mask_frac = batch["mask_frac"].to(device, non_blocking=device.type == "cuda")
            clean_emb, pred_emb = model(clean, masked, mask_frac)
            all_pred.append(pred_emb.float().cpu().numpy())
            all_target.append(clean_emb.float().cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    mean_sim = compute_cosine_similarity(pred, target)

    output_path = run_root / "evaluation" / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "mean_cosine_similarity": float(mean_sim),
                "checkpoint": str(checkpoint_path),
                "num_samples": len(dataset),
                "epoch": int(checkpoint.get("epoch", -1)),
                "val_loss": float(checkpoint.get("val_loss", 0.0)),
            },
            f,
            indent=2,
        )
    return output_path


def _run_inference(config: Dict[str, Any], checkpoint_path: Path, run_root: Path) -> Tuple[Path, Path]:
    infer_cfg = config["inference"]
    data_cfg = config["data"]

    inference_dir = run_root / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    scores_path = inference_dir / "scores.jsonl"
    summary_path = inference_dir / "summary.json"

    score_clips(
        checkpoint_path=str(checkpoint_path),
        manifest_path=data_cfg["manifest_path"],
        output_path=str(scores_path),
        data_root=data_cfg.get("data_root"),
        batch_size=int(infer_cfg["batch_size"]),
        device=str(infer_cfg["device"]),
        power_watts=float(infer_cfg.get("power_watts", 75.0)),
        summary_output=str(summary_path),
        warmup_batches=int(infer_cfg.get("warmup_batches", 1)),
        no_cost_metrics=bool(infer_cfg.get("no_cost_metrics", False)),
        num_workers=infer_cfg.get("num_workers"),
        amp=str(infer_cfg.get("amp", "auto")),
        cpu_threads=infer_cfg.get("cpu_threads"),
    )
    return scores_path, summary_path


def _load_config(hypothesis: str) -> Dict[str, Any]:
    config_path = Path("configs/default.yaml")
    with open(config_path, "r") as f:
        base = yaml.safe_load(f)

    hypotheses = base.get("paper", {}).get("hypotheses", {})
    if hypothesis not in hypotheses:
        available = ", ".join(sorted(hypotheses.keys()))
        raise KeyError(f"Unknown hypothesis '{hypothesis}'. Available: {available}")

    overrides = hypotheses[hypothesis].get("overrides", {})
    return _deep_merge(base, overrides)


def _create_run_root(config: Dict[str, Any], hypothesis: str) -> Path:
    experiments_root = Path(config["paper"]["experiments_root"])
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    ts = now.strftime("%H%M%S")
    safe_hypothesis = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in hypothesis
    )
    run_root = experiments_root / date_str / f"{ts}_{safe_hypothesis}"
    (run_root / "training").mkdir(parents=True, exist_ok=True)
    (run_root / "evaluation").mkdir(parents=True, exist_ok=True)
    (run_root / "inference").mkdir(parents=True, exist_ok=True)
    return run_root


def _append_index(config: Dict[str, Any], run_record: Dict[str, Any]) -> None:
    index_path = Path(config["paper"]["experiments_root"]) / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "a") as f:
        f.write(json.dumps(run_record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full JEPA paper pipeline")
    parser.add_argument("--hypothesis", required=True, help="Configured hypothesis key")
    args = parser.parse_args()

    config = _load_config(args.hypothesis)
    _set_seed(int(config["seed"]))

    run_root = _create_run_root(config, args.hypothesis)
    logger.info("Run root: %s", run_root)

    with open(run_root / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    checkpoint_path = _run_training(config, run_root)
    eval_path = _run_evaluation(config, checkpoint_path, run_root)
    scores_path, summary_path = _run_inference(config, checkpoint_path, run_root)

    run_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hypothesis": args.hypothesis,
        "run_root": str(run_root),
        "checkpoint": str(checkpoint_path),
        "evaluation_json": str(eval_path),
        "inference_scores_jsonl": str(scores_path),
        "inference_summary_json": str(summary_path),
    }

    with open(run_root / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    _append_index(config, run_summary)

    logger.info("Paper run completed.")
    logger.info("Summary: %s", run_root / "run_summary.json")


if __name__ == "__main__":
    main()
