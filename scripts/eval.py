"""JEPA Evaluation Script (CPU-friendly)"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from jepa.data import JEPADataset, TubeletDataset, MaskTubelet
from jepa.models import JEPAModel
from jepa.evaluation import compute_cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval")


def eval_model(
    checkpoint_path: str,
    manifest_path: str,
    output_path: str,
    data_root: Optional[str] = None,
    batch_size: int = 8,
    device: str = "cpu",
):
    """Run evaluation."""
    device_obj = torch.device(device)
    logger.info(f"Using device: {device_obj}")

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    config = checkpoint["config"]

    # Resolve data_root: CLI arg > config > None
    resolved_data_root = data_root or config.get("data", {}).get("data_root")
    if resolved_data_root:
        logger.info(f"Using data_root: {resolved_data_root}")

    # Initialize model
    logger.info("Initializing model...")
    model = JEPAModel(
        encoder_name=config["model"]["encoder_name"],
        predictor_hidden=config["model"]["predictor"]["hidden_dim"],
        predictor_dropout=config["model"]["predictor"]["dropout"],
        freeze_encoder=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    model.eval()

    # Prepare data
    logger.info(f"Loading data from {manifest_path}")
    mask_transform = MaskTubelet(
        mask_ratio=config["data"]["mask_ratio"],
        patch_size=config["data"]["patch_size"],
        seed=42,  # Fixed seed for eval
    )

    dataset = TubeletDataset(
        manifest_path=manifest_path,
        data_root=resolved_data_root,
        tubelet_size=config["data"]["tubelet_size"],
        transform=mask_transform,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Evaluation loop
    logger.info("Starting evaluation...")
    all_pred = []
    all_target = []

    with torch.no_grad():
        for batch in tqdm(loader):
            masked = batch["masked_frames"].to(device_obj)
            clean = batch["clean_frames"].to(device_obj)
            mask_frac = batch["mask_frac"].to(device_obj)

            clean_emb, pred_emb = model(clean, masked, mask_frac)

            all_pred.append(pred_emb.cpu().numpy())
            all_target.append(clean_emb.cpu().numpy())

    # Concatenate results
    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    # Compute metric
    logger.info("Computing metrics...")
    mean_sim = compute_cosine_similarity(all_pred, all_target)

    results = {
        "mean_cosine_similarity": mean_sim,
        "checkpoint": checkpoint_path,
        "num_samples": len(dataset),
        "epoch": checkpoint["epoch"],
        "val_loss": checkpoint["val_loss"],
    }

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation complete. Mean Cosine Similarity: {mean_sim:.4f}")
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate JEPA Model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--manifest", required=True, help="Path to test manifest")
    parser.add_argument(
        "--output", default="eval_results.json", help="Output JSON path"
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root directory for data files (overrides config)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")

    args = parser.parse_args()

    eval_model(
        args.checkpoint,
        args.manifest,
        args.output,
        args.data_root,
        args.batch_size,
        args.device,
    )


if __name__ == "__main__":
    main()
