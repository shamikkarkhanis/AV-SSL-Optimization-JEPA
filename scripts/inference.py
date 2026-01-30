"""JEPA Inference/Scoring Script (CPU-friendly)"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from jepa.data import TubeletDataset, MaskTubelet
from jepa.models import JEPAModel
from jepa.evaluation import compute_novelty_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")


def score_clips(
    checkpoint_path: str,
    manifest_path: str,
    output_path: str,
    data_root: Optional[str] = None,
    batch_size: int = 1,  # Keep small for inference safety
    device: str = "cpu",
):
    """Run inference scoring."""
    device_obj = torch.device(device)
    logger.info(f"Using device: {device_obj}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    config = checkpoint["config"]

    # Resolve data_root: CLI arg > config > None
    resolved_data_root = data_root or config.get("data", {}).get("data_root")
    if resolved_data_root:
        logger.info(f"Using data_root: {resolved_data_root}")

    # Initialize model
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
    mask_transform = MaskTubelet(
        mask_ratio=config["data"]["mask_ratio"],
        patch_size=config["data"]["patch_size"],
        seed=42,
    )

    # Load dataset
    dataset = TubeletDataset(
        manifest_path=manifest_path,
        data_root=resolved_data_root,
        tubelet_size=config["data"]["tubelet_size"],
        transform=mask_transform,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Scoring loop
    logger.info(f"Scoring {len(dataset)} tubelets...")

    # Open output file for streaming writes (JSONL)
    with open(output_path, "w") as f:
        with torch.no_grad():
            for batch in tqdm(loader):
                masked = batch["masked_frames"].to(device_obj)
                clean = batch["clean_frames"].to(device_obj)
                mask_frac = batch["mask_frac"].to(device_obj)

                # Get embeddings
                clean_emb, pred_emb = model(clean, masked, mask_frac)

                # Compute novelty score (1 - cosine_similarity)
                clean_np = clean_emb.cpu().numpy()
                pred_np = pred_emb.cpu().numpy()
                scores = compute_novelty_score(pred_np, clean_np)

                # Write results
                metas = batch["meta"]  # Dict of lists
                batch_size_curr = len(scores)

                for i in range(batch_size_curr):
                    # Reconstruct metadata for this sample
                    meta_sample = {
                        k: v[i] for k, v in metas.items() if isinstance(v, list)
                    }

                    # Add non-list metadata if simple types
                    for k, v in metas.items():
                        if not isinstance(v, list) and k not in meta_sample:
                            if isinstance(v, torch.Tensor):
                                meta_sample[k] = v[i].item()
                            else:
                                meta_sample[k] = v[i]

                    result = {
                        "score": float(scores[i]),
                        "scene": meta_sample.get("scene", "unknown"),
                        "camera": meta_sample.get("camera", "unknown"),
                        "tubelet_idx": int(meta_sample.get("tubelet_idx", 0)),
                    }

                    f.write(json.dumps(result) + "\n")

    logger.info(f"Scoring complete. Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Score Clips with JEPA")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--manifest", required=True, help="Path to clip manifest")
    parser.add_argument("--output", default="scores.jsonl", help="Output JSONL path")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root directory for data files (overrides config)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")

    args = parser.parse_args()

    score_clips(
        args.checkpoint,
        args.manifest,
        args.output,
        args.data_root,
        args.batch_size,
        args.device,
    )


if __name__ == "__main__":
    main()
