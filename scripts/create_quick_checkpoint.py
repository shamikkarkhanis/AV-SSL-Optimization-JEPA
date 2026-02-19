"""Create a quick JEPA checkpoint for script validation."""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from jepa.models import JEPAModel


def create_checkpoint(output_path: str, config_path: str) -> None:
    with open(config_path, "r") as f:
        # Keep dependencies minimal: config file is simple YAML-like but valid JSON keys.
        import yaml  # Local import to avoid importing unless needed.

        config = yaml.safe_load(f)

    model = JEPAModel(
        encoder_name=config["model"]["encoder_name"],
        predictor_hidden=config["model"]["predictor"]["hidden_dim"],
        predictor_dropout=config["model"]["predictor"]["dropout"],
        freeze_encoder=True,
    )

    checkpoint = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "val_loss": float("inf"),
        "config": config,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(json.dumps({"checkpoint": str(output.resolve())}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a quick JEPA checkpoint")
    parser.add_argument(
        "--output",
        default="experiments/checkpoints/quick_checkpoint.pt",
        help="Output checkpoint path",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Config YAML path",
    )
    args = parser.parse_args()

    create_checkpoint(args.output, args.config)


if __name__ == "__main__":
    main()
