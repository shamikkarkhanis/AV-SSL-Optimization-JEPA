# JEPA Clip Mining Pipeline

Production-ready implementation of Joint Embedding Predictive Architecture (JEPA) for autonomous driving clip mining.

## Features

- **Frozen VJEPA Encoder**: Uses `facebook/vjepa2-vitl-fpc64-256` backbone
- **Efficient Training**: Trains only a lightweight predictor head (2-layer MLP)
- **Masked Prediction**: Predicts clean embeddings from masked inputs (75% masking)
- **Novelty Scoring**: Scores clips based on reconstruction difficulty (1 - cosine similarity)
- **Colab Ready**: Includes notebook for GPU training on Google Colab

## 📂 Project Structure

```
src/jepa/               # Core package
├── data/               # JSONL dataset loader & masking transforms
├── models/             # VJEPA encoder & predictor architecture
├── training/           # Training loop & normalized L1 loss
└── evaluation/         # Cosine similarity metrics

scripts/                # CLI utilities
├── eval.py             # CPU evaluation script
└── inference.py        # CPU inference/scoring script

configs/                # Configuration
└── default.yaml        # Research hyperparameters

notebooks/
└── train_jepa.ipynb    # Google Colab training notebook
```

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/jepa-production.git
cd jepa-production

# Install package in editable mode
pip install -e .

# Install dev dependencies (optional)
pip install -e ".[dev]"
```

## Training (Google Colab)

1. Upload `notebooks/train_jepa.ipynb` to Google Colab
2. Mount Google Drive
3. Run cells to install package and start training
4. Checkpoints saved to `experiments/checkpoints/`

**Hyperparameters (Default):**
- Epochs: 30
- Batch Size: 8
- Learning Rate: 1e-4 (AdamW)
- Mask Ratio: 0.75
- Tubelet Size: 2 frames

## Evaluation (Local CPU)

Run evaluation on a test manifest:

```bash
python scripts/eval.py \
  --checkpoint experiments/checkpoints/best_model.pt \
  --manifest data/manifests/clips_manifest.jsonl \
  --output eval_results.json
```

## ⛏️ Inference / Mining

Score clips to identify novel/hard scenarios:

```bash
python scripts/inference.py \
  --checkpoint experiments/checkpoints/best_model.pt \
  --manifest new_clips.jsonl \
  --output scores.jsonl
```

Output `scores.jsonl` contains novelty scores (higher = more novel):
```json
{"score": 0.45, "scene": "scene-001", "camera": "CAM_FRONT", "tubelet_idx": 0}
```

## Testing

Run smoke tests:
```bash
pytest tests/
```
