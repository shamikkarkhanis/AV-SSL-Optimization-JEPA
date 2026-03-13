# JEPA Clip Mining Pipeline

Production-ready implementation of Joint Embedding Predictive Architecture (JEPA) for autonomous driving clip mining.

## Features

- **Frozen VJEPA Encoder**: Uses `facebook/vjepa2-vitl-fpc64-256` backbone
- **Efficient Training**: Trains only a lightweight predictor head (2-layer MLP)
- **Masked Prediction**: Predicts clean embeddings from masked inputs (75% masking)
- **Novelty Scoring**: Scores clips based on reconstruction difficulty (1 - cosine similarity)
- **Colab Ready**: Includes notebook for GPU training on Google Colab

## Project Structure

```
src/jepa/               # Core package
├── data/               # JSONL dataset loader & masking transforms
├── models/             # VJEPA encoder & predictor architecture
├── training/           # Training loop & normalized L1 loss
└── evaluation/         # Cosine similarity metrics

scripts/                # CLI utilities
├── eval.py             # CPU evaluation script
└── inference.py        # CPU/GPU inference/scoring script

experiments/
├── checkpoints/        # Training checkpoints
└── inference_runs/     # Timestamped inference summaries + index

## Paper Pipeline

Run the full train -> eval -> inference pipeline with one flag:

```bash
python scripts/paper_run.py --hypothesis baseline
```

The hypothesis must exist in `configs/default.yaml` under:

`paper.hypotheses.<hypothesis>`

Each run is organized by datetime + hypothesis:

`experiments/paper_runs/YYYY-MM-DD/HHMMSS_<hypothesis>/`

With stage directories:
- `training/`
- `evaluation/`
- `inference/`

And run artifacts:
- `resolved_config.yaml`
- `run_summary.json`

A global run index is appended to:

`experiments/paper_runs/index.jsonl`

## Factorial Experiments (DOE)

Run a full-factorial batch from `configs/factorial.yaml`:

```bash
python scripts/run_factorial.py --config configs/factorial.yaml
```

Override tracking metadata from CLI (optional):

```bash
python scripts/run_factorial.py \
  --config configs/factorial.yaml \
  --design-id fx_v2 \
  --hypothesis "Lower mask_ratio improves cosine under latency budget." \
  --primary-metric evaluation.mean_cosine_similarity \
  --decision-rule "maximize primary metric with inference.runtime_ms.mean <= 200"
```

Optional dry-run to generate only the design matrix:

```bash
python scripts/run_factorial.py --config configs/factorial.yaml --dry-run
```

Each batch writes:
- `design_matrix.jsonl` (all factor/replicate combinations)
- `results.jsonl` (one row per completed run with extracted metrics)
- `batch_summary.json` (design/hypothesis metadata + progress/status)

Analyze a completed batch:

```bash
python scripts/analyze_factorial.py \
  --results experiments/factorial_runs/<date>/batch_<time>/results.jsonl \
  --factorial-config configs/factorial.yaml
```

Analysis outputs include per-metric CSVs for:
- Main effects by factor level
- Two-way interaction tables

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
  --output outputs/scores.jsonl \
  --device auto \
  --amp auto \
  --power-watts 75
```

Output `scores.jsonl` contains novelty scores and per-clip cost telemetry:
```json
{"score": 0.45, "scene": "scene-001", "camera": "CAM_FRONT", "tubelet_idx": 0, "runtime_ms": 18.27, "energy_joules": 1.37, "memory_overhead_mb": 0.22}
```

Run-level cost summary is written to a timestamped path under
`experiments/inference_runs/YYYY-MM-DD/HHMMSS_<run_label>/summary.json`.
An index entry is appended to `experiments/inference_runs/index.jsonl`.

Example summary:
```json
{
  "num_samples": 128,
  "batch_size": 1,
  "device": "cpu",
  "power_watts": 75.0,
  "runtime_ms": {"mean": 17.9, "p50": 17.6, "p95": 20.1, "min": 15.4, "max": 23.5, "total": 2291.2},
  "energy_joules": {"mean": 1.34, "p50": 1.32, "p95": 1.51, "min": 1.15, "max": 1.76, "total": 171.84},
  "memory_overhead_mb": {"peak_rss_delta": 82.4, "mean_per_batch_delta": 0.58}
}
```

## Testing

Run smoke tests:
```bash
pytest tests/
```
