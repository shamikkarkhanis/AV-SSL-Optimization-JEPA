# JEPA Clip Ranking

JEPA-based clip mining for driving data. The repo is organized around one product question:

rank clips by likely human review value.

The training objective remains JEPA embedding prediction. The primary experiment outcome is ranking quality on a human-labeled benchmark. Cosine similarity remains a secondary model-health metric.

## Workflow

The public entrypoints are:

- `train.py`
- `score.py`
- `evaluate.py`
- `run_experiment.py`

The normal sequence is:

1. Build or migrate a clip manifest with explicit `clip_id`, `split`, and `scene_id`.
2. Train JEPA on unlabeled `train` clips.
3. Score held-out clips with a clip-level review-value score.
4. Evaluate ranking quality against human review-value labels.
5. Compare experiment quality and efficiency from the run summaries.

Each run writes:

- `config_resolved.yaml`
- `training/summary.json`
- `training/checkpoints/`
- `scoring/scores.jsonl`
- `scoring/summary.json`
- `evaluation/summary.json`
- `summary.json`

## Repo Layout

```text
src/jepa/
  config.py                 Config loading and runtime-profile resolution
  pipeline.py               train -> score -> evaluate stage runners
  data/                     Manifest loading, dataset classes, transforms
  models/                   VJEPA encoder and JEPA predictor model
  training/                 Training loop and embedding loss
  evaluation/               Ranking metrics, cosine similarity, telemetry helpers
  experiments/              Factorial design utilities

scripts/
  build_manifest_from_frames.py
  migrate_manifest.py
  run_factorial.py
  analyze_factorial.py

train.py
score.py
evaluate.py
run_experiment.py
```

## Installation

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
pip install -e ".[dev]"
```

## Data Preparation

The pipeline consumes JSONL clip manifests, not raw videos directly.

Expected v1 clip record:

```json
{
  "clip_id": "scene_001__CAM_FRONT__000000__000015",
  "split": "train",
  "scene_id": "scene_001",
  "camera": "CAM_FRONT",
  "frame_paths": [
    "scene_001/CAM_FRONT/000000.jpg",
    "scene_001/CAM_FRONT/000001.jpg"
  ],
  "timestamps": ["000000", "000001"],
  "metadata": {}
}
```

### Build a Manifest From Extracted Frames

Frame directory layout:

```text
data/raw/my_dataset/
  scene_001/
    CAM_FRONT/
      000000.jpg
      000001.jpg
```

Build a manifest:

```bash
uv run python scripts/build_manifest_from_frames.py \
  --frames-root data/raw/my_dataset \
  --output data/manifests/my_manifest.jsonl \
  --clip-length 16 \
  --stride 16 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --seed 42 \
  --camera CAM_FRONT
```

### Migrate a Legacy Manifest

```bash
uv run python scripts/migrate_manifest.py \
  --input data/manifests/clips_manifest.jsonl \
  --output data/manifests/clips_manifest_v1.jsonl \
  --train-ratio 0.7 \
  --score-ratio 0.15 \
  --seed 42
```

## Configuration

The default config is [configs/default.yaml](/Users/shamik/Documents/AV-SSL-Optimization-JEPA/configs/default.yaml).

Top-level sections:

- `dataset`
- `model`
- `train`
- `score`
- `evaluation`
- `runtime`
- `experiment`

Important controls:

- `model.init_mode`: `pretrained`, `resume`, `scratch`
- `model.encoder_mode`: `frozen`, `finetune`
- `runtime.profile`: `cpu`, `gpu`, `ddp`
- `dataset.training_manifest`, `dataset.scoring_manifest`, `dataset.evaluation_manifest`
- `dataset.training_split`, `dataset.scoring_split`, `dataset.evaluation_split`
- `dataset.evaluation_labels`

Inline overrides are supported with `--set` as JSON:

```bash
python3 run_experiment.py \
  --config configs/default.yaml \
  --run-dir experiments/runs/smoke_cpu \
  --set '{"runtime":{"profile":"cpu","batch_size_overrides":{"train":1,"score":1,"evaluation":1}}}' \
  --set '{"train":{"epochs":1}}'
```

## Stage Usage

Train only:

```bash
uv run python train.py \
  --config configs/default.yaml \
  --run-dir experiments/runs/baseline
```

Score only:

```bash
uv run python score.py \
  --config configs/default.yaml \
  --run-dir experiments/runs/baseline \
  --checkpoint experiments/runs/baseline/training/checkpoints/best_model.pt
```

Evaluate only:

```bash
uv run python evaluate.py \
  --config configs/default.yaml \
  --run-dir experiments/runs/baseline \
  --scores experiments/runs/baseline/scoring/scores.jsonl
```

Full experiment:

```bash
uv run python run_experiment.py \
  --config configs/default.yaml \
  --run-dir experiments/runs/baseline
```

## Scoring and Evaluation

`score.py` does not require human labels. It produces one clip-level score per clip:

- `review_value_score`
- `mean_cosine_similarity`
- `tubelet_score_mean`
- `tubelet_score_std`
- `tubelet_count`

In the current implementation:

`review_value_score = 1 - cosine_similarity`

This is a novelty proxy, not ground truth.

`evaluate.py` requires human labels keyed by `clip_id`. It joins model scores to benchmark labels and computes:

- `Precision@K`
- `Recall@K`
- `Average Precision`
- `PR-AUC`
- `NDCG`

Human label schema:

```json
{
  "clip_id": "scene_001__CAM_FRONT__000080__000095",
  "review_value": "high_value",
  "review_value_grade": 2,
  "reason_codes": ["safety_critical_interaction"],
  "reviewer_id": "rater_01",
  "adjudicated_label": "high_value",
  "agreement": 1.0
}
```

The intended review-value classes are:

- `high_value`
- `medium_value`
- `low_value`

## Experiment Outputs

Training resource stats are stored in:

- `RUN_DIR/training/summary.json`

Scoring / inference resource stats are stored in:

- `RUN_DIR/scoring/summary.json`

These summaries include:

- wall-clock time
- samples or clips per second
- latency mean / p50 / p95
- peak memory
- effective batch size
- device and runtime profile
- estimated energy for scoring

## Factorial Experiments

Use [configs/factorial.yaml](/Users/shamik/Documents/AV-SSL-Optimization-JEPA/configs/factorial.yaml) to sweep config factors across the new pipeline.

Run a batch:

```bash
uv run python scripts/run_factorial.py --config configs/factorial.yaml
```

Analyze a completed batch:

```bash
uv run python scripts/analyze_factorial.py \
  --results experiments/factorial_runs/<date>/batch_<time>/results.jsonl \
  --factorial-config configs/factorial.yaml
```

Each batch writes:

- `design_matrix.jsonl`
- `results.jsonl`
- `batch_summary.json`
- `runs/<run_name>/...`

## Testing

```bash
uv run pytest tests/
```
