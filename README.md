# JEPA Clip Ranking

Self-supervised **driving-clip triage**: rank dashcam clips by how likely they are to be worth a human's review, using a frozen [V-JEPA 2](https://arxiv.org/abs/2506.09985) encoder. The training objective is JEPA embedding prediction (masking); clips are then scored by prediction-error *novelty*. Ranking quality is measured against a human-labeled benchmark.

> [!NOTE]
> **Key finding (FTC 2026 study).** On a *fair, within-dataset* benchmark, self-supervised JEPA novelty ranks interesting clips **no better than chance** and no better than trivial no-training baselines — and the trained predictor head adds nothing. Earlier strong-looking numbers came from a **cross-dataset domain confound** (the "novelty" score is really a dataset detector). The interestingness signal *is* present in the frozen embeddings, though: a light **supervised** probe recovers it. See [EXPERIMENTS.md](EXPERIMENTS.md) for the full story and results.

## Quickstart

> [!TIP]
> Python 3.10–3.12. The pinned PyTorch 2.5.x stack does not support 3.13+.

```bash
# install (uv recommended)
uv sync
# or: pip install -e ".[dev]"

# verify the setup (no data or GPU needed — pure-function tests included)
uv run pytest tests/ -q

# run one experiment end-to-end (train -> score -> evaluate) on a manifest
uv run python run_experiment.py \
  --config configs/default.yaml \
  --run-dir experiments/runs/smoke \
  --set '{"runtime":{"profile":"cpu"}}' --set '{"train":{"epochs":1}}'
```

> [!IMPORTANT]
> Raw frames and the V-JEPA encoder weights are **not** in this repo. The pipeline consumes JSONL clip manifests that point at frames on disk, and the encoder is pulled from Hugging Face on first use (cache it for offline/cluster runs). The FTC experiments were run on an offline GPU cluster — see [EXPERIMENTS.md](EXPERIMENTS.md) and `scripts/cluster/` for the offline setup.

## Repo layout

```text
src/jepa/
  config.py        Config loading + runtime-profile resolution
  pipeline.py      train -> score -> evaluate stage runners
  baselines.py     No-training baselines (masked-gap, kNN-density, untrained predictor)
  data/            Manifest loading, dataset classes, masking transforms
  models/          Frozen V-JEPA encoder + trainable predictor head
  training/        Training loop + embedding loss
  evaluation/      Ranking metrics (AP, PR-AUC, NDCG, P@K, R@K), cost telemetry
  experiments/     Factorial design utilities

scripts/
  build_manifest.py            Build a v1 clip manifest from extracted frames
  build_labeling_pool.py       Sample a clip pool for human interestingness labeling
  review_labels.py             Local browser app for binary clip labeling
  merge_benchmark.py           Merge labeled manifests into one fixed benchmark
  run_main_experiments.py      source x frames x seeds matrix + baselines (SLURM-array ready)
  run_supervised_experiment.py Supervised probe on frozen embeddings (scene-disjoint CV)
  domain_confound_test*.py     Evidence that novelty tracks dataset, not interestingness
  score_baselines.py           Score a no-training baseline and evaluate it
  score_curve.py               AP-vs-epoch curve from per-epoch checkpoints
  analyze_main_experiments.py  Aggregate runs into mean +/- std tables
  cluster/                     SLURM launch scripts + offline weight staging

run_experiment.py / train.py / score.py / evaluate.py   Public entrypoints
configs/    default.yaml, benchmark.yaml, cluster.yaml, factorial.yaml
EXPERIMENTS.md   The FTC 2026 experiment cycle: design, commands, findings
```

## Workflow

1. Build a JSONL clip manifest with explicit `clip_id`, `split`, and `scene_id`.
2. Train the JEPA predictor on unlabeled `train` clips (encoder stays frozen).
3. Score held-out clips with a clip-level `review_value_score` (novelty).
4. Evaluate the ranking against human review-value labels.
5. Compare quality + efficiency across runs.

Each run writes `config_resolved.yaml`, `training/`, `scoring/`, `evaluation/`, and a top-level `summary.json`.

### Build a manifest from extracted frames

```bash
uv run python scripts/build_manifest.py \
  --dataroot data/raw/my_dataset \
  --output data/manifests/my_manifest.jsonl \
  --window 16 --stride 16 --absolute
```

### Score / evaluate / baselines

```bash
# trained-predictor novelty
uv run python score.py    --config configs/default.yaml --run-dir experiments/runs/baseline
uv run python evaluate.py --config configs/default.yaml --run-dir experiments/runs/baseline

# a no-training baseline (masked_gap | embedding_density | untrained_predictor)
uv run python scripts/score_baselines.py \
  --config configs/benchmark.yaml --run-dir experiments/runs/base_density \
  --method embedding_density --k 5
```

## Reproducing the FTC experiment cycle

The full matrix (training source × clip length × seeds), the fixed benchmark, the no-training baselines, and the supervised diagnostic are documented end-to-end — including the offline-cluster setup — in **[EXPERIMENTS.md](EXPERIMENTS.md)**.

## Scoring and evaluation

`score.py` needs no labels and emits one clip-level score per clip:

```
review_value_score = 1 - cosine_similarity(predicted_embedding, target_embedding)   # novelty proxy
```

`evaluate.py` joins scores to human labels keyed by `clip_id` and computes Precision@K, Recall@K, Average Precision, PR-AUC, and NDCG.

## Label review

```bash
uv run python scripts/review_labels.py \
  --labels-path data/manifests/<name>_evaluation_labels.jsonl \
  --manifest-path data/manifests/<name>.jsonl \
  --data-root data/raw/<dataset>
```

Open `http://127.0.0.1:8765`. Keys: `1` positive, `0` negative, `u` clear, `j`/`k` next/previous. Edits save back to the JSONL in place.

## Testing

```bash
uv run pytest tests/
```

> [!NOTE]
> The paper is written in a separate Overleaf project, cloned into `./paper` (git-ignored here). It is not part of this code repository.
