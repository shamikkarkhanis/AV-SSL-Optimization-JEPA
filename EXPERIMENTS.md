# FTC 2026 Experiment Cycle

This document describes the experiment cycle built to address the three reviewer
asks: (1) a real baseline, (2) longer training + multiple seeds, (3) a fixed,
credible benchmark. Related-work expansion is a writing task and lives in the paper.

## What changed in the code

| Piece | File | Purpose |
|---|---|---|
| No-training baselines A/B/C | `src/jepa/baselines.py`, `scripts/score_baselines.py` | Answer "is the JEPA predictor actually helping?" |
| Held-fixed benchmark | `pipeline._stage_frames_per_clip`, `configs/benchmark.yaml` | Score every run on identical eval clips regardless of training clip length |
| Experiment driver | `scripts/run_main_experiments.py` | source × frames × seeds matrix + baselines, indexed for SLURM arrays |
| Aggregation | `scripts/analyze_main_experiments.py` | mean ± std tables for the paper |
| Training curves | `scripts/score_curve.py` | AP-vs-epoch from per-epoch checkpoints |
| Benchmark assembly | `scripts/merge_benchmark.py` | one self-contained labeled benchmark (nuScenes + Waymo + BDD) |
| Cluster glue | `scripts/cluster/*` | data build, SLURM array, offline weight staging |

## The baselines (the core new contribution)

All reuse the frozen V-JEPA2 encoder; none require a trained predictor.

- **A — `masked_gap`**: `1 − cos(encoder(masked), encoder(clean))`. The trivial,
  zero-cost floor — no predictor, no training. Shows whether training the predictor
  adds anything over raw V-JEPA masking sensitivity.
- **B — `embedding_density`**: leave-one-out kNN novelty in embedding space. The
  standard "raw embedding novelty" baseline; this is what the professor's phrasing
  most directly refers to. It is the baseline that can't be dismissed as a strawman.
- **C — `untrained_predictor`**: the full pipeline with the predictor left at random
  init. Sanity check that isolates the value of the training loop.

If the trained method beats all three, the training step is justified from three angles.

## Experiment matrix

Four training conditions (the report's A–D, but with longer training + seeds):

| Condition | Source | Frames | Train manifest |
|---|---|---|---|
| samples_16 | nuScenes samples | 16 | nuscenes_samples_16.jsonl |
| samples_32 | nuScenes samples | 32 | nuscenes_samples_32.jsonl |
| sweeps_16 | nuScenes sweeps | 16 | nuscenes_sweeps_16.jsonl |
| sweeps_32 | nuScenes sweeps | 32 | nuscenes_sweeps_32.jsonl |

Each runs with 5 seeds at 25 epochs. Plus the 3 baselines (A & B once, C × 3 seeds).
Total plan = **25 indexed items** (`run_main_experiments.py --dry-run` to list them).

Winning factorial hyperparameters are baked into `configs/benchmark.yaml`:
`mask_ratio=0.5`, `predictor.hidden_dim=512`, `lr=1e-4`.

## End-to-end workflow (on the cluster)

```bash
# 0. One-time: stage encoder weights for offline nodes (run where there is internet)
bash scripts/cluster/prestage_model.sh        # then set HF_HOME on the cluster

# 1. One-time: build training manifests + the fixed benchmark (edit *_ROOT vars first)
bash scripts/cluster/build_data.sh

# 2. Launch the full matrix as a GPU job array
sbatch scripts/cluster/run_experiments.slurm

# 3. Aggregate into paper tables (point at the batch dir; merges results_*.jsonl)
python scripts/analyze_main_experiments.py --results experiments/main/<JOBID>

# 4. (Optional) AP-vs-epoch curve for one run per condition, for the convergence figure
python scripts/score_curve.py --config configs/benchmark.yaml \
    --run-dir experiments/main/<JOBID>/runs/samples_32__seed1 --every 2
```

Run baselines standalone if desired:
```bash
python scripts/score_baselines.py --config configs/benchmark.yaml \
    --run-dir experiments/runs/baseline_B --method embedding_density --k 5
```

## Compute budget (single V100, ~report timings extrapolated to 25 epochs)

| Condition | ~min/run | × 5 seeds |
|---|---|---|
| samples_16 | ~13 | ~1.1 h |
| samples_32 | ~22 | ~1.8 h |
| sweeps_16 | ~56 | ~4.6 h |
| sweeps_32 | ~100 | ~8.3 h |

≈ 16 GPU-hours total for training + negligible for baselines. As a `--array` it
finishes in wall-clock ≈ the longest single run if enough GPUs are free. Dial down
with `--seeds 1,2,3` and/or `--epochs 15` if the queue is tight before July 1.

## Known caveats to disclose in the paper

- **Benchmark domain confound**: all Waymo/BDD benchmark clips are positive; all
  negatives are nuScenes. A model could partly exploit domain shift rather than
  "interestingness." Mitigation/disclosure needed; ideally add nuScenes positives /
  cross-domain negatives later.
- **Positive rate ≈ 0.40** on the merged benchmark (42 pos / 105), so random-ranking
  AP ≈ 0.40 — that is the floor the baselines and method must clear.
- **Encoder clip-length mismatch**: V-JEPA2 was pretrained at 64 frames (`fpc64`);
  we feed 16/32-frame clips in 2-frame tubelets. Works, but worth a sentence.
