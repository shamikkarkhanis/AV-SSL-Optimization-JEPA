## Efficient Clip Mining for Autonomous Driving via JEPA-Based Representation Learning

Summary
- Self-supervised clip mining pipeline for autonomous driving video datasets using a JEPA-style VideoMAE backbone.
- Scores each clip by masked-patch reconstruction difficulty; higher scores indicate rare, complex, or novel scenarios.
- Produces visualizations and distributions to guide data selection and training set curation.

Why This Matters
- Large AV datasets contain heavy redundancy (common scenes, normal weather), wasting compute when sampled uniformly.
- JEPA-style representations capture predictive structure without labels; reconstruction difficulty is a useful proxy for novelty.
- Curating by score and diversity can reduce dataset size while maintaining or improving downstream performance.

Key Features
- VideoMAE-based masked autoencoding with MPS/CUDA/CPU support.
- Per-clip scoring (masked-patch MSE) with JSONL logging.
- Frame-level renderings of GT vs. predicted reconstructions and optional mask overlays.
- Histogram utility to visualize score distributions and summary stats.

Repository Structure
- `main.py`: Runs scoring and rendering; logs per-clip scores to `scores.jsonl` and images to `renders/`.
- `renderer.py`: Utilities to reconstruct predictions and save GT/pred frames and side-by-side panels.
- `distribution.py`: Plots histogram of scores and writes summary stats.
- `nuscenes_clips.py`: Minimal dataset loader for JSONL clip manifests.
- `clips_manifest.jsonl`: Example manifest stub; each line describes one 16-frame clip.
- `core_flow.txt`: Notes on the processing flow.

Installation
1) Python 3.10+ recommended. Create and activate a virtual environment.
2) Install dependencies: `pip install -r requirements.txt`
3) Ensure your system has a supported accelerator if desired (CUDA or Apple MPS). The code auto-detects device.

Data: Clip Manifest Format
Each line of `clips_manifest.jsonl` should be a JSON object:
`{"frame_paths": ["/abs/or/rel/path/img_000.jpg", ..., "img_015.jpg"], "scene": "scene-xxx", "camera": "CAM_FRONT"}`
- Exactly 16 frames per record (default). Alternative key `frames` is also supported.
- Optional fields like `scene` and `camera` are propagated into logs.

Quick Start
- Score clips and render examples: `python main.py`
  - Outputs per-sample logs to `scores.jsonl` and images to `renders/`.
  - Env vars:
    - `SCORES_OUT`: path for JSONL output (default `scores.jsonl`).
    - `SEED`: RNG seed (default 42).
- Plot score distribution: `python distribution.py scores.jsonl --out score_distribution.png --bins 40`
  - Also writes `score_distribution_stats.json` next to the plot.

Outputs
- `scores.jsonl`: One line per scored clip with fields `{step, scene, camera, first_frame, score}`.
- `renders/*.png`: GT, predicted, and optional side-by-side visualizations for mid-frame per sample.
- `score_distribution.png` and `score_distribution_stats.json`: Histogram and summary statistics.

Method Overview
- Representation learning: VideoMAE (JEPA-style) encodes context and predicts masked spatiotemporal patches.
- Scoring: Compute masked-patch reconstruction error per clip; higher error implies novelty/complexity.
- Selection: Rank by score and optionally enforce coverage/diversity in embedding space.

Roadmap (Optional Extensions)
- Add kNN/cluster-based diversification on embeddings for balanced selection.
- Calibrate scores per-camera or per-scene to reduce bias.
- Integrate mined clips with downstream imitation/RL policy training for end-to-end evaluation.

Citation/References
- VideoMAE: MCG-NJU/videomae-base (Hugging Face)
- JEPA: Joint Embedding Predictive Architectures for self-supervised learning
