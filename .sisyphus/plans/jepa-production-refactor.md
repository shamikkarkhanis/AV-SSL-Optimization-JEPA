# JEPA Production Refactor: Research to Production ML Pipeline

## TL;DR

> **Quick Summary**: Refactor Jupyter research notebooks into a production-ready ML repository with proper train/eval/inference scripts for JEPA-based clip mining.
> 
> **Deliverables**:
> - Python package `src/jepa/` with modular components (data, models, training, evaluation)
> - Colab training notebook for GPU-heavy tasks
> - CPU-friendly eval and inference scripts
> - YAML config management with CLI overrides
> - Checkpoint management and resumption support
> 
> **Estimated Effort**: Medium (3-5 days for core implementation)
> **Parallel Execution**: YES - 3 waves (setup → core components → scripts)
> **Critical Path**: Data module → Models → Training → Eval/Inference

---

## Context

### Original Request
User has two research Jupyter notebooks (`jepa/embeddings.ipynb`, `jepa/prediction.ipynb`) that demonstrate JEPA's feasibility for autonomous driving clip mining. The notebooks work well but need to be refactored into a production-ready codebase with proper structure, reusability, and scalability.

### Interview Summary
**Key Discussions**:
- **Framework choice**: Pure PyTorch with custom loops (no Lightning/Accelerate)
- **Config management**: YAML files + CLI overrides (no Hydra)
- **Experiment tracking**: Local filesystem only (no W&B/MLflow)
- **Architecture**: Colab notebook for GPU training, local CPU scripts for eval/inference
- **Data format**: JSONL manifests pointing to frame paths (nuScenes v1.0-mini)
- **Removed**: `jepa_mining.py` (old zero-shot approach, no longer needed)

**Research Findings**:
- VJEPA encoder (`facebook/vjepa2-vitl-fpc64-256`) frozen during training
- Predictor head: 2-layer MLP (1024 → 1024 → 1024) with GELU, dropout 0.1
- Training: 30 epochs, AdamW lr=1e-4, batch_size=8, L1 loss on normalized embeddings
- Masking: 75% patch ratio, 16×16 patches, tubelet_size=2 frames
- Eval metric: Cosine similarity (research achieved 0.95 predicted vs clean)
- Inference metric: 1 - cosine_similarity(clean, masked) = novelty score

---

## Work Objectives

### Core Objective
Transform research notebook code into a production-ready Python package with:
1. Modular, reusable components (not monolithic notebook code)
2. Clean separation of training (GPU/Colab) and inference (CPU/local)
3. Config-driven hyperparameter management
4. Checkpoint save/load with resumption support
5. Executable acceptance criteria (no manual verification)

### Concrete Deliverables
- `src/jepa/` Python package (installable via setup.py)
- `notebooks/train_jepa.ipynb` for Colab GPU training
- `scripts/eval.py` for local CPU evaluation
- `scripts/inference.py` for local CPU batch scoring
- `configs/default.yaml` with all hyperparameters
- `requirements.txt` with pinned dependencies
- `tests/` with smoke tests for data loading and model forward passes
- `README.md` with setup and usage instructions

### Definition of Done
- [ ] Training notebook runs end-to-end on Colab and saves checkpoint
- [ ] Eval script loads checkpoint and computes cosine similarity metric (CPU)
- [ ] Inference script processes JSONL manifest and outputs scores JSONL (CPU)
- [ ] Smoke tests pass (`python -m pytest tests/`)
- [ ] Config can be overridden via CLI (`--lr 1e-3` overrides `config.yaml`)
- [ ] Checkpoint resumption works (train 10 epochs → kill → resume from epoch 11)

### Must Have
- Exact reproduction of notebook training logic (same architecture, loss, optimizer)
- JSONL manifest data loading (compatible with nuScenes v1.0-mini format)
- Frozen encoder weights (no fine-tuning)
- L1 loss on L2-normalized embeddings
- Checkpoint schema: `{model_state, optimizer_state, epoch, config, best_metric}`
- CPU fallback for scripts (graceful degradation if no GPU)

### Must NOT Have (Guardrails)
- **NO** modification of research notebooks (`jepa/*.ipynb` stay untouched)
- **NO** ML frameworks beyond PyTorch (no Lightning, Accelerate, Hydra, Ray)
- **NO** distributed training (single GPU only)
- **NO** mixed precision (AMP)
- **NO** gradient accumulation
- **NO** learning rate schedulers (not in research spec)
- **NO** data augmentation beyond masking
- **NO** experiment tracking dashboards (W&B, MLflow, TensorBoard UI)
- **NO** inference optimizations (ONNX, TensorRT, quantization)
- **NO** abstract base classes or generic frameworks ("could be used for other models")
- **NO** user manual verification steps in acceptance criteria

---

## Verification Strategy

### Automated Verification Only

> **CRITICAL**: ALL acceptance criteria MUST be executable by agents without user intervention.

**Training Verification** (Colab notebook):
```bash
# Smoke test: Train 2 epochs, assert loss decreases
# Run in Colab cell:
!python -c "from src.jepa.training.trainer import Trainer; ..."
# Expected: Epoch 2 loss < Epoch 1 loss (numerical check)
```

**Evaluation Verification** (Local CPU):
```bash
# Eval script with sample checkpoint
python scripts/eval.py --checkpoint experiments/sample_ckpt.pt --manifest data/test_manifest.jsonl
# Expected output: JSON file with {"mean_cosine_similarity": <value between 0.8-1.0>}
# Verification: jq '.mean_cosine_similarity' output.json | awk '$1 >= 0.8 && $1 <= 1.0 {print "PASS"}'
```

**Inference Verification** (Local CPU):
```bash
# Batch scoring from manifest
python scripts/inference.py --checkpoint experiments/sample_ckpt.pt --manifest data/test_manifest.jsonl --output scores.jsonl
# Expected: scores.jsonl exists with 1 line per input clip
# Verification: wc -l scores.jsonl == wc -l data/test_manifest.jsonl
```

**Checkpoint Resumption Verification**:
```python
# Train 5 epochs → save checkpoint
# Load checkpoint → assert epoch=5
# Train 3 more → assert final epoch=8
# Compare loss trajectory continuity
```

**Reproducibility Verification**:
```python
# Train with --seed 42 for 2 epochs → save loss1
# Train with --seed 42 for 2 epochs → save loss2
# assert abs(loss1 - loss2) < 1e-6  # Identical
```

**Evidence to Capture**:
- Terminal stdout logs (training progress, loss values)
- Checkpoint file sizes and contents (model.pt -> assert keys exist)
- Output JSON/JSONL files (eval metrics, inference scores)
- Test execution logs (`pytest -v` output)

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Repository Setup - No dependencies):
├── Task 1: Create directory structure
├── Task 2: Setup package configuration (setup.py, requirements.txt)
└── Task 3: Create base config YAML

Wave 2 (Core Components - Depends on Wave 1):
├── Task 4: Data loading module (dataset.py, transforms.py)
├── Task 5: Model components (encoder.py, predictor.py, jepa.py)
├── Task 6: Training utilities (trainer.py, losses.py)
└── Task 7: Evaluation utilities (metrics.py)

Wave 3 (Interfaces - Depends on Wave 2):
├── Task 8: Colab training notebook
├── Task 9: Eval script
├── Task 10: Inference script
└── Task 11: Smoke tests + README

Critical Path: Task 1 → Task 4 → Task 5 → Task 6 → Task 8
Parallel Speedup: ~40% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2,3,4,5,6,7 | None (must go first) |
| 2 | 1 | 4,5,6,7,8,9,10 | 3 |
| 3 | 1 | 8,9,10 | 2 |
| 4 | 1,2 | 8,9,10 | 5,6,7 |
| 5 | 1,2 | 6,8,9,10 | 4,7 |
| 6 | 1,2,5 | 8 | 7 |
| 7 | 1,2 | 9 | 4,6 |
| 8 | 2,3,4,5,6 | None | 9,10,11 |
| 9 | 2,3,4,5,7 | None | 8,10,11 |
| 10 | 2,3,4,5,7 | None | 8,9,11 |
| 11 | 2,3 | None | 8,9,10 |

---

## TODOs

### Wave 1: Repository Setup

- [ ] 1. Create Directory Structure

  **What to do**:
  - Create `src/jepa/` package with `__init__.py`
  - Create subdirectories: `data/`, `models/`, `training/`, `evaluation/`, `utils/`
  - Create `scripts/`, `configs/`, `notebooks/`, `tests/`, `experiments/` directories
  - Add `__init__.py` to all Python package directories
  - Create `.gitignore` entries for `experiments/`, `*.pyc`, `__pycache__/`

  **Must NOT do**:
  - Modify existing `jepa/` directory (research notebooks stay untouched)
  - Create unnecessary subdirectories (keep flat structure within each module)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple directory creation, no complex logic
  - **Skills**: []
    - No special skills needed for basic filesystem operations

  **Parallelization**:
  - **Can Run In Parallel**: NO (foundational task)
  - **Parallel Group**: Sequential (Wave 1 start)
  - **Blocks**: Tasks 2,3,4,5,6,7,8,9,10,11
  - **Blocked By**: None

  **References**:
  - Python package structure: https://packaging.python.org/en/latest/tutorials/packaging-projects/
  - Existing structure: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA` (see jepa/, videomae/, v1.0-mini/)

  **Acceptance Criteria**:
  ```bash
  # Directory structure verification
  test -d src/jepa/data && echo "PASS: src/jepa/data exists"
  test -d src/jepa/models && echo "PASS: src/jepa/models exists"
  test -d src/jepa/training && echo "PASS: src/jepa/training exists"
  test -d src/jepa/evaluation && echo "PASS: src/jepa/evaluation exists"
  test -d scripts && echo "PASS: scripts exists"
  test -d configs && echo "PASS: configs exists"
  test -d notebooks && echo "PASS: notebooks exists"
  test -f src/jepa/__init__.py && echo "PASS: Package init exists"
  ```

  **Commit**: YES
  - Message: `feat(repo): initialize production package structure`
  - Files: `src/jepa/`, `scripts/`, `configs/`, `notebooks/`, `tests/`
  - Pre-commit: N/A (no code yet)

---

- [ ] 2. Setup Package Configuration

  **What to do**:
  - Create `setup.py` to make `src/jepa` installable (`pip install -e .`)
  - Create `requirements.txt` with EXACT versions from research notebooks:
    - torch==2.8.0
    - torchvision==0.23.0
    - transformers==4.57.0
    - pillow==11.3.0
    - numpy==2.3.3
    - pyyaml==6.0.3
    - (add others from `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/videomae/requirements.txt`)
  - Add `[dev]` extra with pytest, matplotlib (for visualizations)

  **Must NOT do**:
  - Use version ranges (e.g., `torch>=2.0`) - pin exact versions for reproducibility
  - Add unnecessary dependencies (sklearn, pandas, etc.) not used in notebooks

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard Python packaging, template-based
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 3)
  - **Parallel Group**: Wave 1 (with Task 3)
  - **Blocks**: Tasks 4,5,6,7,8,9,10
  - **Blocked By**: Task 1

  **References**:
  - Existing requirements: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/videomae/requirements.txt`
  - Setup.py template: https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/

  **Acceptance Criteria**:
  ```bash
  # Package installable
  pip install -e . && python -c "import jepa" && echo "PASS: Package imports"
  
  # Requirements match notebook versions
  pip list | grep "torch " | grep "2.8.0" && echo "PASS: torch version correct"
  pip list | grep "transformers " | grep "4.57.0" && echo "PASS: transformers version correct"
  ```

  **Commit**: YES
  - Message: `feat(build): add package configuration and dependencies`
  - Files: `setup.py`, `requirements.txt`
  - Pre-commit: `python setup.py check`

---

- [ ] 3. Create Base Config YAML

  **What to do**:
  - Create `configs/default.yaml` with all hyperparameters from research notebooks
  - Structure: `data:`, `model:`, `training:`, `evaluation:` sections
  - Extract exact values from `jepa/prediction.ipynb`:
    - model.encoder_name: "facebook/vjepa2-vitl-fpc64-256"
    - model.predictor_hidden_dim: 1024
    - model.predictor_dropout: 0.1
    - training.epochs: 30
    - training.batch_size: 8
    - training.lr: 1e-4
    - training.optimizer: "adamw"
    - training.loss: "l1"
    - training.checkpoint_every: 5
    - data.tubelet_size: 2
    - data.mask_ratio: 0.75
    - data.patch_size: 16
    - data.image_size: 224
    - evaluation.metric: "cosine_similarity"
    - seed: 42

  **Must NOT do**:
  - Add hyperparameters not in research notebooks (lr scheduler, weight decay values, etc.)
  - Use complex hierarchical config (Hydra-style composition)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple YAML file creation from notebook values
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 2)
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Tasks 8,9,10
  - **Blocked By**: Task 1

  **References**:
  - Research notebook: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/jepa/prediction.ipynb` (cells with hyperparameters)
  - YAML syntax: https://yaml.org/spec/1.2/spec.html

  **Acceptance Criteria**:
  ```bash
  # YAML valid and loadable
  python -c "import yaml; c=yaml.safe_load(open('configs/default.yaml')); assert c['training']['lr'] == 1e-4; print('PASS: Config valid')"
  
  # All required sections present
  python -c "import yaml; c=yaml.safe_load(open('configs/default.yaml')); assert all(k in c for k in ['data','model','training']); print('PASS: Sections exist')"
  ```

  **Commit**: YES
  - Message: `feat(config): add default hyperparameters from research`
  - Files: `configs/default.yaml`
  - Pre-commit: `python -c "import yaml; yaml.safe_load(open('configs/default.yaml'))"`

---

### Wave 2: Core Components

- [ ] 4. Data Loading Module

  **What to do**:
  - Create `src/jepa/data/dataset.py` with `JEPADataset(torch.utils.data.Dataset)`:
    - `__init__(manifest_path, transform=None)` - loads JSONL manifest
    - `__getitem__(idx)` - returns dict `{"clean_frames": tensor, "masked_frames": tensor, "meta": dict}`
    - Support both "frame_paths" and "frames" keys (from manifest compatibility)
    - Load images as PIL, convert to RGB, resize to 224×224
  - Create `src/jepa/data/transforms.py` with `MaskTubelet` transform:
    - Extract masking logic from `jepa/prediction.ipynb:mask_tubelet_pixels()`
    - Input: tensor (T, C, H, W), Output: masked tensor + mask (T, H/patch_size, W/patch_size)
    - Use random seed from config for reproducibility
  - Create `src/jepa/data/__init__.py` exposing `JEPADataset`, `MaskTubelet`

  **Must NOT do**:
  - Add data augmentation (rotation, color jitter, random crops) beyond masking
  - Add caching or preprocessing pipelines
  - Add multi-processing optimizations (keep simple for CPU scripts)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Straightforward data loading, low complexity
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 5,6,7)
  - **Parallel Group**: Wave 2 (with Tasks 5,7)
  - **Blocks**: Tasks 8,9,10
  - **Blocked By**: Tasks 1,2

  **References**:
  - Data loading pattern: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/jepa/prediction.ipynb:63-99` (tubelet creation)
  - Masking logic: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/jepa/prediction.ipynb:237-286` (mask_tubelet_pixels function)
  - Manifest format: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/videomae/clips_manifest.jsonl`
  - Existing loader: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/videomae/nuscenes_clips.py:6-37`

  **Acceptance Criteria**:
  ```bash
  # Dataset loads manifest
  python -c "from jepa.data import JEPADataset; ds=JEPADataset('videomae/clips_manifest.jsonl'); assert len(ds) > 0; print('PASS: Dataset loads')"
  
  # Returns correct tensor shapes
  python -c "from jepa.data import JEPADataset; ds=JEPADataset('videomae/clips_manifest.jsonl'); batch=ds[0]; assert batch['clean_frames'].shape == (2,3,224,224); print('PASS: Tensor shape correct')"
  
  # Masking transform works
  python -c "import torch; from jepa.data import MaskTubelet; t=torch.rand(2,3,224,224); masked,mask=MaskTubelet(mask_ratio=0.75)(t); assert masked.shape == t.shape; print('PASS: Masking works')"
  ```

  **Commit**: YES
  - Message: `feat(data): add JSONL manifest dataset and masking transform`
  - Files: `src/jepa/data/dataset.py`, `src/jepa/data/transforms.py`, `src/jepa/data/__init__.py`
  - Pre-commit: `python -c "from jepa.data import JEPADataset, MaskTubelet"`

---

- [ ] 5. Model Components

  **What to do**:
  - Create `src/jepa/models/encoder.py` with `VJEPAEncoder` wrapper:
    - `__init__(model_name="facebook/vjepa2-vitl-fpc64-256")` - loads HF model
    - `forward(pixel_values)` - returns last_hidden_state
    - Freeze all parameters in `__init__` (no fine-tuning)
    - `.eval()` mode by default
  - Create `src/jepa/models/predictor.py` with `PredictorHead`:
    - Extract architecture from `jepa/prediction.ipynb:737-762`
    - `__init__(emb_dim=1024)` - LayerNorm + MLP(emb_dim+1 → emb_dim → emb_dim)
    - `forward(masked_emb, mask_frac)` - concatenate mask_frac, pass through MLP
    - Use nn.GELU(), nn.Dropout(0.1)
  - Create `src/jepa/models/jepa.py` with `JEPAModel`:
    - Combines encoder + predictor
    - `forward(pixel_values, masked_pixel_values, mask_frac)` - returns (clean_emb, pred_emb)
    - Global average pooling (GAP) on encoder output: `mean(dim=1)`
  - Create `src/jepa/models/__init__.py` exposing all models

  **Must NOT do**:
  - Add encoder fine-tuning support (weights must stay frozen)
  - Add multiple predictor architectures (stick to 2-layer MLP from research)
  - Add EMA (exponential moving average) of weights

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Straightforward model wrappers, low complexity
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4,7)
  - **Parallel Group**: Wave 2 (with Tasks 4,7)
  - **Blocks**: Tasks 6,8,9,10
  - **Blocked By**: Tasks 1,2

  **References**:
  - Encoder usage: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/jepa/embeddings.ipynb:36-46` (model loading)
  - Predictor architecture: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/jepa/prediction.ipynb:737-762` (PredictorHead class)
  - Forward pass: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/jepa/prediction.ipynb:166-174` (GAP logic)

  **Acceptance Criteria**:
  ```bash
  # Encoder loads and freezes
  python -c "from jepa.models import VJEPAEncoder; e=VJEPAEncoder(); assert all(not p.requires_grad for p in e.parameters()); print('PASS: Encoder frozen')"
  
  # Predictor forward pass works
  python -c "import torch; from jepa.models import PredictorHead; p=PredictorHead(); out=p(torch.rand(2,1024), torch.rand(2,1)); assert out.shape==(2,1024); print('PASS: Predictor works')"
  
  # JEPA model forward pass
  python -c "import torch; from jepa.models import JEPAModel; m=JEPAModel(); clean,pred=m(torch.rand(1,2,3,224,224), torch.rand(1,2,3,224,224), torch.tensor([0.75])); assert clean.shape==(1,1024); print('PASS: JEPA forward')"
  ```

  **Commit**: YES
  - Message: `feat(models): add VJEPA encoder and predictor head`
  - Files: `src/jepa/models/encoder.py`, `src/jepa/models/predictor.py`, `src/jepa/models/jepa.py`, `src/jepa/models/__init__.py`
  - Pre-commit: `python -c "from jepa.models import VJEPAEncoder, PredictorHead, JEPAModel"`

---

- [ ] 6. Training Utilities

  **What to do**:
  - Create `src/jepa/training/losses.py` with `jepa_loss()` function:
    - Extract from `jepa/prediction.ipynb:843-846` (L1 loss on normalized embeddings)
    - Input: `(pred_emb, target_emb)`, Output: scalar loss
    - L2 normalize both embeddings (`F.normalize(p=2, dim=1)`)
    - Return `F.l1_loss(pred_norm, target_norm)`
  - Create `src/jepa/training/trainer.py` with `train_epoch()` and `validate_epoch()` functions:
    - `train_epoch(model, loader, optimizer, device)` - returns avg_loss
    - `validate_epoch(model, loader, device)` - returns avg_loss (no optimizer step)
    - Use `jepa_loss` from losses.py
    - Move tensors to device, handle mixed CPU/GPU gracefully
  - Create `src/jepa/training/__init__.py` exposing `jepa_loss`, `train_epoch`, `validate_epoch`

  **Must NOT do**:
  - Add learning rate schedulers (not in research spec)
  - Add gradient clipping (not in research spec)
  - Add mixed precision (AMP) training
  - Add distributed training (DDP/FSDP)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Standard PyTorch training loop, low complexity
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 7)
  - **Parallel Group**: Wave 2 (with Task 7)
  - **Blocks**: Task 8
  - **Blocked By**: Tasks 1,2,5

  **References**:
  - Loss function: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/jepa/prediction.ipynb:843-846` (criterion + normalization)
  - Training loop: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/jepa/prediction.ipynb:836-867` (epoch loop)

  **Acceptance Criteria**:
  ```bash
  # Loss function works
  python -c "import torch; from jepa.training import jepa_loss; l=jepa_loss(torch.rand(2,1024), torch.rand(2,1024)); assert l.item() >= 0; print('PASS: Loss computes')"
  
  # Train epoch runs
  python -c "import torch; from jepa.training import train_epoch; from jepa.models import JEPAModel; m=JEPAModel(); opt=torch.optim.AdamW(m.predictor.parameters(), lr=1e-4); # TODO: create mock loader and test"
  ```

  **Commit**: YES
  - Message: `feat(training): add JEPA loss and training loop`
  - Files: `src/jepa/training/losses.py`, `src/jepa/training/trainer.py`, `src/jepa/training/__init__.py`
  - Pre-commit: `python -c "from jepa.training import jepa_loss, train_epoch"`

---

- [ ] 7. Evaluation Utilities

  **What to do**:
  - Create `src/jepa/evaluation/metrics.py` with `compute_cosine_similarity()`:
    - Input: `(pred_embeddings, target_embeddings)` as numpy arrays
    - L2 normalize both: `arr / np.linalg.norm(arr, axis=1, keepdims=True)`
    - Return: mean cosine similarity across batch
  - Create `src/jepa/evaluation/__init__.py` exposing `compute_cosine_similarity`

  **Must NOT do**:
  - Add PCA visualization (keep in notebooks, not production code)
  - Add clustering metrics (not in research spec)
  - Add t-SNE, UMAP, or other dimensionality reduction

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple metric computation, very low complexity
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4,6)
  - **Parallel Group**: Wave 2 (with Tasks 4,6)
  - **Blocks**: Task 9
  - **Blocked By**: Tasks 1,2

  **References**:
  - Cosine similarity: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/jepa/prediction.ipynb:909-914` (l2_normalize + sklearn cosine_similarity)

  **Acceptance Criteria**:
  ```bash
  # Cosine similarity computes
  python -c "import numpy as np; from jepa.evaluation import compute_cosine_similarity; sim=compute_cosine_similarity(np.random.rand(10,1024), np.random.rand(10,1024)); assert 0 <= sim <= 1; print('PASS: Metric works')"
  ```

  **Commit**: YES
  - Message: `feat(evaluation): add cosine similarity metric`
  - Files: `src/jepa/evaluation/metrics.py`, `src/jepa/evaluation/__init__.py`
  - Pre-commit: `python -c "from jepa.evaluation import compute_cosine_similarity"`

---

### Wave 3: Interfaces

- [ ] 8. Colab Training Notebook

  **What to do**:
  - Create `notebooks/train_jepa.ipynb` for Colab GPU training:
    - Cell 1: Mount Google Drive (for checkpoints)
    - Cell 2: Install package (`!pip install -e /content/drive/MyDrive/jepa-repo`)
    - Cell 3: Load config (`yaml.safe_load(open('configs/default.yaml'))`)
    - Cell 4: Setup data loaders (train/val split)
    - Cell 5: Initialize model, optimizer, device
    - Cell 6: Training loop (30 epochs, save checkpoint every 5)
    - Cell 7: Save final checkpoint to Drive
    - Cell 8: Plot loss curves (matplotlib)
  - Add markdown cells explaining each step
  - Add config override via notebook variables (e.g., `OVERRIDE_LR = 1e-3`)
  - Checkpoint schema: `{"model_state_dict": ..., "optimizer_state_dict": ..., "epoch": ..., "config": ..., "best_val_loss": ...}`
  - Support resumption: check if checkpoint exists, load and continue

  **Must NOT do**:
  - Add complex logging (progress bars are fine, no TensorBoard)
  - Add multi-GPU support (single GPU Colab)
  - Add automated hyperparameter search

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Notebook UI, user-facing interface, visualization
  - **Skills**: [`frontend-ui-ux`]
    - frontend-ui-ux: Notebook is a user interface, needs clear structure and explanations

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 9,10,11)
  - **Parallel Group**: Wave 3 (with Tasks 9,10,11)
  - **Blocks**: None
  - **Blocked By**: Tasks 2,3,4,5,6

  **References**:
  - Training loop: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/jepa/prediction.ipynb:826-867`
  - Checkpoint save: Python `torch.save()` pattern
  - Colab mounting: `from google.colab import drive; drive.mount('/content/drive')`

  **Acceptance Criteria**:
  ```python
  # Notebook has all required cells
  !jupyter nbconvert --to python notebooks/train_jepa.ipynb --stdout | grep -q "# Cell 1: Mount Drive" && echo "PASS: Structure correct"
  
  # Smoke test: Run 2 epochs (mock, not full training)
  # TODO: Create minimal test that runs 2 epochs and checks loss decreases
  ```

  **Commit**: YES
  - Message: `feat(notebook): add Colab training notebook`
  - Files: `notebooks/train_jepa.ipynb`
  - Pre-commit: `jupyter nbconvert --to python notebooks/train_jepa.ipynb --stdout | python -m py_compile -`

---

- [ ] 9. Eval Script

  **What to do**:
  - Create `scripts/eval.py` CLI script:
    - Argparse: `--checkpoint`, `--manifest`, `--output`, `--config` (optional)
    - Load checkpoint (model state, config)
    - Load eval dataset from manifest
    - Compute embeddings (clean + predicted from masked)
    - Compute cosine similarity metric
    - Save to JSON: `{"mean_cosine_similarity": 0.95, "per_scene": {...}}`
    - Print summary to stdout
  - Support CPU-only (no GPU required)
  - Graceful error handling (missing checkpoint, invalid manifest)

  **Must NOT do**:
  - Add PCA visualization (notebook-only feature)
  - Add dataset generation (eval only, no training)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Standard CLI script, moderate complexity
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 8,10,11)
  - **Parallel Group**: Wave 3 (with Tasks 8,10,11)
  - **Blocks**: None
  - **Blocked By**: Tasks 2,3,4,5,7

  **References**:
  - Eval logic: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/jepa/prediction.ipynb:885-899` (inference and metric computation)
  - Argparse: Python stdlib `argparse.ArgumentParser` pattern

  **Acceptance Criteria**:
  ```bash
  # Script runs and produces output
  python scripts/eval.py --checkpoint experiments/test_ckpt.pt --manifest videomae/clips_manifest.jsonl --output eval_results.json
  test -f eval_results.json && echo "PASS: Output file created"
  
  # Output has correct schema
  python -c "import json; r=json.load(open('eval_results.json')); assert 'mean_cosine_similarity' in r; print('PASS: Schema correct')"
  
  # Metric value in valid range
  python -c "import json; r=json.load(open('eval_results.json')); assert 0 <= r['mean_cosine_similarity'] <= 1; print('PASS: Metric valid')"
  ```

  **Commit**: YES
  - Message: `feat(scripts): add evaluation script`
  - Files: `scripts/eval.py`
  - Pre-commit: `python scripts/eval.py --help`

---

- [ ] 10. Inference Script

  **What to do**:
  - Create `scripts/inference.py` CLI script:
    - Argparse: `--checkpoint`, `--manifest`, `--output`
    - Load checkpoint (model state)
    - Process manifest line-by-line (memory-efficient for large datasets)
    - For each clip: compute novelty score = 1 - cosine_similarity(clean_emb, masked_emb)
    - Write to JSONL: `{"scene": "...", "camera": "...", "first_frame": "...", "score": 0.05}`
    - Print progress every 10 clips
  - Support CPU-only (no GPU required)
  - Batch processing optional (start with single-clip for simplicity)

  **Must NOT do**:
  - Add distributed inference (single machine only)
  - Add model optimizations (ONNX, TensorRT)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Standard CLI script, moderate complexity
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 8,9,11)
  - **Parallel Group**: Wave 3 (with Tasks 8,9,11)
  - **Blocks**: None
  - **Blocked By**: Tasks 2,3,4,5,7

  **References**:
  - Scoring logic: `/Users/shamik/Documents/AV-SSL-Optimization-JEPA/videomae/jepa_mining.py:175-187` (cosine distance computation)
  - JSONL writing: Python `json.dumps() + newline` per line

  **Acceptance Criteria**:
  ```bash
  # Script runs and produces output
  python scripts/inference.py --checkpoint experiments/test_ckpt.pt --manifest videomae/clips_manifest.jsonl --output scores.jsonl
  test -f scores.jsonl && echo "PASS: Output file created"
  
  # Output line count matches input
  test $(wc -l < scores.jsonl) -eq $(wc -l < videomae/clips_manifest.jsonl) && echo "PASS: Line count correct"
  
  # Each line has score field
  head -1 scores.jsonl | python -c "import sys, json; r=json.load(sys.stdin); assert 'score' in r; print('PASS: Schema correct')"
  ```

  **Commit**: YES
  - Message: `feat(scripts): add inference scoring script`
  - Files: `scripts/inference.py`
  - Pre-commit: `python scripts/inference.py --help`

---

- [ ] 11. Smoke Tests + README

  **What to do**:
  - Create `tests/test_data.py`:
    - Test `JEPADataset` loads manifest
    - Test `MaskTubelet` produces correct shapes
  - Create `tests/test_models.py`:
    - Test `VJEPAEncoder` forward pass (mock input)
    - Test `PredictorHead` forward pass
    - Test `JEPAModel` end-to-end forward
  - Create `tests/test_training.py`:
    - Test `jepa_loss` computes non-negative loss
    - Test `train_epoch` runs without errors (1 step)
  - Create `README.md`:
    - Project description
    - Installation instructions (`pip install -e .`)
    - Colab training quickstart
    - Local eval/inference usage examples
    - Checkpoint format documentation
  - Add pytest configuration in `pyproject.toml` or `pytest.ini`

  **Must NOT do**:
  - Add integration tests (full training run) - too slow for CI
  - Add extensive unit tests (100% coverage) - smoke tests only

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: README is documentation, tests are simple validation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 8,9,10)
  - **Parallel Group**: Wave 3 (with Tasks 8,9,10)
  - **Blocks**: None
  - **Blocked By**: Tasks 2,3

  **References**:
  - Pytest patterns: https://docs.pytest.org/en/stable/getting-started.html
  - README structure: Existing `videomae/README.md` as reference

  **Acceptance Criteria**:
  ```bash
  # Tests run and pass
  python -m pytest tests/ -v && echo "PASS: All tests pass"
  
  # README has required sections
  grep -q "## Installation" README.md && echo "PASS: README has installation"
  grep -q "## Training" README.md && echo "PASS: README has training instructions"
  ```

  **Commit**: YES
  - Message: `test: add smoke tests and project README`
  - Files: `tests/test_data.py`, `tests/test_models.py`, `tests/test_training.py`, `README.md`, `pytest.ini`
  - Pre-commit: `python -m pytest tests/`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(repo): initialize production package structure` | src/jepa/, scripts/, configs/, notebooks/ | `test -d src/jepa` |
| 2 | `feat(build): add package configuration` | setup.py, requirements.txt | `pip install -e .` |
| 3 | `feat(config): add default hyperparameters` | configs/default.yaml | `python -c "import yaml; yaml.safe_load(open('configs/default.yaml'))"` |
| 4 | `feat(data): add dataset and masking` | src/jepa/data/*.py | `python -c "from jepa.data import JEPADataset"` |
| 5 | `feat(models): add encoder and predictor` | src/jepa/models/*.py | `python -c "from jepa.models import JEPAModel"` |
| 6 | `feat(training): add training loop` | src/jepa/training/*.py | `python -c "from jepa.training import train_epoch"` |
| 7 | `feat(evaluation): add metrics` | src/jepa/evaluation/*.py | `python -c "from jepa.evaluation import compute_cosine_similarity"` |
| 8 | `feat(notebook): add Colab training` | notebooks/train_jepa.ipynb | `jupyter nbconvert --to python notebooks/train_jepa.ipynb` |
| 9 | `feat(scripts): add eval script` | scripts/eval.py | `python scripts/eval.py --help` |
| 10 | `feat(scripts): add inference script` | scripts/inference.py | `python scripts/inference.py --help` |
| 11 | `test: add smoke tests and README` | tests/, README.md | `pytest tests/` |

---

## Success Criteria

### Verification Commands

**Package Installation**:
```bash
pip install -e .
python -c "import jepa; from jepa.data import JEPADataset; from jepa.models import JEPAModel; print('PASS')"
```

**Config Loading**:
```bash
python -c "import yaml; c=yaml.safe_load(open('configs/default.yaml')); assert c['training']['epochs'] == 30; print('PASS')"
```

**Data Loading**:
```bash
python -c "from jepa.data import JEPADataset; ds=JEPADataset('videomae/clips_manifest.jsonl'); print(f'Loaded {len(ds)} clips')"
```

**Model Forward Pass** (CPU):
```bash
python -c "import torch; from jepa.models import JEPAModel; m=JEPAModel(); m.eval(); clean, pred = m(torch.rand(1,2,3,224,224), torch.rand(1,2,3,224,224), torch.tensor([0.75])); print(f'PASS: clean={clean.shape}, pred={pred.shape}')"
```

**Smoke Tests**:
```bash
python -m pytest tests/ -v
# Expected: 3-5 tests pass, 0 failures
```

**Colab Training** (manual verification on Colab):
```python
# Upload notebook to Colab, run all cells
# Expected: Checkpoint saved to Drive after 30 epochs
```

**Eval Script** (CPU):
```bash
python scripts/eval.py --checkpoint experiments/sample_ckpt.pt --manifest videomae/clips_manifest.jsonl --output eval_out.json
cat eval_out.json | jq '.mean_cosine_similarity'
# Expected: Value between 0.8 and 1.0
```

**Inference Script** (CPU):
```bash
python scripts/inference.py --checkpoint experiments/sample_ckpt.pt --manifest videomae/clips_manifest.jsonl --output scores.jsonl
head -3 scores.jsonl
# Expected: 3 lines with {"scene": ..., "score": ...} format
```

### Final Checklist

- [ ] All 11 tasks completed and committed
- [ ] Package installs without errors (`pip install -e .`)
- [ ] Smoke tests pass (`pytest tests/`)
- [ ] Colab notebook runs end-to-end (manual verification)
- [ ] Eval script produces valid JSON output
- [ ] Inference script produces valid JSONL output
- [ ] README documentation is complete and accurate
- [ ] No modifications to `jepa/*.ipynb` research notebooks
- [ ] Config can be overridden via CLI (test: `--lr 1e-3` changes learning rate)
- [ ] Checkpoint resumption works (train → save → load → continue)
