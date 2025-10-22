# main clip processing and MAE ViT computations

# local imports
import nuscenes_clips

# stl imports
import os
import json
from typing import List

# third-party imports
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import AutoImageProcessor, VideoMAEForPreTraining #* HuggingFace library 
import renderer


# TODO: optimize for mps (metal performance shaders) for macos !
# load model 
MODEL_NAME = "MCG-NJU/videomae-base" #* https://huggingface.co/MCG-NJU/videomae-base

# Device selection with MPS support (macOS)
DEVICE = (
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = VideoMAEForPreTraining.from_pretrained(MODEL_NAME).to(DEVICE)

# (optional) freeze encoder to use as pretrained frozen backbone
for p in model.videomae.parameters():
    p.requires_grad = False
model.eval()  # keep in eval unless you're training decoder/head


# simple toy model that performs a no-op forward
class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pixel_values, mask=None):
        return pixel_values

toy_model = ToyModel().to(DEVICE).eval()


# perform masking
def make_mask_bool(batch: int, T: int, H: int, W: int, patch_size: int, tublet_size: int, mask_ratio: float = 0.9, device: str = "cpu") -> torch.Tensor:
    """
    Returns a boolean mask of shape (B, num_tokens)
    Tokens are tubelets of size (tublet_size * patch_size^2)
    This is basically a matrix of booleans indicating which tokens are masked and which are not
    """

    assert 0.0 <= mask_ratio <= 1.0
    assert H % patch_size == 0, "H must be divisible by patch_size"
    assert W % patch_size == 0, "W must be divisible by patch_size"
    assert T % tublet_size == 0, "T must be divisible by tublet_size"
    t_tokens = T // tublet_size
    h_tokens = H // patch_size
    w_tokens = W // patch_size
    num_tokens = t_tokens * h_tokens * w_tokens
    k = int(mask_ratio * num_tokens)
    m = torch.zeros((batch, num_tokens), dtype=torch.bool, device=device)
    for b in range(batch):
        idx = torch.randperm(num_tokens, device=device)[:k]
        m[b, idx] = True
    return m

def collate_processor(batch):
    """
    batch: list where each item is either
      - list[16 PIL Images] (legacy) or
      - {"frames": list[16 PIL Images], "meta": {...}}

    Returns a dict from HF processor with:
      - pixel_values -> (B, T, C, H, W)
      - metas -> list of metadata per sample (if available)
    """
    # Extract frames and collect metadata when available
    frames_list = []
    metas = []
    for item in batch:
        if isinstance(item, dict) and "frames" in item:
            frames_list.append(item["frames"])  # list of PIL images
            metas.append(item.get("meta"))
        else:
            frames_list.append(item)
            metas.append(None)

    out = processor(frames_list, return_tensors="pt")
    out["metas"] = metas
    return out
    
    
# send mask clips to model which will encode and decode resulting in generated output clip
def forward_masked_mae(
    pixel_values: torch.Tensor,
    mask_ratio: float = 0.9,
    *,  # force keyword-only after this
    mae_model: VideoMAEForPreTraining = model,
):
    """
    Ensures (B, T, C, H, W) layout, builds a token (tubelet) mask, runs VideoMAE forward,
    and attempts to decode predicted patches back to a video clip.

    Returns: (out, mask, pixel_values_btcwh, predicted_clip_btcwh|None)
    """
    if pixel_values.dim() != 5:
        raise ValueError(f"Expected 5D tensor, got shape {tuple(pixel_values.shape)}")

    # Ensure layout matches VideoMAE expectations: (B, T, C, H, W)
    nc = getattr(mae_model.config, "num_channels", 3)
    if pixel_values.shape[2] == nc:
        # already (B, T, C, H, W)
        pass
    elif pixel_values.shape[1] == nc:
        # (B, C, T, H, W) -> (B, T, C, H, W)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
    else:
        raise ValueError(
            f"Cannot infer layout for VideoMAE: expected channel dim={nc} at axis 1 or 2, got {tuple(pixel_values.shape)}"
        )

    B, T, C, H, W = pixel_values.shape

    # Contract checks: T match config.num_frames, divisibility by patch and tubelet sizes
    cfg = mae_model.config
    expected_T = getattr(cfg, "num_frames", None)
    if expected_T is not None:
        assert T == expected_T, f"T ({T}) must equal config.num_frames ({expected_T})"

    ps = cfg.patch_size
    if not isinstance(ps, int):
        ps = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
    ts = getattr(cfg, "tubelet_size", 1)

    assert H % ps == 0 and W % ps == 0, f"H,W ({H},{W}) must be divisible by patch_size {ps}"
    assert T % ts == 0, f"T ({T}) must be divisible by tubelet_size {ts}"

    # Normalize config values
    # ps, ts already normalized above

    # Build token mask at tubelet level
    mask = make_mask_bool(
        batch=B,
        T=T,
        H=H,
        W=W,
        patch_size=ps,
        tublet_size=ts,
        mask_ratio=mask_ratio,
        device=pixel_values.device,
    )

    # Validate mask shape equals number of tokens
    t_tokens = T // ts
    h_tokens = H // ps
    w_tokens = W // ps
    total_tokens = t_tokens * h_tokens * w_tokens
    assert mask.shape == (B, total_tokens), f"mask shape {tuple(mask.shape)} != (B,total_tokens)=({B},{total_tokens})"

    # Forward
    out = mae_model(pixel_values=pixel_values, bool_masked_pos=mask)

    # Try to decode predicted tokens back to (B, C, T, H, W)
    # Use a single prediction tensor consistently for scoring downstream
    def _pick_prediction(m):
        for key in ("reconstruction", "predicted_pixel_values", "logits"):
            if hasattr(m, key):
                return getattr(m, key)
        return None

    preds = _pick_prediction(out)
    predicted_clip = None
    if preds is not None and preds.dim() == 3:
        # Fill masked tokens with predictions and unmasked tokens with GT for clearer preview
        feat_dim = C * ts * ps * ps
        if preds.shape[0] == B and preds.shape[2] == feat_dim and mask.shape == (B, total_tokens):
            # Tokenize GT frames into same order to fill unmasked
            gt = pixel_values
            gt_grid = gt.view(B, T // ts, ts, C, H // ps, ps, W // ps, ps)
            gt_grid = gt_grid.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
            gt_tokens = gt_grid.view(B, total_tokens, feat_dim)

            full_tokens = gt_tokens.clone()
            full_tokens[mask] = preds.reshape(-1, feat_dim)
            # (B, T//ts, H//ps, W//ps, ts, ps, ps, C)
            x = full_tokens.view(B, t_tokens, h_tokens, w_tokens, ts, ps, ps, C)
            # -> (B, T, H, W, C)
            x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, C)
            # return as (B, T, C, H, W)
            predicted_clip = x.permute(0, 1, 4, 2, 3).contiguous()

    return out, preds, mask, pixel_values, predicted_clip


# perform loss calculation
def compute_mae_loss(out) -> torch.Tensor:
    if getattr(out, "loss", None) is None:
        raise RuntimeError(
            "Model output has no `.loss`. Ensure you're using `VideoMAEForPreTraining` "
            "and provided `bool_masked_pos`, or implement a manual masked-patch loss."
        )
    return out.loss

def compute_per_sample_loss(pixel_values_btcwh: torch.Tensor, mask: torch.Tensor, predictions: torch.Tensor, cfg) -> torch.Tensor:
    """Compute per-sample MSE over masked patches, exactly mirroring VideoMAE target construction.

    Returns: tensor of shape (B,) with mean loss per sample.
    """
    with torch.no_grad():
        frames = pixel_values_btcwh  # keep processor-normalized inputs

        B, T, C, H, W = frames.shape
        ts = cfg.tubelet_size
        ps = cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0]

        # step 1: split up dimensions (time by tubelet_size, height by patch_size, width by patch_size)
        frames = frames.view(
            B,
            T // ts,
            ts,
            C,
            H // ps,
            ps,
            W // ps,
            ps,
        )
        # step 2: move dimensions to concatenate:
        # (B, T//ts, H//ps, W//ps, ts, ps, ps, C)
        frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()

        if getattr(cfg, "norm_pix_loss", False):
            # step 3a: normalize per patch over (ts*ps*ps)
            frames_norm = (frames - frames.mean(dim=-2, keepdim=True)) / (
                frames.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
            )
            # step 4a: reshape to (B, tokens, ts*ps*ps*C)
            videos_patch = frames_norm.view(
                B,
                (T // ts) * (H // ps) * (W // ps),
                ts * ps * ps * C,
            )
        else:
            # step 3b: reshape without normalization
            videos_patch = frames.view(
                B,
                (T // ts) * (H // ps) * (W // ps),
                ts * ps * ps * C,
            )

        # labels for masked positions -> (B, N_masked, F)
        labels = videos_patch[mask].reshape(B, -1, videos_patch.shape[-1])

    # MSE reduced over feature dim first, then tokens
    diff2 = (predictions - labels) ** 2  # (B, N_masked, F)
    per_token = diff2.mean(dim=-1)      # (B, N_masked)
    per_sample = per_token.mean(dim=-1) # (B,)
    return per_sample

# load clips and send to masking


if __name__ == "__main__":
    MANIFEST = "clips_manifest.jsonl"
    BATCH_SIZE = 2
    MASK_RATIO = 0.1
    SCORES_OUT = os.environ.get("SCORES_OUT", "scores.jsonl")
    BASELINE_STATS = os.environ.get("BASELINE_STATS")  # optional path with per-camera mean/std JSON

    # Determinism: seed RNGs (helps for mask generation and shuffling)
    import random, numpy as np, platform, json
    SEED = int(os.environ.get("SEED", "42"))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    dataset = nuscenes_clips.NuScenesClips(MANIFEST) #* defaults to 16 frames per clip
    
    #* Created an efficient dataloader process my fetching samples in parallel 
    # On macOS with MPS, multiprocess dataloading can be flaky; prefer num_workers=0
    nworkers = 0 if (DEVICE == "mps" or platform.system() == "Darwin") else 4
    loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = nworkers, collate_fn = collate_processor)

    print(f"dataset_size={len(dataset)}, batch_size={BATCH_SIZE}")
    # Note: len(loader) equals number of batches if no worker errors occur
    try:
        print(f"expected_batches={len(loader)} (drop_last=False)")
    except Exception:
        pass

    # iterate manually so we can catch per-batch errors (including fetch errors)
    iterator = iter(loader)
    step = 0
    while True:
        try:
            batch = next(iterator)
            step += 1
            pixel_values = batch["pixel_values"].to(DEVICE)
            metas = batch.get("metas", [None] * pixel_values.shape[0])
            out, preds, mask, pv_btcwh, pred_clip = forward_masked_mae(pixel_values, mask_ratio=MASK_RATIO, mae_model=model)

            loss = compute_mae_loss(out)
            # Select one prediction tensor consistently for per-sample scoring
            pred_tensor = preds
            per_sample = compute_per_sample_loss(pv_btcwh, mask, pred_tensor, model.config)
            # Identify which clips were fed to the model this batch
            idents = []
            for m in metas:
                if isinstance(m, dict):
                    scene = m.get("scene") or "?scene"
                    cam = m.get("camera") or "?cam"
                    first = (m.get("frames") or [None])[0]
                    ident = f"{scene}/{cam} [{os.path.basename(first) if first else 'unknown'}]"
                else:
                    ident = "unknown"
                idents.append(ident)

            msg = (
                f"batch {step}: clips={idents}, "
                f"loss={loss.item():.6f}, per_sample={[round(x,6) for x in per_sample.tolist()]}, "
                f"mask_tokens={mask.shape[-1]}"
            )
            if pred_clip is not None:
                msg += f", pred_clip={tuple(pred_clip.shape)}"
            print(msg)

            # Persist per-sample scores with metadata for later calibration
            try:
                os.makedirs(os.path.dirname(SCORES_OUT) or ".", exist_ok=True)
                with open(SCORES_OUT, "a") as f:
                    for i, m in enumerate(metas):
                        rec = {
                            "step": step,
                            "scene": (m or {}).get("scene"),
                            "camera": (m or {}).get("camera"),
                            "first_frame": os.path.basename(((m or {}).get("frames") or [None])[0] or ""),
                            "score": float(per_sample[i].detach().cpu().item()),
                        }
                        f.write(json.dumps(rec) + "\n")
            except Exception as we:
                print(f"batch {step}: score logging failed: {we}")

            # Render GT vs Pred frames used for loss visualization
            try:
                renderer.render_batch(
                    step=step,
                    pv_btcwh=pv_btcwh.detach().cpu(),
                    pred_clip_btcwh=pred_clip.detach().cpu() if pred_clip is not None else None,
                    out_logits=pred_tensor.detach().cpu(),
                    mask=mask.detach().cpu(),
                    cfg=model.config,
                    out_dir="renders",
                    frame_index=None,  # middle frame by default
                )
            except Exception as re:
                print(f"batch {step}: render skipped due to error: {re}")
        except StopIteration:
            break
        except Exception as e:
            # log and continue to next batch
            print(f"Step {step}: skipping batch due to error: {e}")
            continue
