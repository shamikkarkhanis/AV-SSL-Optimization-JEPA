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


# TODO: optimize for mps (metal performance shaders) for macos !
# load model 
MODEL_NAME = "MCG-NJU/videomae-base" #* https://huggingface.co/MCG-NJU/videomae-base
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    batch: list where each item is a list[16 PIL Images]
    processor will output: pixel_values -> (B, C, T, H, W)
    """
    return processor(batch, return_tensors="pt")
    
    
# send mask clips to model which will encode and decode resulting in generated output clip
def forward_masked_mae(
    pixel_values: torch.Tensor,
    mask_ratio: float = 0.9,
    *,  # force keyword-only after this
    mae_model: VideoMAEForPreTraining = model,
):
    """
    Ensures (B, C, T, H, W) layout, builds a token (tubelet) mask, runs VideoMAE forward,
    and attempts to decode predicted patches back to a video clip.

    Returns: (out, mask, pixel_values_bcthw, predicted_clip_bcthw|None)
    """
    if pixel_values.dim() != 5:
        raise ValueError(f"Expected 5D tensor, got shape {tuple(pixel_values.shape)}")

    # Ensure channel dimension matches model config
    nc = getattr(mae_model.config, "num_channels", 3)
    if pixel_values.shape[1] == nc:
        # already (B, C, T, H, W)
        pass
    elif pixel_values.shape[2] == nc:
        # (B, T, C, H, W) -> (B, C, T, H, W)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
    else:
        raise ValueError(
            f"Cannot infer layout: expected channel dim={nc} at axis 1 or 2, got {tuple(pixel_values.shape)}"
        )

    B, C, T, H, W = pixel_values.shape

    # Normalize config values
    ps = mae_model.config.patch_size
    if not isinstance(ps, int):
        ps = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
    ts = getattr(mae_model.config, "tubelet_size", 1)

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

    # Forward
    out = mae_model(pixel_values=pixel_values, bool_masked_pos=mask)

    # Try to decode predicted tokens back to (B, C, T, H, W)
    def _maybe_get_logits(m):
        for key in ("logits", "reconstruction", "predicted_pixel_values", "preds"):
            if hasattr(m, key):
                return getattr(m, key)
        return None

    preds = _maybe_get_logits(out)
    predicted_clip = None
    if preds is not None and preds.dim() == 3:
        t_tokens = T // ts
        h_tokens = H // ps
        w_tokens = W // ps
        expected_tokens = t_tokens * h_tokens * w_tokens
        expected_dim = C * ts * ps * ps
        if preds.shape[0] == B and preds.shape[1] == expected_tokens and preds.shape[2] == expected_dim:
            x = preds.view(B, t_tokens, h_tokens, w_tokens, C, ts, ps, ps)
            x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
            predicted_clip = x.view(B, C, T, H, W)

    return out, mask, pixel_values, predicted_clip


# perform loss calculation
def compute_mae_loss(out) -> torch.Tensor:
    if getattr(out, "loss", None) is None:
        raise RuntimeError(
            "Model output has no `.loss`. Ensure you're using `VideoMAEForPreTraining` "
            "and provided `bool_masked_pos`, or implement a manual masked-patch loss."
        )
    return out.loss

# load clips and send to masking


if __name__ == "__main__":
    MANIFEST = "clips_manifest.jsonl"
    BATCH_SIZE = 2
    MASK_RATIO = 0.9

    dataset = nuscenes_clips.NuScenesClips(MANIFEST) #* defaults to 16 frames per clip
    
    #* Created an efficient dataloader process my fetching samples in parallel 
    loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, collate_fn = collate_processor)

    # iterate manually so we can catch per-batch errors (including fetch errors)
    iterator = iter(loader)
    step = 0
    while True:
        try:
            batch = next(iterator)
            step += 1
            pixel_values = batch["pixel_values"].to(DEVICE)
            out, mask, pv_bcthw, pred_clip = forward_masked_mae(pixel_values, mask_ratio=MASK_RATIO, mae_model=model)

            loss = compute_mae_loss(out)
            msg = f"batch {step}: loss={loss.item():.6f}, mask_tokens={mask.shape[-1]}"
            if pred_clip is not None:
                msg += f", pred_clip={tuple(pred_clip.shape)}"
            print(msg)
        except StopIteration:
            break
        except Exception as e:
            # log and continue to next batch
            print(f"Step {step}: skipping batch due to error: {e}")
            continue
