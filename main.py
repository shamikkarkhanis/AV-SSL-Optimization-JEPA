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


# perform loss calculation




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
            # standardize to (B, C, T, H, W); processor often yields (B, T, C, H, W)
            if pixel_values.shape[1] in (1, 3):
                B, C, T, H, W = pixel_values.shape
                pv = pixel_values
            else:
                B, T, C, H, W = pixel_values.shape
                pv = pixel_values.permute(0, 2, 1, 3, 4).contiguous()

            # compute a random boolean mask over tokens
            mask = make_mask_bool(
                batch=B,
                T=T,
                H=H,
                W=W,
                patch_size=16,
                tublet_size=2,
                mask_ratio=MASK_RATIO,
                device=DEVICE,
            )
            # send to toy model (no-op)
            out = toy_model(pv, mask)
            print(f"batch {step}: {tuple(out.shape)} (mask_tokens={mask.shape[-1]})")
        except StopIteration:
            break
        except Exception as e:
            # log and continue to next batch
            print(f"Step {step}: skipping batch due to error: {e}")
            continue
