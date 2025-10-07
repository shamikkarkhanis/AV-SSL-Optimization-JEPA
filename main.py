# main clip processing and MAE ViT computations

# stl imports
import os
import json
from typing import List

# third-party imports
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from Pil import Image
from transformer import AutoImageProcessor, VideoMAEForPreTraining #* HuggingFace library 


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


# perform masking
def make_mask_bool(batch: int, T: int, H: int, W: int, patch_size: int, tublet_size: int, mask_ratio: float = 0.9, device: str = "cpu") -> torch.Tensor:
    """
    Returns a boolean mask of shape (B, num_tokens)
    Tokens are tubelets of size (tublet_size * patch_size^2)
    This is basically a matrix of booleans indicating which tokens are masked and which are not
    """

    assert T % tubelet_size == 0
    t_tokens = T // tubelet_size
    h_tokens = H // patch_size
    w_tokens = W // patch_size
    num_tokens = t_tokens * h_tokens * w_tokens
    k = int(mask_ratio * num_tokens)
    m = torch.zeros((batch, num_tokens), dtype=torch.bool)
    for b in range(batch):
        idx = torch.randperm(num_tokens)[:k]
        m[b, idx] = True
    return m.to(device)
    
    
# send mask clips to model which will encode and decode resulting in generated output clip
#
#
#
#
#

# perform loss calculation
#
#
#


# load clips and send to masking
#
#
