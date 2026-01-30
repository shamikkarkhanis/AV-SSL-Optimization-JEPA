"""Transforms for JEPA Training - Masking Logic"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


class MaskTubelet:
    """Mask tubelet pixels for JEPA training.
    
    Implements masking logic from jepa/prediction.ipynb (lines 238-286).
    Masks random spatial patches across all frames in the tubelet.
    
    Args:
        mask_ratio: Fraction of patches to mask (default: 0.75)
        patch_size: Size of spatial patches (default: 16x16)
        fill_value: Value to fill masked pixels (default: 0, black)
        seed: Random seed for reproducibility (default: None)
    
    Returns:
        Dictionary with:
            - clean_frames: Original tubelet tensor (T, C, H, W)
            - masked_frames: Tubelet with masked patches (T, C, H, W)
            - mask: Boolean mask array (T, H/patch_size, W/patch_size)
            - mask_frac: Actual mask fraction as tensor
    """
    
    def __init__(
        self,
        mask_ratio: float = 0.75,
        patch_size: int = 16,
        fill_value: int = 0,
        seed: Optional[int] = None,
    ):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.fill_value = fill_value
        self.seed = seed
        
        # PIL to tensor transform
        self.to_tensor = T.ToTensor()
    
    def __call__(self, frames: List[Image.Image]) -> dict:
        """Apply masking to a list of PIL Images.
        
        Args:
            frames: List of PIL Images (tubelet)
        
        Returns:
            Dictionary with clean_frames, masked_frames, mask, mask_frac
        """
        # Convert PIL Images to numpy array: (T, H, W, C)
        tubelet = np.stack([np.array(img) for img in frames], axis=0)
        
        # Apply masking
        masked_tubelet, mask = mask_tubelet_pixels(
            tubelet,
            mask_ratio=self.mask_ratio,
            patch_size=self.patch_size,
            seed=self.seed,
            fill_value=self.fill_value,
        )
        
        # Convert to tensors (T, C, H, W)
        clean_tensor = torch.from_numpy(tubelet).permute(0, 3, 1, 2).float() / 255.0
        masked_tensor = torch.from_numpy(masked_tubelet).permute(0, 3, 1, 2).float() / 255.0
        mask_tensor = torch.from_numpy(mask)
        
        # Calculate actual mask fraction
        mask_frac = mask.sum() / mask.size
        
        return {
            "clean_frames": clean_tensor,
            "masked_frames": masked_tensor,
            "mask": mask_tensor,
            "mask_frac": torch.tensor(mask_frac, dtype=torch.float32),
        }


def mask_tubelet_pixels(
    tubelet: np.ndarray,
    mask_ratio: float = 0.75,
    patch_size: int = 16,
    seed: Optional[int] = None,
    fill_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mask random patches in a tubelet.
    
    Exact implementation from jepa/prediction.ipynb lines 238-286.
    
    Args:
        tubelet: Numpy array of shape (T, H, W, C)
        mask_ratio: Fraction of patches to mask (0.0 to 1.0)
        patch_size: Size of spatial patches (must divide H and W evenly)
        seed: Random seed for reproducibility
        fill_value: Value to fill masked pixels (0-255)
    
    Returns:
        Tuple of (masked_tubelet, mask_tokens)
            - masked_tubelet: Tubelet with masked patches filled (T, H, W, C)
            - mask_tokens: Boolean array marking masked patches (T, H//patch_size, W//patch_size)
    """
    T, H, W, C = tubelet.shape
    
    # Validate divisibility
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(
            f"Image dimensions ({H}x{W}) must be divisible by patch_size ({patch_size})"
        )
    
    # Calculate patch grid dimensions
    pH = H // patch_size  # Number of patches vertically
    pW = W // patch_size  # Number of patches horizontally
    num_patches = T * pH * pW
    
    # Determine number of patches to mask
    num_masked = int(mask_ratio * num_patches)
    
    # Create random patch indices
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    all_indices = np.arange(num_patches)
    rng.shuffle(all_indices)
    masked_indices = all_indices[:num_masked]
    
    # Create mask tokens (3D boolean array)
    mask_tokens = np.zeros((T, pH, pW), dtype=bool)
    
    # Mark masked patches in 3D grid
    for idx in masked_indices:
        t = idx // (pH * pW)  # Temporal index
        spatial_idx = idx % (pH * pW)
        i = spatial_idx // pW  # Patch row
        j = spatial_idx % pW   # Patch column
        mask_tokens[t, i, j] = True
    
    # Apply masking to pixels
    masked_tubelet = tubelet.copy()
    for t in range(T):
        for i in range(pH):
            for j in range(pW):
                if mask_tokens[t, i, j]:
                    # Fill entire patch with fill_value
                    h_start = i * patch_size
                    h_end = h_start + patch_size
                    w_start = j * patch_size
                    w_end = w_start + patch_size
                    masked_tubelet[t, h_start:h_end, w_start:w_end, :] = fill_value
    
    return masked_tubelet, mask_tokens


class VideoProcessor:
    """Wrapper for HuggingFace VideoProcessor compatibility.
    
    Applies standard preprocessing: resize, normalize, convert to tensor.
    Compatible with transformers.AutoVideoProcessor output format.
    """
    
    def __init__(
        self,
        size: int = 224,
        mean: List[float] = [0.485, 0.456, 0.406],  # ImageNet stats
        std: List[float] = [0.229, 0.224, 0.225],
    ):
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    
    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        """Process list of frames to tensor.
        
        Args:
            frames: List of PIL Images
        
        Returns:
            Tensor of shape (T, C, H, W)
        """
        return torch.stack([self.transform(frame) for frame in frames])
