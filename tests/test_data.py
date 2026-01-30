"""Smoke tests for Data Module"""

import pytest
import torch
from PIL import Image
from src.jepa.data import MaskTubelet

def test_mask_tubelet_shapes():
    """Test if MaskTubelet produces correct tensor shapes."""
    # Create mock tubelet (2 frames of 224x224 RGB)
    frames = [Image.new('RGB', (224, 224)) for _ in range(2)]
    
    transform = MaskTubelet(mask_ratio=0.75, patch_size=16)
    output = transform(frames)
    
    # Check keys
    assert "clean_frames" in output
    assert "masked_frames" in output
    assert "mask" in output
    assert "mask_frac" in output
    
    # Check shapes
    # (T, C, H, W)
    assert output["clean_frames"].shape == (2, 3, 224, 224)
    assert output["masked_frames"].shape == (2, 3, 224, 224)
    
    # Mask shape: (T, H/patch, W/patch) -> (2, 14, 14)
    assert output["mask"].shape == (2, 14, 14)
    
    # Check mask fraction
    assert 0.7 <= output["mask_frac"].item() <= 0.8  # Approx 75%
