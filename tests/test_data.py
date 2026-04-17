"""Smoke tests for Data Module"""

import torch
from PIL import Image
from jepa.data import MaskTubelet, VideoProcessor

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


def test_mask_tubelet_seeded_calls_produce_different_masks():
    """A fixed base seed should still yield different masks across samples."""
    frames = [Image.new("RGB", (224, 224)) for _ in range(2)]
    transform = MaskTubelet(mask_ratio=0.75, patch_size=16, seed=42)

    out1 = transform(frames)
    out2 = transform(frames)

    assert not torch.equal(out1["mask"], out2["mask"])


def test_mask_tubelet_applies_frame_processor_normalization():
    """Frame processor should normalize tensors before they reach the encoder."""
    frames = [Image.new("RGB", (224, 224), color=(0, 0, 0)) for _ in range(2)]
    transform = MaskTubelet(
        mask_ratio=0.0,
        patch_size=16,
        frame_processor=VideoProcessor(size=224),
    )

    output = transform(frames)

    expected = torch.tensor(
        [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        dtype=output["clean_frames"].dtype,
    )
    assert torch.allclose(output["clean_frames"][0, :, 0, 0], expected, atol=1e-4)
