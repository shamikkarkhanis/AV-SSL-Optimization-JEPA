"""Smoke tests for Training Utilities"""

import pytest
import torch
from src.jepa.training import jepa_loss

def test_jepa_loss():
    """Test JEPA loss computation."""
    batch_size = 2
    emb_dim = 1024
    
    pred = torch.randn(batch_size, emb_dim)
    target = torch.randn(batch_size, emb_dim)
    
    # With normalization
    loss = jepa_loss(pred, target, normalize=True)
    assert loss.item() >= 0
    
    # Without normalization
    loss_raw = jepa_loss(pred, target, normalize=False)
    assert loss_raw.item() >= 0
