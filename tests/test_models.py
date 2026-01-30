"""Smoke tests for Model Components"""

import pytest
import torch
from src.jepa.models import PredictorHead, JEPAModel

def test_predictor_forward():
    """Test PredictorHead forward pass."""
    batch_size = 2
    emb_dim = 1024
    
    model = PredictorHead(emb_dim=emb_dim, hidden_dim=1024)
    
    masked_emb = torch.randn(batch_size, emb_dim)
    mask_frac = torch.tensor([[0.75], [0.5]])
    
    output = model(masked_emb, mask_frac)
    
    assert output.shape == (batch_size, emb_dim)

def test_jepa_initialization():
    """Test JEPAModel initialization (no forward pass to avoid loading weights)."""
    model = JEPAModel(freeze_encoder=True)
    
    # Check encoder is frozen
    for param in model.encoder.parameters():
        assert param.requires_grad is False
        
    # Check predictor is trainable
    for param in model.predictor.parameters():
        assert param.requires_grad is True
