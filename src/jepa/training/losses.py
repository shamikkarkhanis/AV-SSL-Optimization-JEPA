"""JEPA Loss Functions"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss(nn.Module):
    """L1 Loss on L2-normalized embeddings.
    
    Implements loss logic from jepa/prediction.ipynb (lines 843-846).
    
    Why L1 on normalized?
    - Normalization ensures embeddings lie on hypersphere (cosine-like space)
    - L1 is more robust to outliers than MSE
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        self.criterion = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss.
        
        Args:
            pred: Predicted embeddings (B, D)
            target: Target (clean) embeddings (B, D)
            
        Returns:
            Scalar loss
        """
        if self.normalize:
            # L2 normalize embeddings (dim=1)
            pred = F.normalize(pred, p=2, dim=1)
            target = F.normalize(target, p=2, dim=1)
            
        return self.criterion(pred, target)


def jepa_loss(pred: torch.Tensor, target: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Functional interface for JEPA loss."""
    if normalize:
        pred = F.normalize(pred, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)
    return F.l1_loss(pred, target)
