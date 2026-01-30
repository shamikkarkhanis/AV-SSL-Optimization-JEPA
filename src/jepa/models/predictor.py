"""JEPA Predictor Head"""

import torch
import torch.nn as nn


class PredictorHead(nn.Module):
    """Predictor head for JEPA.
    
    Architecture extracted from jepa/prediction.ipynb (lines 737-762).
    Predicts clean embeddings from masked embeddings + mask fraction.
    
    Structure:
    1. LayerNorm(emb_dim)
    2. Linear(emb_dim + 1 -> emb_dim)  (concatenated with mask_fraction)
    3. GELU()
    4. Dropout(0.1)
    5. Linear(emb_dim -> emb_dim)
    
    Args:
        emb_dim: Embedding dimension (default: 1024)
        hidden_dim: Hidden layer dimension (default: 1024)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        emb_dim: int = 1024,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Layer normalization applied to embeddings
        self.ln = nn.LayerNorm(emb_dim)
        
        # MLP architecture
        # Input dim is emb_dim + 1 (for mask_fraction conditioning)
        self.net = nn.Sequential(
            nn.Linear(emb_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim)
        )
    
    def forward(self, masked_emb: torch.Tensor, mask_frac: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            masked_emb: Masked embeddings of shape (B, emb_dim)
            mask_frac: Mask fraction scalar or tensor of shape (B, 1)
            
        Returns:
            Predicted clean embeddings of shape (B, emb_dim)
        """
        # Normalize input embeddings (Line 751)
        x = self.ln(masked_emb)
        
        # Ensure mask_frac has correct shape (B, 1)
        if mask_frac.dim() == 0:
            mask_frac = mask_frac.unsqueeze(0).expand(x.size(0), 1)
        elif mask_frac.dim() == 1:
            mask_frac = mask_frac.unsqueeze(1)
        
        # Concatenate embeddings with mask fraction (Line 753)
        # x: (B, emb_dim), mask_frac: (B, 1) -> combined: (B, emb_dim + 1)
        x_combined = torch.cat([x, mask_frac], dim=1)
        
        # Pass through MLP
        output = self.net(x_combined)
        
        return output
