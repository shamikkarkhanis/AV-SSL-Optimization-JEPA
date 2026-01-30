"""JEPA Model (Encoder + Predictor)"""

import torch
import torch.nn as nn
from typing import Tuple

from .encoder import VJEPAEncoder
from .predictor import PredictorHead


class JEPAModel(nn.Module):
    """Joint Embedding Predictive Architecture (JEPA) Model.
    
    Combines a frozen VJEPA encoder and a trainable predictor head.
    
    Args:
        encoder_name: HuggingFace model ID for encoder
        predictor_hidden: Hidden dimension for predictor MLP
        predictor_dropout: Dropout for predictor
        freeze_encoder: Whether to freeze encoder weights
    """
    
    def __init__(
        self,
        encoder_name: str = "facebook/vjepa2-vitl-fpc64-256",
        predictor_hidden: int = 1024,
        predictor_dropout: float = 0.1,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        
        # Encoder (frozen by default)
        self.encoder = VJEPAEncoder(model_name=encoder_name, freeze=freeze_encoder)
        
        # Predictor (trainable)
        self.predictor = PredictorHead(
            emb_dim=self.encoder.embedding_dim,
            hidden_dim=predictor_hidden,
            dropout=predictor_dropout
        )
    
    def forward(
        self,
        clean_frames: torch.Tensor,
        masked_frames: torch.Tensor,
        mask_frac: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """End-to-end forward pass.
        
        Args:
            clean_frames: Original frames (B, T, C, H, W)
            masked_frames: Masked frames (B, T, C, H, W)
            mask_frac: Mask fraction (B, 1) or scalar
            
        Returns:
            Tuple (clean_emb, predicted_emb)
                - clean_emb: Target embeddings from clean frames
                - predicted_emb: Predicted embeddings from masked frames
        """
        # 1. Encode clean frames (Target)
        # Note: Encoder handles no_grad internally based on its training mode
        clean_emb = self.encoder.get_embedding(clean_frames)  # (B, 1024)
        
        # 2. Encode masked frames (Context)
        masked_emb = self.encoder.get_embedding(masked_frames) # (B, 1024)
        
        # 3. Predict clean embeddings from masked embeddings + mask fraction
        pred_emb = self.predictor(masked_emb, mask_frac)      # (B, 1024)
        
        return clean_emb, pred_emb
