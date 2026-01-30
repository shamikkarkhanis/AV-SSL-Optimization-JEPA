"""VJEPA Encoder Wrapper"""

import torch
import torch.nn as nn
from transformers import AutoModel


class VJEPAEncoder(nn.Module):
    """Frozen VJEPA encoder from HuggingFace.

    Loads facebook/vjepa2-vitl-fpc64-256 and freezes all parameters.
    Implements encoder usage pattern from jepa/embeddings.ipynb (lines 37-46).

    Args:
        model_name: HuggingFace model identifier
        freeze: Whether to freeze encoder weights (default: True)
    """

    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitl-fpc64-256",
        freeze: bool = True,
    ):
        super().__init__()

        # Load pretrained VJEPA model
        self.model = AutoModel.from_pretrained(model_name)

        # Freeze parameters (no fine-tuning)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()  # Set to evaluation mode

        self.model_name = model_name
        self.embedding_dim = 1024  # VJEPA-L output dimension

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract vision features from pixel values.

        Args:
            pixel_values: Tensor of shape (B, T, C, H, W) or (B, C, H, W)

        Returns:
            Tensor of shape (B, num_patches, embedding_dim)
                where num_patches=256 for VJEPA-L
        """
        # Get vision features (last hidden state)
        # Fix: VJEPA2Model expects 'pixel_values_videos' instead of 'pixel_values'
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(pixel_values_videos=pixel_values)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            return outputs[0]  # Fallback for tuple output

    def get_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract embeddings with global average pooling.

        Implements GAP pattern from jepa/prediction.ipynb (line 166-174).

        Args:
            pixel_values: Tensor of shape (B, T, C, H, W)

        Returns:
            Pooled embedding of shape (B, embedding_dim)
        """
        features = self.forward(pixel_values)  # (B, num_patches, embedding_dim)
        pooled = features.mean(dim=1)  # (B, embedding_dim)
        return pooled
