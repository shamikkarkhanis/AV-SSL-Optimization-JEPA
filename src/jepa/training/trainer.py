"""JEPA Training Loop"""

import logging
from typing import Dict, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import jepa_loss


logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for JEPA model.
    
    Implements training loop from jepa/prediction.ipynb (lines 836-867).
    Handles training, validation, and checkpointing.
    
    Args:
        model: JEPAModel instance
        optimizer: PyTorch optimizer
        device: torch.device
        checkpoint_dir: Directory to save checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: Union[str, Path] = "experiments/checkpoints",
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Run one training epoch.
        
        Args:
            loader: Training DataLoader
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        # Only set predictor to train mode (encoder stays frozen)
        self.model.encoder.eval()
        self.model.predictor.train()
        
        total_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
        for batch in pbar:
            # Move data to device
            masked = batch["masked_frames"].to(self.device)
            clean = batch["clean_frames"].to(self.device)
            mask_frac = batch["mask_frac"].to(self.device)
            
            # Forward pass
            # clean_emb: Target, pred_emb: Prediction
            clean_emb, pred_emb = self.model(clean, masked, mask_frac)
            
            # Calculate loss (normalized L1)
            loss = jepa_loss(pred_emb, clean_emb, normalize=True)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            loss_val = loss.item()
            total_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            
        return total_loss / len(loader)
    
    def validate_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Run validation epoch.
        
        Args:
            loader: Validation DataLoader
            epoch: Current epoch number
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Val Epoch {epoch}")
            for batch in pbar:
                masked = batch["masked_frames"].to(self.device)
                clean = batch["clean_frames"].to(self.device)
                mask_frac = batch["mask_frac"].to(self.device)
                
                clean_emb, pred_emb = self.model(clean, masked, mask_frac)
                loss = jepa_loss(pred_emb, clean_emb, normalize=True)
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch}: Val Loss = {avg_loss:.4f}")
        return avg_loss
    
    def save_checkpoint(
        self, 
        epoch: int, 
        val_loss: float, 
        is_best: bool = False,
        config: Optional[Dict] = None
    ):
        """Save training checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
            config: Configuration dictionary to save
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": config,
        }
        
        # Save epoch checkpoint
        filename = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, filename)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with loss {val_loss:.4f} to {best_path}")
