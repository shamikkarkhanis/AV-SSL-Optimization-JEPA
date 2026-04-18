"""JEPA training loop."""

import logging
from contextlib import nullcontext
from typing import Callable, Dict, Optional, Union
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
        amp_context_factory: Optional[Callable[[], object]] = None,
        grad_accum_steps: int = 1,
        normalize_embeddings: bool = True,
        encoder_mode: str = "frozen",
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.amp_context_factory = amp_context_factory or nullcontext
        self.grad_accum_steps = max(int(grad_accum_steps), 1)
        self.normalize_embeddings = normalize_embeddings
        self.encoder_mode = encoder_mode
        
    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Run one training epoch.
        
        Args:
            loader: Training DataLoader
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        if self.encoder_mode == "frozen":
            self.model.encoder.eval()
        else:
            self.model.encoder.train()
        self.model.predictor.train()
        
        total_loss = 0.0
        
        self.optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
        for step_idx, batch in enumerate(pbar, start=1):
            non_blocking = self.device.type == "cuda"
            # Move data to device
            masked = batch["masked_frames"].to(self.device, non_blocking=non_blocking)
            clean = batch["clean_frames"].to(self.device, non_blocking=non_blocking)
            mask_frac = batch["mask_frac"].to(self.device, non_blocking=non_blocking)
            
            # Forward pass
            # clean_emb: Target, pred_emb: Prediction
            with self.amp_context_factory():
                clean_emb, pred_emb = self.model(clean, masked, mask_frac)
                loss = jepa_loss(
                    pred_emb,
                    clean_emb,
                    normalize=self.normalize_embeddings,
                ) / self.grad_accum_steps

            loss.backward()

            if step_idx % self.grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            loss_val = loss.item() * self.grad_accum_steps
            total_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        if len(loader) % self.grad_accum_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
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
                non_blocking = self.device.type == "cuda"
                masked = batch["masked_frames"].to(self.device, non_blocking=non_blocking)
                clean = batch["clean_frames"].to(self.device, non_blocking=non_blocking)
                mask_frac = batch["mask_frac"].to(self.device, non_blocking=non_blocking)
                
                with self.amp_context_factory():
                    clean_emb, pred_emb = self.model(clean, masked, mask_frac)
                    loss = jepa_loss(
                        pred_emb,
                        clean_emb,
                        normalize=self.normalize_embeddings,
                    )
                
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
