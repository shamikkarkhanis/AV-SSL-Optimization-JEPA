"""JEPA Evaluation Metrics"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    """L2 normalize numpy array along axis 1.
    
    Safe normalization (avoids division by zero).
    
    Args:
        arr: Input array of shape (N, D)
        
    Returns:
        Normalized array of shape (N, D)
    """
    denom = np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
    return arr / denom


def compute_cosine_similarity(
    pred_emb: np.ndarray,
    target_emb: np.ndarray,
) -> float:
    """Compute mean cosine similarity between predictions and targets.
    
    Implements metric logic from jepa/prediction.ipynb (lines 1113-1121).
    
    Args:
        pred_emb: Predicted embeddings (N, D)
        target_emb: Target clean embeddings (N, D)
        
    Returns:
        Mean cosine similarity (scalar between -1 and 1)
    """
    # L2 normalize both (critical for cosine similarity)
    pred_norm = l2_normalize(pred_emb)
    target_norm = l2_normalize(target_emb)
    
    # Compute cosine similarity
    # Returns matrix (N, N), we want diagonal elements (corresponding pairs)
    cos_sim_matrix = cosine_similarity(pred_norm, target_norm)
    cos_sims = np.diag(cos_sim_matrix)
    
    return float(cos_sims.mean())


def compute_novelty_score(
    pred_emb: np.ndarray,
    target_emb: np.ndarray,
) -> np.ndarray:
    """Compute novelty score (1 - cosine_similarity).
    
    Used for mining hard clips.
    
    Args:
        pred_emb: Predicted embeddings (N, D)
        target_emb: Target embeddings (N, D)
        
    Returns:
        Novelty scores of shape (N,)
    """
    pred_norm = l2_normalize(pred_emb)
    target_norm = l2_normalize(target_emb)
    
    cos_sim_matrix = cosine_similarity(pred_norm, target_norm)
    cos_sims = np.diag(cos_sim_matrix)
    
    # Novelty = distance = 1 - similarity
    return 1.0 - cos_sims
