"""
Binary "Interesting Clip" Classifier using Video MAE Encoder

This module implements a binary classifier that determines if a video clip is "interesting"
based on temporal embedding change measured via cosine similarity.

A clip is considered interesting if it exhibits unusual or significant semantic change
over time (e.g., rare events, abrupt motion, occlusions, unexpected interactions).

Methodology:
- Uses only the pretrained Video MAE encoder (no decoder, no training)
- Extracts per-frame embeddings
- Computes frame-to-frame cosine distances
- Aggregates distances and thresholds to produce binary prediction
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Union, Optional
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, VideoMAEModel


# Device selection with MPS support (macOS)
DEVICE = (
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Default configuration
HF_MODEL_NAME = "MCG-NJU/videomae-base"
DEFAULT_FRAME_SAMPLE_RATE = 1  # Process every frame
DEFAULT_AGGREGATION_METHOD = "mean"  # "mean" or "top_k"
DEFAULT_TOP_K = None  # Number of top distances to use (only for top_k aggregation)
DEFAULT_THRESHOLD = 0.1  # Default threshold (should be tuned on validation set)


class InterestingClipClassifier:
    """
    Binary classifier for determining if a video clip is interesting.
    
    Uses temporal embedding change measured via cosine similarity between consecutive frames.
    """
    
    def __init__(
        self,
        model_name: str = HF_MODEL_NAME,
        device: str = DEVICE,
        frame_sample_rate: int = DEFAULT_FRAME_SAMPLE_RATE,
        aggregation_method: str = DEFAULT_AGGREGATION_METHOD,
        top_k: Optional[int] = DEFAULT_TOP_K,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: HuggingFace model name for VideoMAE
            device: Device to run inference on
            frame_sample_rate: Sample every Nth frame (1 = all frames)
            aggregation_method: "mean" or "top_k" for aggregating distances
            top_k: Number of top distances to use (only for top_k aggregation)
            threshold: Threshold for binary classification (score >= threshold -> interesting)
        """
        self.device = device
        self.frame_sample_rate = frame_sample_rate
        self.aggregation_method = aggregation_method
        self.top_k = top_k
        self.threshold = threshold
        
        # Load processor and model
        print(f"Loading VideoMAE model: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Get model configuration
        self.num_frames = getattr(self.model.config, 'num_frames', 16)
        print(f"Model expects {self.num_frames} frames per clip")
    
    def extract_frame_embeddings(
        self,
        frames: List[Image.Image],
        batch_size: int = 8
    ) -> torch.Tensor:
        """
        Extract embeddings for each frame in the clip.
        
        For each frame, we process it as a single-frame video by repeating it
        to match the model's expected number of frames.
        
        Args:
            frames: List of PIL Images representing the video clip
            batch_size: Number of frames to process in parallel (for efficiency)
            
        Returns:
            embeddings: Tensor of shape (num_frames, hidden_size) with per-frame embeddings
        """
        num_frames = len(frames)
        embeddings_list = []
        
        with torch.no_grad():
            # Process frames in batches for efficiency
            for i in range(0, num_frames, batch_size):
                batch_frames = frames[i:i+batch_size]
                
                # Prepare batch: each frame repeated to form a "video"
                batch_sequences = []
                for frame in batch_frames:
                    frame_sequence = [frame] * self.num_frames
                    batch_sequences.append(frame_sequence)
                
                # Process batch
                processed = self.processor(batch_sequences, return_tensors="pt")
                pixel_values = processed['pixel_values']  # (B, T, C, H, W)
                pixel_values = pixel_values.to(self.device)
                
                # Forward through encoder
                outputs = self.model(pixel_values=pixel_values)
                
                # Extract embeddings: average over all tokens (spatial + temporal)
                # last_hidden_state shape: (B, num_tokens, hidden_size)
                hidden_states = outputs.last_hidden_state  # (B, num_tokens, hidden_size)
                batch_embeddings = hidden_states.mean(dim=1)  # (B, hidden_size)
                
                embeddings_list.append(batch_embeddings)
        
        # Concatenate all batches and return
        embeddings = torch.cat(embeddings_list, dim=0)  # (num_frames, hidden_size)
        return embeddings
    
    def compute_cosine_distances(
        self,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute frame-to-frame cosine distances.
        
        For consecutive embeddings e_t and e_{t+1}, compute:
            d_t = 1 - cosine_similarity(e_t, e_{t+1})
        
        Args:
            embeddings: Tensor of shape (num_frames, hidden_size)
            
        Returns:
            distances: Tensor of shape (num_frames - 1,) with cosine distances
        """
        num_frames = embeddings.shape[0]
        
        if num_frames < 2:
            # Need at least 2 frames to compute distances
            return torch.tensor([0.0], device=embeddings.device)
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # (num_frames, hidden_size)
        
        # Compute cosine similarity between consecutive frames
        # embeddings_norm[:-1] -> (num_frames-1, hidden_size)
        # embeddings_norm[1:] -> (num_frames-1, hidden_size)
        cosine_similarities = (embeddings_norm[:-1] * embeddings_norm[1:]).sum(dim=1)  # (num_frames-1,)
        
        # Convert to distances: d = 1 - similarity
        distances = 1.0 - cosine_similarities  # (num_frames-1,)
        
        return distances
    
    def aggregate_distances(
        self,
        distances: torch.Tensor
    ) -> float:
        """
        Aggregate frame-to-frame distances into a single clip-level score.
        
        Args:
            distances: Tensor of shape (num_distances,) with cosine distances
            
        Returns:
            score: Scalar score for the clip
        """
        if len(distances) == 0:
            return 0.0
        
        distances_np = distances.cpu().numpy()
        
        if self.aggregation_method == "mean":
            score = float(np.mean(distances_np))
        elif self.aggregation_method == "top_k":
            if self.top_k is None:
                raise ValueError("top_k must be specified when using top_k aggregation")
            k = min(self.top_k, len(distances_np))
            if k == 0:
                return 0.0
            # Get top-k largest distances
            top_k_distances = np.partition(distances_np, -k)[-k:]
            score = float(np.mean(top_k_distances))
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        return score
    
    def is_interesting(
        self,
        clip: Union[List[Image.Image], List[str], str]
    ) -> int:
        """
        Determine if a clip is interesting.
        
        Args:
            clip: Can be:
                - List of PIL Images
                - List of image file paths
                - Directory path containing frame images
        
        Returns:
            1 if clip is interesting, 0 otherwise
        """
        # Load frames if needed
        frames = self._load_frames(clip)
        
        if len(frames) < 2:
            # Need at least 2 frames to compute temporal change
            return 0
        
        # Sample frames at specified rate
        sampled_frames = frames[::self.frame_sample_rate]
        
        if len(sampled_frames) < 2:
            # Need at least 2 sampled frames
            return 0
        
        # Extract embeddings
        embeddings = self.extract_frame_embeddings(sampled_frames)
        
        # Compute cosine distances
        distances = self.compute_cosine_distances(embeddings)
        
        # Aggregate distances
        score = self.aggregate_distances(distances)
        
        # Threshold to get binary prediction
        prediction = 1 if score >= self.threshold else 0
        
        return prediction
    
    def compute_score(
        self,
        clip: Union[List[Image.Image], List[str], str]
    ) -> float:
        """
        Compute the raw score for a clip (before thresholding).
        
        This is useful for threshold selection on a validation set.
        
        Args:
            clip: Can be:
                - List of PIL Images
                - List of image file paths
                - Directory path containing frame images
        
        Returns:
            score: Raw score (aggregated cosine distance)
        """
        # Load frames if needed
        frames = self._load_frames(clip)
        
        if len(frames) < 2:
            return 0.0
        
        # Sample frames at specified rate
        sampled_frames = frames[::self.frame_sample_rate]
        
        if len(sampled_frames) < 2:
            return 0.0
        
        # Extract embeddings
        embeddings = self.extract_frame_embeddings(sampled_frames)
        
        # Compute cosine distances
        distances = self.compute_cosine_distances(embeddings)
        
        # Aggregate distances
        score = self.aggregate_distances(distances)
        
        return score
    
    def _load_frames(
        self,
        clip: Union[List[Image.Image], List[str], str]
    ) -> List[Image.Image]:
        """
        Load frames from various input formats.
        
        Args:
            clip: Can be:
                - List of PIL Images
                - List of image file paths
                - Directory path containing frame images
        
        Returns:
            List of PIL Images
        """
        if isinstance(clip, list):
            if len(clip) == 0:
                return []
            
            # Check if first element is a PIL Image or a path
            if isinstance(clip[0], Image.Image):
                return clip
            elif isinstance(clip[0], str):
                # List of file paths
                frames = []
                for path in sorted(clip):
                    if os.path.isfile(path) and path.lower().endswith(('.jpg', '.jpeg', '.png')):
                        frames.append(Image.open(path).convert("RGB"))
                return frames
            else:
                raise ValueError(f"Unsupported list element type: {type(clip[0])}")
        
        elif isinstance(clip, str):
            if os.path.isdir(clip):
                # Directory path - load all images
                frame_files = sorted([
                    f for f in os.listdir(clip)
                    if os.path.isfile(os.path.join(clip, f)) and
                    f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])
                frames = []
                for frame_file in frame_files:
                    img_path = os.path.join(clip, frame_file)
                    frames.append(Image.open(img_path).convert("RGB"))
                return frames
            else:
                raise ValueError(f"Path is not a directory: {clip}")
        else:
            raise ValueError(f"Unsupported clip type: {type(clip)}")


# Convenience function for easy usage
def is_interesting(
    clip: Union[List[Image.Image], List[str], str],
    model_name: str = HF_MODEL_NAME,
    device: str = DEVICE,
    frame_sample_rate: int = DEFAULT_FRAME_SAMPLE_RATE,
    aggregation_method: str = DEFAULT_AGGREGATION_METHOD,
    top_k: Optional[int] = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> int:
    """
    Determine if a clip is interesting.
    
    This is a convenience function that creates a classifier and runs inference.
    For multiple clips, it's more efficient to create a classifier once and reuse it.
    
    Args:
        clip: Can be:
            - List of PIL Images
            - List of image file paths
            - Directory path containing frame images
        model_name: HuggingFace model name for VideoMAE
        device: Device to run inference on
        frame_sample_rate: Sample every Nth frame (1 = all frames)
        aggregation_method: "mean" or "top_k" for aggregating distances
        top_k: Number of top distances to use (only for top_k aggregation)
        threshold: Threshold for binary classification (score >= threshold -> interesting)
    
    Returns:
        1 if clip is interesting, 0 otherwise
    """
    classifier = InterestingClipClassifier(
        model_name=model_name,
        device=device,
        frame_sample_rate=frame_sample_rate,
        aggregation_method=aggregation_method,
        top_k=top_k,
        threshold=threshold,
    )
    return classifier.is_interesting(clip)


if __name__ == "__main__":
    """
    Example usage and testing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test interesting clip classifier")
    parser.add_argument(
        "--clip",
        type=str,
        required=True,
        help="Path to clip (directory with frames or list of frame paths)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Threshold for binary classification"
    )
    parser.add_argument(
        "--frame-sample-rate",
        type=int,
        default=DEFAULT_FRAME_SAMPLE_RATE,
        help="Sample every Nth frame"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["mean", "top_k"],
        default=DEFAULT_AGGREGATION_METHOD,
        help="Aggregation method"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top distances to use (for top_k aggregation)"
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Output raw score instead of binary prediction"
    )
    
    args = parser.parse_args()
    
    # Create classifier
    classifier = InterestingClipClassifier(
        threshold=args.threshold,
        frame_sample_rate=args.frame_sample_rate,
        aggregation_method=args.aggregation,
        top_k=args.top_k,
    )
    
    # Process clip
    if args.score_only:
        score = classifier.compute_score(args.clip)
        print(f"Score: {score:.6f}")
    else:
        prediction = classifier.is_interesting(args.clip)
        score = classifier.compute_score(args.clip)
        print(f"Prediction: {prediction} (interesting={prediction==1})")
        print(f"Score: {score:.6f}")
        print(f"Threshold: {args.threshold:.6f}")

