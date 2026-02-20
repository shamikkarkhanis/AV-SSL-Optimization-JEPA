"""JEPA Evaluation Module"""

from .metrics import compute_cosine_similarity, compute_novelty_score, l2_normalize
from .inference_cost import (
    compute_energy_joules,
    derive_summary_output_path,
    distribution_stats,
)

__all__ = [
    "compute_cosine_similarity",
    "compute_novelty_score",
    "l2_normalize",
    "compute_energy_joules",
    "derive_summary_output_path",
    "distribution_stats",
]
