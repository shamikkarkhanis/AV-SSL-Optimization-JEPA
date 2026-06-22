"""Tests for the no-training baseline novelty functions (A/B)."""

import numpy as np

from jepa.baselines import embedding_density_novelty, masked_clean_novelty


def test_masked_clean_novelty_is_zero_for_identical_embeddings():
    emb = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    novelty = masked_clean_novelty(emb, emb)
    assert np.allclose(novelty, 0.0, atol=1e-9)


def test_masked_clean_novelty_is_scale_invariant():
    clean = np.array([[3.0, 4.0]])
    masked = np.array([[6.0, 8.0]])  # same direction, different magnitude
    novelty = masked_clean_novelty(clean, masked)
    assert np.allclose(novelty, 0.0, atol=1e-9)


def test_masked_clean_novelty_orthogonal_is_one():
    clean = np.array([[1.0, 0.0]])
    masked = np.array([[0.0, 1.0]])
    novelty = masked_clean_novelty(clean, masked)
    assert np.allclose(novelty, 1.0, atol=1e-9)


def test_embedding_density_ranks_outlier_highest():
    # Three clips clustered tightly + one far-away outlier.
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.97, 0.03],
            [-1.0, 0.0],  # outlier, opposite direction
        ]
    )
    novelty = embedding_density_novelty(embeddings, k=2)
    assert int(np.argmax(novelty)) == 3
    # The three clustered clips should all be less novel than the outlier.
    assert novelty[3] > novelty[:3].max()


def test_embedding_density_handles_degenerate_small_pool():
    assert embedding_density_novelty(np.array([[1.0, 0.0]]), k=5).shape == (1,)
    two = embedding_density_novelty(np.array([[1.0, 0.0], [0.0, 1.0]]), k=5)
    assert two.shape == (2,)
    # Orthogonal pair: each is maximally novel relative to the other.
    assert np.allclose(two, 1.0, atol=1e-9)


def test_embedding_density_k_caps_at_pool_size():
    embeddings = np.random.RandomState(0).randn(4, 8)
    # k larger than n-1 should not raise and should match k=n-1.
    big_k = embedding_density_novelty(embeddings, k=99)
    exact = embedding_density_novelty(embeddings, k=3)
    assert np.allclose(big_k, exact)
