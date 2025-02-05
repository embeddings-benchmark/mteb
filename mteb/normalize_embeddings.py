from __future__ import annotations

import numpy as np
import torch


def normalize_embeddings_to_numpy(
    embeddings: torch.Tensor | np.ndarray | list[np.ndarray] | list[torch.Tensor],
) -> np.ndarray:
    """Normalize embeddings to be numpy arrays


    Args:
        embeddings: embeddings to normalize

    Returns:
        Normalized embeddings
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach().float().numpy()
    elif isinstance(embeddings, list):
        if isinstance(embeddings[0], torch.Tensor):
            embeddings = [
                embedding.cpu().detach().float().numpy() for embedding in embeddings
            ]
        elif isinstance(embeddings[0], np.ndarray):
            embeddings = embeddings

    numpy_embeddings = np.array(embeddings)

    return numpy_embeddings
